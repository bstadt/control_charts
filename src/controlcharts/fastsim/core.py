"""Vectorized event-level engine for the no-LLM simulation.

State is held in flat arrays over (agent, question) instead of per-agent
FAISS databases. The retrieval top-k is modeled as a per-(agent, question)
recency window of the k most recent same-question insertions, which is what
FAISS additive-decay ranking reduces to for the queried question (cosine 1.0
plus recency beats cross-question entries except for the cross-propagation
event, which the config already summarizes as a probability).

Response resolution for a query (b answers q at step t), mirroring
Agent.answer() in no-LLM mode:
  1. b owns temporal q            -> live value (checked before adversary,
                                     as in the original).
  2. b adversarial w.p. p_adv(t)  -> quine.
  3. any quine in b's window(q):
       roll < p_same              -> quine
       else freshest entry quine  -> quine (the original's exact-match
                                     fall-through returns it anyway)
       else                       -> correct (freshest correct value)
  4. no same-q quine, but b holds a quine under another question that could
     crack top-k (window underfull, or a quine inserted within
     CROSS_RECENCY steps): roll < p_cross -> quine.
  5. window non-empty             -> correct; else "I don't know".

"I don't know" responses are not inserted (receive_answer skips them).
Insertion shifts the recency window, so k fresh correct re-learns push a
quine out of the window: retrieval_k is the memory depth and the recovery
mechanism.
"""

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

IDK, CORRECT, QUINE = 0, 1, 2
QUINE_TEXT = "I lost the game"
IDK_TEXT = "I don't know"

# A quine stored under another question competes on cosine + recency; once
# its recency boost decays (~3 time constants at decay 0.05) it can no longer
# crack a full top-k window.
CROSS_RECENCY = 60

# Same-question entries are also crowded out of the FAISS top-k by the
# agent's continual fresh insertions under other questions (score =
# cosine + recency): an entry of age D scores 1 + exp(-c*D), and once that
# decays toward 1.0 a handful of fresh cross entries (cos + ~1) outrank it.
# Entries older than this are treated as not retrievable; the agent's
# re-asking (driven by the same decay) is what keeps knowledge visible.
# Calibrated against archived original runs (see validate.py); override via
# FASTSIM_SAMEQ_VIS for calibration sweeps.
import os
SAME_Q_VISIBILITY = int(os.environ.get("FASTSIM_SAMEQ_VIS", "17"))


def defection_probability(t: int, sched: dict | None) -> float:
    """Port of Agent.get_adversarial_probability (sigmoid schedule)."""
    if sched is None:
        return 0.0
    start, duration, max_p, shape = (
        sched["start"], sched["duration"], sched["max_p"], sched["shape"])
    if t <= start:
        return 0.0
    if t >= start + duration:
        return max_p
    x = shape * (2 * (t - start) / duration - 1)
    sig = 1 / (1 + math.exp(-x))
    sig_lo = 1 / (1 + math.exp(shape))
    sig_hi = 1 / (1 + math.exp(-shape))
    return max_p * (sig - sig_lo) / (sig_hi - sig_lo)


class FastSim:
    def __init__(
        self,
        n_agents: int,
        n_nontemporal: int,
        n_temporal: int,
        answers_nontemporal: list[str],
        seeded: np.ndarray,          # [n_agents, questions_per_agent] question idx, or ragged list
        retrieval_k: int,
        propagation_probability: float,
        cross_question_propagation: float,
        decay_coefficient: float,
        forget_strategy: str,
        temporal_change_probability: float,
        defection_schedules: dict[int, dict],   # agent id -> sigmoid schedule
        questions_per_turn: int,
        seed: int,
        nbr_indptr: np.ndarray | None = None,   # CSR offsets; None => full mesh
        nbr_indices: np.ndarray | None = None,  # CSR neighbor ids
    ):
        self.N = n_agents
        self.Qnt = n_nontemporal
        self.Qt = n_temporal
        self.Q = self.Qnt + self.Qt
        self.answers_nt = answers_nontemporal
        self.k = min(retrieval_k, 7)
        self.mask = (1 << self.k) - 1
        self.p_same = propagation_probability
        self.p_cross = cross_question_propagation
        self.decay_c = decay_coefficient
        self.forget = forget_strategy
        self.temporal_change_p = temporal_change_probability
        self.qpt = questions_per_turn
        self.rng = np.random.default_rng(seed)

        N, Q = self.N, self.Q
        self.occ = np.zeros((N, Q), dtype=np.uint8)        # window occupancy (0..k)
        self.qbits = np.zeros((N, Q), dtype=np.uint8)      # quine flags, bit0 = freshest
        self.learn_time = np.full((N, Q), -1, dtype=np.int32)
        self.qtime = np.full((N, Q), -(10 ** 9), dtype=np.int32)  # last quine insertion
        self.tval = np.zeros((N, self.Qt), dtype=np.int32) if self.Qt else None
        self.inf_count = np.zeros(N, dtype=np.int32)       # questions with >=1 quine
        self.last_quine_t = np.full(N, -(10 ** 9), dtype=np.int64)

        # Temporal ownership: round-robin, tq_idx % N (as in cli.py)
        self.tq_owner = (np.arange(self.Qt) % N).astype(np.int64) if self.Qt else np.zeros(0, np.int64)
        self.temporal_values = np.zeros(self.Qt, dtype=np.int64)

        # Adversaries
        self.adv_sched = dict(defection_schedules)
        self.is_adv = np.zeros(N, dtype=bool)
        for aid in self.adv_sched:
            self.is_adv[aid] = True

        # known bookkeeping (known = window non-empty OR owned temporal)
        self.known_count = np.zeros(N, dtype=np.int64)
        if self.Qt:
            np.add.at(self.known_count, self.tq_owner, 1)

        # Initial seeding: correct entries at t=0
        for a in range(N):
            qs = np.asarray(seeded[a], dtype=np.int64)
            if len(qs) == 0:
                continue
            self.occ[a, qs] = 1
            self.learn_time[a, qs] = 0
            self.known_count[a] += len(qs)

        # Contact graph (None => full mesh, uniform peer). CSR neighbor lists.
        self.nbr_indptr = nbr_indptr
        self.nbr_indices = nbr_indices
        if nbr_indptr is not None:
            self.nbr_deg = np.diff(nbr_indptr)
        else:
            self.nbr_deg = None

        self.history = []          # per-step aggregates
        self.dropped_slots = 0

    # ----- question selection (asker side) --------------------------------

    def _select_questions(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample (agent, question) pairs for this step's queries.

        Decay mode: the original builds a candidate pool including every
        unknown question (weight 1) and each known question with probability
        p = 1 - exp(-c * dt) (weight p), then samples proportionally.
        Marginally that is P(q) proportional to 1 for unknown and p^2 for
        known (inclusion x weight), which we sample by rejection with a
        uniform envelope. 'none' mode: uniform over unknown questions.
        """
        n_slots = self.N * self.qpt
        a = np.repeat(np.arange(self.N, dtype=np.int64), self.qpt)
        q = np.full(n_slots, -1, dtype=np.int64)
        pending = np.arange(n_slots)

        # Fast path: rejection sampling with a uniform envelope (marginal
        # proportional to the weights). Dominates at scale, where most
        # questions are unknown and accepted immediately.
        for _ in range(4):
            if len(pending) == 0:
                break
            cand = self.rng.integers(0, self.Q, size=len(pending))
            pa = a[pending]
            accept_p = self._weight_at(pa, cand, t)
            u = self.rng.random(len(pending))
            ok = u < accept_p
            q[pending[ok]] = cand[ok]
            pending = pending[~ok]

        # Slow path: exact weighted sampling over the full question row for
        # the remaining slots (dominant in the saturated regime, where the
        # rejection envelope rarely accepts). The original renormalizes over
        # its realized candidate pool and (almost) always asks; dropping
        # slots would bias the query rate down, so only truly-zero rows skip.
        # Vectorized categorical draw over all pending slots at once.
        if len(pending):
            pa = a[pending]
            W = self._weight_matrix(pa, t)           # [len(pending), Q]
            s = W.sum(axis=1)
            good = s > 0
            self.dropped_slots += int((~good).sum())
            if good.any():
                Wg = W[good]
                cdf = np.cumsum(Wg, axis=1)
                cdf /= cdf[:, -1:]
                u = self.rng.random(good.sum())
                picks = (cdf < u[:, None]).sum(axis=1)
                picks = np.minimum(picks, self.Q - 1)
                q[pending[good]] = picks

        filled = q >= 0
        return a[filled], q[filled]

    def _weight_matrix(self, agents: np.ndarray, t: int) -> np.ndarray:
        """Selection-weight rows for a set of (possibly repeated) agents:
        1 for unknown, p^2 for known (decay), or 1/0 unknown/known ('none')."""
        occ = self.occ[agents]                       # [A, Q]
        known = occ > 0
        if self.Qt:
            owned = self.tq_owner[None, :] == agents[:, None]   # [A, Qt]
            known[:, self.Qnt:] |= owned
        if self.forget == "decay":
            lt = self.learn_time[agents]
            dt = t - np.where(lt >= 0, lt, 0)
            p = 1.0 - np.exp(-self.decay_c * np.maximum(dt, 0))
            return np.where(known, p * p, 1.0)
        return (~known).astype(np.float64)

    def _weight_at(self, agents: np.ndarray, qs: np.ndarray, t: int) -> np.ndarray:
        """Selection weight at (agent, question) pairs: 1 for unknown,
        p^2 for known (decay mode), 0 for known ('none' mode)."""
        known = self.occ[agents, qs] > 0
        if self.Qt:
            owned = (qs >= self.Qnt) & (self.tq_owner[np.clip(qs - self.Qnt, 0, self.Qt - 1)] == agents)
            known |= owned
        if self.forget == "decay":
            lt = self.learn_time[agents, qs]
            dt = t - np.where(lt >= 0, lt, 0)
            p = 1.0 - np.exp(-self.decay_c * np.maximum(dt, 0))
            return np.where(known, p * p, 1.0)
        return np.where(known, 0.0, 1.0)

    # ----- response resolution (responder side) ---------------------------

    def _resolve(self, b: np.ndarray, q: np.ndarray, t: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Resolve responses. Returns (codes, values); values only meaningful
        for CORRECT answers to temporal questions."""
        m = len(b)
        codes = np.zeros(m, dtype=np.int8)
        values = np.full(m, -1, dtype=np.int64)

        is_t = q >= self.Qnt
        tq = np.where(is_t, q - self.Qnt, 0)

        # 1. owned temporal -> live value (before the adversary check)
        owned = is_t & (self.tq_owner[tq] == b)
        codes[owned] = CORRECT
        values[owned] = self.temporal_values[tq[owned]]
        todo = ~owned

        # 2. adversarial defection
        if self.adv_sched:
            adv_p = np.zeros(m)
            for aid, sched in self.adv_sched.items():
                adv_p[b == aid] = defection_probability(t, sched)
            defect = todo & (rng.random(m) < adv_p)
            codes[defect] = QUINE
            todo &= ~defect

        # 3-5. window lookup. A quine older than SAME_Q_VISIBILITY has been
        # crowded out of the top-k by fresh cross-question insertions;
        # correct entries stay exact-matchable (the DB never deletes, and
        # the original's re-ask cadence keeps a correct same-q entry ranked;
        # its pre-attack wrong-type fraction on temporal probes is ~0.05).
        o = self.occ[b, q]
        bits = self.qbits[b, q]
        quine_visible = (bits != 0) & (t - self.qtime[b, q] <= SAME_Q_VISIBILITY)
        freshest_quine = ((bits & 1) == 1) & quine_visible
        inf_other = self.inf_count[b] - quine_visible.astype(np.int32) > 0
        cross_can = inf_other & ((o < self.k) | (t - self.last_quine_t[b] <= CROSS_RECENCY))

        roll = rng.random(m)
        same_q_quine = todo & quine_visible & ((roll < self.p_same) | freshest_quine)
        codes[same_q_quine] = QUINE
        same_q_correct = todo & quine_visible & ~same_q_quine
        todo &= ~quine_visible

        cross_roll = rng.random(m)
        cross_quine = todo & cross_can & (cross_roll < self.p_cross)
        codes[cross_quine] = QUINE
        todo &= ~cross_quine

        correct = same_q_correct | (todo & (o > 0))
        codes[correct] = CORRECT
        if self.Qt:
            ct = correct & is_t
            values[ct] = self.tval[b[ct], tq[ct]]
        # everything else stays IDK
        return codes, values

    # ----- insertion (asker receives answer) ------------------------------

    def _insert(self, a: np.ndarray, q: np.ndarray, is_quine: np.ndarray,
                values: np.ndarray, t: int) -> None:
        if len(a) == 0:
            return
        key = a * self.Q + q
        order = np.argsort(key, kind="stable")
        a, q, is_quine, values, key = a[order], q[order], is_quine[order], values[order], key[order]

        def apply(ai, qi, isq, val):
            was_zero_bits = self.qbits[ai, qi] == 0
            was_empty = self.occ[ai, qi] == 0
            self.qbits[ai, qi] = ((self.qbits[ai, qi] << 1) | isq.astype(np.uint8)) & self.mask
            now_bits = self.qbits[ai, qi] != 0
            self.occ[ai, qi] = np.minimum(self.occ[ai, qi] + 1, self.k)
            self.learn_time[ai, qi] = t
            # infection bookkeeping (0 <-> nonzero transitions)
            delta = now_bits.astype(np.int32) - (~was_zero_bits).astype(np.int32)
            np.add.at(self.inf_count, ai, delta)
            if isq.any():
                self.last_quine_t[np.unique(ai[isq])] = t
                self.qtime[ai[isq], qi[isq]] = t
            # known bookkeeping: newly non-empty window on a not-owned pair
            newly = was_empty.copy()
            if self.Qt:
                is_t = qi >= self.Qnt
                tqi = np.where(is_t, qi - self.Qnt, 0)
                owned = is_t & (self.tq_owner[tqi] == ai)
                newly &= ~owned
                # temporal value of freshest correct entry
                ct = is_t & ~isq
                if ct.any():
                    self.tval[ai[ct], tqi[ct]] = val[ct]
            np.add.at(self.known_count, ai[newly], 1)

        # One vectorized apply per unique (agent, question), keeping the
        # freshest insert of the step (last in stable key order). Duplicate
        # (agent, question) inserts within a single step are rare (~one per
        # few hundred slots); collapsing their multi-shift to a single window
        # shift is a negligible approximation and removes a Python-level loop
        # that otherwise dominated runtime (~350 apply() calls/step).
        last = np.ones(len(key), dtype=bool)
        last[:-1] = key[1:] != key[:-1]
        apply(a[last], q[last], is_quine[last], values[last])

    def _select_peers(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pick one peer per asker. Full mesh (no graph): uniform excluding
        self. Bounded degree: a uniform random neighbor; askers that are
        isolated (degree 0) are dropped. Sets self._peer_keep to the surviving
        mask so the caller can realign the parallel question array."""
        if self.nbr_indptr is None:
            b = self.rng.integers(0, self.N - 1, size=len(a))
            b += (b >= a).astype(np.int64)
            self._peer_keep = slice(None)
            return a, b
        deg = self.nbr_deg[a]
        keep = deg > 0
        self._peer_keep = keep
        a = a[keep]
        deg = deg[keep]
        r = (self.rng.random(len(a)) * deg).astype(np.int64)   # 0..deg-1
        b = self.nbr_indices[self.nbr_indptr[a] + r]
        return a, b

    # ----- main loop -------------------------------------------------------

    def step(self, t: int) -> dict:
        # temporal question random walk
        if self.Qt and self.temporal_change_p > 0:
            bump = self.rng.random(self.Qt) < self.temporal_change_p
            self.temporal_values[bump] += 1

        a, q = self._select_questions(t)
        a, b = self._select_peers(a)
        # some askers may have been dropped (isolated nodes) -> realign q
        if len(b) != len(q):
            q = q[self._peer_keep]
        codes, values = self._resolve(b, q, t, self.rng)
        got = codes != IDK
        self._insert(a[got], q[got], (codes[got] == QUINE), values[got], t)

        rec = {
            "step": t,
            "num_queries": int(len(a)),
            "num_knowledge_added": int(got.sum()),
            "quine_responses": int((codes == QUINE).sum()),
            "mean_known": float(self.known_count.mean()),
            "infected_agents": int((self.inf_count > 0).sum()),
            "infected_pairs": int(self.inf_count.sum()),
        }
        self.history.append(rec)
        return rec

    def run(self, max_iterations: int, snapshot_cb=None, snapshot_interval: int = 0):
        milestones = {int(max_iterations * p) for p in (0.25, 0.5, 0.75, 1.0)}
        for t in range(max_iterations):
            if snapshot_cb is not None and snapshot_interval and t % snapshot_interval == 0:
                snapshot_cb(t)
            self.step(t)
            if (t + 1) in milestones:
                h = self.history[-1]
                logger.info(
                    f"step {t + 1}/{max_iterations} known/agent={h['mean_known']:.1f} "
                    f"infected_agents={h['infected_agents']} quine_resp={h['quine_responses']}")
        return self.history

    # ----- snapshot responses (no state mutation, separate rng) ------------

    def snapshot_responses(self, agent_ids: np.ndarray, probe_qs: np.ndarray,
                           t: int, rng: np.random.Generator) -> list[list[str]]:
        """Response strings, shape [len(agent_ids)][len(probe_qs)]."""
        A, P = len(agent_ids), len(probe_qs)
        bb = np.repeat(agent_ids, P)
        qq = np.tile(probe_qs, A)
        codes, values = self._resolve(bb, qq, t, rng)
        out = []
        idx = 0
        for _ in range(A):
            row = []
            for j in range(P):
                c, v, qj = codes[idx], values[idx], qq[idx]
                if c == QUINE:
                    row.append(QUINE_TEXT)
                elif c == IDK:
                    row.append(IDK_TEXT)
                elif qj >= self.Qnt:
                    row.append(str(int(v)))
                else:
                    row.append(self.answers_nt[qj])
                idx += 1
            out.append(row)
        return out
