"""Agent implementation with RAG database and LLM backend."""

from dataclasses import dataclass, field
import numpy as np
from openai import OpenAI

from .config import DEFAULT_SYSTEM_PROMPT, DEFAULT_PROMPT_TEMPLATE
from .database import VectorDatabase, QAPair


@dataclass
class Agent:
    """An agent with a RAG database and LLM backend."""

    id: int
    database: VectorDatabase
    model: str = "gpt-4o-mini"
    retrieval_k: int = 5
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE

    # Forget strategy configuration
    forget_strategy: str = "none"  # "none" or "decay"
    decay_coefficient: float = 0.1  # Exponential decay rate

    # Question tracking
    questions_in_play: set[str] = field(default_factory=set)
    known_questions: set[str] = field(default_factory=set)

    # Temporal question ownership - agent owns these temporal questions
    # and always knows the correct answer (current value from shared state)
    owned_temporal_questions: set[str] = field(default_factory=set)
    temporal_questions: set[str] = field(default_factory=set)  # All temporal questions in play
    temporal_values: dict[str, int] = field(default_factory=dict)  # Shared dict: question -> current value

    # Track when questions were last learned (for decay strategy)
    # Maps question -> iteration when it was learned
    question_learn_time: dict[str, int] = field(default_factory=dict)
    current_iteration: int = 0

    # OpenAI client (initialized lazily)
    _client: OpenAI | None = field(default=None, repr=False)

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    @property
    def unknown_questions(self) -> set[str]:
        """Questions in play that this agent doesn't know."""
        return self.questions_in_play - self.known_questions

    def initialize_knowledge(self, qa_pairs: list[QAPair]) -> None:
        """Initialize agent with a set of QA pairs, marking them as known."""
        self.database.add_many(qa_pairs)
        for qa in qa_pairs:
            self.known_questions.add(qa.question)
            self.question_learn_time[qa.question] = 0  # Learned at iteration 0

    def set_iteration(self, iteration: int) -> None:
        """Set the current iteration (for decay calculations)."""
        self.current_iteration = iteration

    def set_questions_in_play(self, questions: set[str]) -> None:
        """Set the questions that are in play for this simulation."""
        self.questions_in_play = questions

    def set_temporal_questions(self, temporal_questions: set[str], owned: set[str], temporal_values: dict[str, int]) -> None:
        """Set the temporal questions in play and which ones this agent owns.

        Args:
            temporal_questions: All temporal questions in the simulation
            owned: The temporal questions this agent owns (always knows correct answer)
            temporal_values: Shared dict mapping question -> current value (updated by simulation)
        """
        self.temporal_questions = temporal_questions
        self.owned_temporal_questions = owned
        self.temporal_values = temporal_values  # Reference to shared state
        # Agent always "knows" their owned temporal questions
        for q in owned:
            self.known_questions.add(q)

    def mark_known(self, question: str) -> None:
        """Mark a question as known."""
        self.known_questions.add(question)

    def select_question_to_ask(self, rng: np.random.Generator) -> str | None:
        """Randomly select a question to ask a peer.

        In 'none' mode: only asks unknown questions.
        In 'decay' mode: can re-ask known questions with probability
        that increases as time since learning increases.
        """
        if self.forget_strategy == "none":
            # Original behavior: only ask unknown questions
            unknown = list(self.unknown_questions)
            if not unknown:
                return None
            return rng.choice(unknown)

        elif self.forget_strategy == "decay":
            # Decay mode: build candidate pool with decay-based probabilities
            candidates = []
            weights = []

            for question in self.questions_in_play:
                if question in self.known_questions:
                    # Known question: probability to ask based on time since learned
                    time_since_learned = self.current_iteration - self.question_learn_time.get(question, 0)
                    # P = 1 - exp(-decay_coefficient * time)
                    prob = 1.0 - np.exp(-self.decay_coefficient * time_since_learned)
                    # Only include if we "pass" the probability check
                    if rng.random() < prob:
                        candidates.append(question)
                        weights.append(prob)
                else:
                    # Unknown question: always a candidate with weight 1.0
                    candidates.append(question)
                    weights.append(1.0)

            if not candidates:
                return None

            # Normalize weights and select
            weights = np.array(weights)
            weights = weights / weights.sum()
            return rng.choice(candidates, p=weights)

        else:
            raise ValueError(f"Unknown forget strategy: {self.forget_strategy}")

    def answer(self, question: str, question_embedding: np.ndarray) -> str:
        """Answer a question using RAG retrieval and LLM.

        For temporal questions that this agent owns, returns the current
        value from the shared temporal_values dict (agent always knows
        the correct answer for their assigned temporal questions).
        """
        # Check if this is a temporal question we own - return current value directly
        if question in self.owned_temporal_questions:
            temporal_value = self.temporal_values.get(question, 0)
            return str(temporal_value)

        # Retrieve relevant QA pairs (with decay discounting if in decay mode)
        if self.forget_strategy == "decay":
            retrieved = self.database.search(
                question_embedding,
                k=self.retrieval_k,
                current_iteration=self.current_iteration,
                decay_coefficient=self.decay_coefficient
            )
        else:
            retrieved = self.database.search(question_embedding, k=self.retrieval_k)

        # Format context
        if retrieved:
            context_parts = []
            for qa in retrieved:
                context_parts.append(f"Q: {qa.question}\nA: {qa.answer}")
            context = "\n\n".join(context_parts)
        else:
            context = "(No relevant information found)"

        # Build prompt
        user_prompt = self.prompt_template.format(
            retrieved_context=context,
            question=question
        )

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    def receive_answer(self, question: str, answer: str, question_embedding: np.ndarray) -> bool:
        """Receive an answer from a peer. Returns True if knowledge was added."""
        # Check if this is a valid answer (not "I don't know")
        if self._is_dont_know(answer):
            return False

        # Check if this is a temporal question
        is_temporal = question in self.temporal_questions

        # Always insert (no duplicate checking per spec)
        qa_pair = QAPair(
            question=question,
            answer=answer,
            embedding=question_embedding,
            insertion_time=self.current_iteration,  # Track when inserted for decay
            is_temporal=is_temporal
        )
        self.database.add(qa_pair)
        self.mark_known(question)
        # Track when this question was learned (or re-learned in decay mode)
        self.question_learn_time[question] = self.current_iteration
        return True

    def _is_dont_know(self, answer: str) -> bool:
        """Check if an answer is a 'don't know' response."""
        lower = answer.lower().strip()
        dont_know_phrases = [
            "i don't know",
            "i do not know",
            "i dont know",
            "i don't have",
            "i do not have",
            "no relevant",
            "cannot answer",
            "can't answer",
            "unable to answer",
            "not enough information",
        ]
        return any(phrase in lower for phrase in dont_know_phrases)

    def get_state(self) -> dict:
        """Get agent state for logging/debugging."""
        # Count known temporal vs non-temporal questions
        known_temporal = len(self.known_questions & self.temporal_questions)
        known_nontemporal = len(self.known_questions) - known_temporal

        return {
            "id": self.id,
            "db_size": len(self.database),
            "known_count": len(self.known_questions),
            "unknown_count": len(self.unknown_questions),
            "in_play_count": len(self.questions_in_play),
            "known_temporal": known_temporal,
            "known_nontemporal": known_nontemporal,
            "owned_temporal_count": len(self.owned_temporal_questions),
        }
