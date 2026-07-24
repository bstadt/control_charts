#!/bin/bash
# Fan out phase-sweep cells over (N x mean_degree x seed) at controlled
# parallelism. Each cell runs sim + embed + TDKPS + classification.
# Env: NS, KS, SEEDS (space lists), OUT (dir), P (parallelism), DATA (parquet).
cd ~/root/controlcharts
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
OUT="${OUT:-/tmp/phase}"
NS="${NS:-100 1000}"
KS="${KS:-2 4 8 16 full}"
SEEDS="${SEEDS:-42}"
P="${P:-4}"
DATA="${DATA:-/home/bstadt/root/data/nq_embedded.parquet}"
export OUT DATA
PROG="$OUT/progress.txt"
mkdir -p "$OUT"; : > "$PROG"

jobs=()
for N in $NS; do for k in $KS; do for s in $SEEDS; do jobs+=("$N $k $s"); done; done; done
echo "TOTAL ${#jobs[@]} cells (P=$P)" >> "$PROG"
printf '%s\n' "${jobs[@]}" | xargs -P "$P" -I ARGS bash -c '
  set -- ARGS
  timeout 900 .venv/bin/python -m controlcharts.fastsim.phase_cell \
    --N "$1" --k "$2" --seed "$3" --data-path "$DATA" --out-dir "$OUT" \
    >> "'"$PROG"'" 2>&1 || echo "FAIL $1 $2 $3" >> "'"$PROG"'"
'
echo PHASEGRID_DONE >> "$PROG"
