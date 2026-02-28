#!/bin/bash
cd ~/root/controlcharts

CONFIGS=(experiments/configs/grid-search-*.yaml)
BATCH_SIZE=10
TOTAL=${#CONFIGS[@]}

echo "Running $TOTAL configs in batches of $BATCH_SIZE"

for ((i=0; i<TOTAL; i+=BATCH_SIZE)); do
    batch=("${CONFIGS[@]:i:BATCH_SIZE}")
    echo ""
    echo "=== Batch starting at index $i ==="

    pids=()
    for cfg in "${batch[@]}"; do
        name=$(basename "$cfg" .yaml)
        echo "Launching $name"
        .venv/bin/python -m controlcharts.cli run "$cfg" > "/tmp/grid-${name}.log" 2>&1 &
        pids+=($!)
    done

    echo "Waiting for ${#pids[@]} jobs..."
    for pid in "${pids[@]}"; do
        wait $pid
    done
    echo "Batch done."
done

echo ""
echo "=== ALL GRID SEARCH SIMS COMPLETE ==="
