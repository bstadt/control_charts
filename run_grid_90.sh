#!/bin/bash
cd ~/root/controlcharts
BATCH_SIZE=20
count=0

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-10.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-11.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-12.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-13.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-14.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-15.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-16.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-17.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-18.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-19.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-20.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-21.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-22.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-23.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-24.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-25.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-26.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-27.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-28.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-29.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-30.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-31.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-32.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-33.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-34.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-35.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-36.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-37.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-38.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-39.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-40.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-41.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-42.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-43.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-44.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-45.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-46.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-47.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-48.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-49.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-50.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-51.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-52.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-53.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-54.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-55.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-56.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-57.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-58.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-59.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-60.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-61.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-62.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-63.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-64.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-65.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-66.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-67.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-68.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-69.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-70.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-71.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-72.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-73.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-74.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-75.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-76.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-77.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-78.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-79.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-80.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-81.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-82.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-83.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-84.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-85.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-86.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-87.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-88.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-89.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-90.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-91.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-92.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-93.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-94.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-95.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-96.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-97.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-98.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-noadv-99.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-10.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-11.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-12.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-13.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-14.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-15.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-16.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-17.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-18.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-19.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-20.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-21.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-22.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-23.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-24.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-25.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-26.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-27.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-28.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-29.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-30.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-31.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-32.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-33.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-34.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-35.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-36.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-37.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-38.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-39.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-40.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-41.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-42.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-43.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-44.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-45.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-46.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-47.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-48.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-49.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-50.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-51.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-52.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-53.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-54.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-55.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-56.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-57.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-58.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-59.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-60.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-61.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-62.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-63.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-64.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-65.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-66.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-67.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-68.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-69.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-70.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-71.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-72.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-73.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-74.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-75.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-76.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-77.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-78.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-79.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-80.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-81.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-82.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-83.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-84.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-85.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-86.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-87.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-88.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-89.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-90.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-91.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-92.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-93.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-94.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-95.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-96.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-97.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-98.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d100-99.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-10.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-11.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-12.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-13.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-14.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-15.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-16.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-17.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-18.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-19.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-20.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-21.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-22.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-23.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-24.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-25.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-26.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-27.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-28.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-29.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-30.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-31.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-32.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-33.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-34.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-35.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-36.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-37.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-38.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-39.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-40.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-41.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-42.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-43.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-44.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-45.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-46.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-47.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-48.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-49.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-50.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-51.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-52.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-53.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-54.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-55.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-56.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-57.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-58.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-59.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-60.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-61.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-62.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-63.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-64.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-65.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-66.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-67.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-68.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-69.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-70.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-71.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-72.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-73.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-74.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-75.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-76.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-77.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-78.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-79.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-80.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-81.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-82.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-83.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-84.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-85.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-86.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-87.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-88.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-89.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-90.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-91.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-92.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-93.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-94.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-95.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-96.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-97.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-98.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

.venv/bin/python -m controlcharts.cli run experiments/configs/grid-search-d800-99.yaml &
count=$((count+1))
if [ $((count % BATCH_SIZE)) -eq 0 ]; then
  echo "Waiting for batch $((count/BATCH_SIZE))..."
  wait
  echo "Batch done. Starting next..."
fi

wait
echo "All 270 simulations complete!"
