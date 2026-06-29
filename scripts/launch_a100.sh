#!/usr/bin/env bash
# GPT-OSS-Lite full A100 run launcher.
# Prepares data → runs benchmarks → starts pretraining.
#
# Hardware target: 1x A100 80GB SXM.
# Projected wall time: ~16-20h at 35-40% MFU.

set -euo pipefail

CONFIG="${CONFIG:-configs/pretrain_a100_502m.yaml}"
DATA_DIR="${DATA_DIR:-data/shards}"

echo "[launch] GPT-OSS-Lite A100 run launcher"
echo "[launch] Config: $CONFIG"
echo "[launch] Data: $DATA_DIR"

# 1. Prepare data (downloads + tokenises; can be skipped if data already exists)
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR"/shard_*.bin 2>/dev/null)" ]; then
    echo "[launch] Step 1: Prepare data via the universal pipeline..."
    python3 data/prepare_data.py --stage pretrain
    echo "[launch] See data/DATA_PIPELINE.md for the full per-project guide."
else
    echo "[launch] Step 1: Skipped (data already prepared)"
fi

# 2. Run benchmarks before training
echo "[launch] Step 2: KV-cache benchmark..."
python3 scripts/kv_cache_benchmark.py

echo "[launch] Step 3: Memory microbench..."
python3 scripts/microbench_a100.py

echo "[launch] Step 4: Step-time benchmark..."
python3 scripts/step_time_a100.py --steps 20 --warmup 5

# 3. Start training
echo "[launch] Step 5: Start pretraining..."
python3 training/pretrain.py --config "$CONFIG"

echo "[launch] Done."