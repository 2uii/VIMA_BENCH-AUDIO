#!/usr/bin/env bash
set -e

EPISODES=100
SEED=123

TASKS=(
  "rearrangement/rearrange"
  "require_memory/manipulate_old_neighbor"
  "require_memory/pick_in_order_then_restore"
  "require_memory/rearrange_then_restore"
  "require_reasoning/same_shape"
  "require_reasoning/same_texture"
)

mkdir -p results

echo "============================================================"
echo "RESUMING REMAINING TASKS (BASELINE vs AUDIO-VIMA)"
echo "EPISODES=$EPISODES  SEED=$SEED"
echo "============================================================"

for t in "${TASKS[@]}"; do
  echo ""
  echo "==================== BASELINE: $t ===================="
  python scripts/eval_baseline.py --task "$t" --episodes "$EPISODES" --seed "$SEED"

  echo ""
  echo "==================== AUDIO-VIMA: $t ==================="
  python scripts/eval_audio_vima.py --task "$t" --episodes "$EPISODES" --seed "$SEED"
done
