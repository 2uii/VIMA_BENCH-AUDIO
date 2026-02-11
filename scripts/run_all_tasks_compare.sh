#!/usr/bin/env bash
set -e

EPISODES=100
SEED=123

TASKS=(
  "constraint_satisfaction/sweep_without_exceeding"
  "constraint_satisfaction/sweep_without_touching"
  "instruction_following/rotate"
  "instruction_following/scene_understanding"
  "instruction_following/visual_manipulation"
  "novel_concept_grounding/novel_adj"
  "novel_concept_grounding/novel_adj_and_noun"
  "novel_concept_grounding/novel_noun"
  "novel_concept_grounding/twist"
  "one_shot_imitation/follow_motion"
  "one_shot_imitation/follow_order"
  "rearrangement/rearrange"
  "require_memory/manipulate_old_neighbor"
  "require_memory/pick_in_order_then_restore"
  "require_memory/rearrange_then_restore"
  "require_reasoning/same_shape"
  "require_reasoning/same_texture"
)

mkdir -p results

echo "============================================================"
echo "RUNNING ALL TASKS (BASELINE vs AUDIO-VIMA)"
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
