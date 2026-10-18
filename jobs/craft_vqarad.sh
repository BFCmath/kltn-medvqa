#!/usr/bin/env bash
# CRAFT replication, VQA-RAD text-conflict (arbitration) pipeline, on mounted inputs only.
#
# Mounts: --model qwen-lm/qwen-3-vl/transformers/<variation>/1 --dataset phandai/vqa-rad-craft
# Env:
#   STAGES          space-separated subset of: prep filter split trace scan select ablate
#   MODEL_VARIATION Kaggle model variation (default 4b-instruct)
#   MODEL_NAME      CRAFT model key (default qwen3vl-4b)
#   FILTER_LIMIT / TRACE_LIMIT / SCAN_LIMIT   0 = all rows
#   PREV_RESULTS    optional mounted dir with a previous run's outputs/craft (to continue stages)
#
# Every stage reads and writes under $OUT_DIR/craft, so outputs come back to the control plane.
set -euo pipefail

STAGES="${STAGES:-prep filter split trace}"
MODEL_VARIATION="${MODEL_VARIATION:-4b-instruct}"
MODEL_NAME="${MODEL_NAME:-qwen3vl-4b}"
POSITION="${POSITION:-before_question}"
METRIC="${METRIC:-follow_conflict}"

MODEL_DIR=$(python -c "from medvqa.paths import find_model_dir; print(find_model_dir('$MODEL_VARIATION'))")
SRC=$(python -c "from medvqa.paths import find_dataset_dir; print(find_dataset_dir('vqa-rad-craft', 'vqa_rad_closed_all.csv'))")
IMG="$SRC/VQA_RAD_Image_Folder"
RUN="$OUT_DIR/craft"
DATA="$RUN/data"
RES="$RUN/result_${POSITION}_vqarad"
mkdir -p "$DATA" "$RES"
if [[ -n "${PREV_RESULTS:-}" && -d "$PREV_RESULTS" ]]; then
  cp -rn "$PREV_RESULTS"/. "$RUN"/
fi
echo "model=$MODEL_DIR data=$SRC stages=[$STAGES]"

COMMON=(--model "$MODEL_DIR" --model_name "$MODEL_NAME")
cd third_party/CRAFT/VQA_RAD/text_conflict
has() { [[ " $STAGES " == *" $1 "* ]]; }
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

if has prep; then
  stamp prep
  python generate_wrong_candidates.py --in_csv "$SRC/vqa_rad_closed_all.csv" --out_csv "$DATA/vqa_rad_closed_with_wrong.csv"
fi
if has filter; then
  stamp filter
  python filter_fine_grained.py --in_csv "$DATA/vqa_rad_closed_with_wrong.csv" --image_root "$IMG" "${COMMON[@]}" \
    --out_csv "$DATA/vqa_rad_nc_cc_both_correct.csv" --dtype fp16 --device_map cuda:0 \
    --resize_max_side 672 --limit "${FILTER_LIMIT:-0}"
fi
if has split; then
  stamp split
  python split_train_val.py --in_csv "$DATA/vqa_rad_nc_cc_both_correct.csv" --val_percent 20 --seed 42 \
    "${COMMON[@]}" --out_train "$DATA/train.csv" --out_val "$DATA/val.csv"
fi
if has trace; then
  stamp trace
  python layer_trace_vqarad_mm_current_fixed_fastcache.py --data_csv "$DATA/train.csv" --image_root "$IMG" \
    "${COMMON[@]}" --device cuda:0 --dtype fp16 --trace_mode conflict --position "$POSITION" \
    --metrics "$METRIC" --limit "${TRACE_LIMIT:-0}" --out "$RES/trace_conflict.json" \
    --plot_prefix "$RES/trace_conflict" --scan_plan_out "$RES/trace_conflict_scan_plan.json" \
    --scan_plan_metric "$METRIC"
fi
if has scan; then
  stamp scan
  python head_scan_vqarad_mm_fastcache.py --data_csv "$DATA/train.csv" --image_root "$IMG" "${COMMON[@]}" \
    --device cuda:0 --dtype fp16 --trace_mode conflict --position "$POSITION" --metrics "$METRIC" \
    --plan "$RES/trace_conflict_scan_plan.json" --plan_out_dir "$RES/headscan_vqarad_mm" \
    --limit "${SCAN_LIMIT:-300}"
fi
stamp done
