#!/bin/bash
# True GRPO 4-way training: 24 layers split across 4 GPUs (2 groups)
# Group A (cuda:0 + cuda:1): layers 0-11, model on cuda:1, train on cuda:0
# Group B (cuda:2 + cuda:3): layers 12-23, model on cuda:3, train on cuda:2

set -e

PYTHON=python
SCRIPT=./grpo_grouping_search.py
MODEL_PATH=/root/hujianmin/LLMs/Qwen1.5-MoE-A2.7B-Chat
CALIB_JSONL=obqa_calib.jsonl
ROUTING_STATS=/root/hujianmin/qwen2_moePLUSV5/data/routing_counts_base_s256_len256_20260210.json
OUTPUT_DIR=4_way_united_experts_obqa-GRPO
LOG_DIR=${OUTPUT_DIR}/logs

mkdir -p ${LOG_DIR}

echo "============================================"
echo "True GRPO 4-Way OBQA Training"
echo "============================================"
echo "Group A: cuda:0(train) + cuda:1(model) -> layers 0-11"
echo "Group B: cuda:2(train) + cuda:3(model) -> layers 12-23"
echo "Output: ${OUTPUT_DIR}/"
echo "============================================"

# Group A: layers 0-11
echo "[Group A] Starting layers 0-11 on cuda:0,cuda:1 ..."
CUDA_VISIBLE_DEVICES=0,1 nohup $PYTHON $SCRIPT \
    --model_path ${MODEL_PATH} \
    --calib_jsonl ${CALIB_JSONL} \
    --routing_stats ${ROUTING_STATS} \
    --output_dir ${OUTPUT_DIR}/group_a \
    --devices cuda:0,cuda:1 \
    --way 4 \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
    --calib_max_samples 5507 \
    --calib_max_length 256 \
    --grpo_steps 100 \
    --batch_size 4 \
    --group_size 4 \
    --policy_lr 0.01 \
    --grpo_distill_steps 10 \
    --perturb_distill_steps 5 \
    --grpo_distill_lr 1e-3 \
    --grpo_clip_eps 0.2 \
    --grpo_epochs 3 \
    --grpo_patience 30 \
    --entropy_coef 0.01 \
    --ref_kl_coef 0.02 \
    --distill_steps 2000 \
    --distill_lr 1e-4 \
    --early_stop_patience 300 \
    --early_stop_min_delta 1e-7 \
    --seed 42 \
    > ${LOG_DIR}/group_a.log 2>&1 &
PID_A=$!

# Group B: layers 12-23
echo "[Group B] Starting layers 12-23 on cuda:2,cuda:3 ..."
CUDA_VISIBLE_DEVICES=2,3 nohup $PYTHON $SCRIPT \
    --model_path ${MODEL_PATH} \
    --calib_jsonl ${CALIB_JSONL} \
    --routing_stats ${ROUTING_STATS} \
    --output_dir ${OUTPUT_DIR}/group_b \
    --devices cuda:0,cuda:1 \
    --way 4 \
    --layers 12,13,14,15,16,17,18,19,20,21,22,23 \
    --calib_max_samples 5507 \
    --calib_max_length 256 \
    --grpo_steps 100 \
    --batch_size 4 \
    --group_size 4 \
    --policy_lr 0.01 \
    --grpo_distill_steps 10 \
    --perturb_distill_steps 5 \
    --grpo_distill_lr 1e-3 \
    --grpo_clip_eps 0.2 \
    --grpo_epochs 3 \
    --grpo_patience 30 \
    --entropy_coef 0.01 \
    --ref_kl_coef 0.02 \
    --distill_steps 2000 \
    --distill_lr 1e-4 \
    --early_stop_patience 300 \
    --early_stop_min_delta 1e-7 \
    --seed 42 \
    > ${LOG_DIR}/group_b.log 2>&1 &
PID_B=$!

echo ""
echo "Both groups running in parallel:"
echo "  Group A (PID ${PID_A}): layers 0-11"
echo "  Group B (PID ${PID_B}): layers 12-23"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/group_a.log"
echo "  tail -f ${LOG_DIR}/group_b.log"
echo ""
echo "Waiting for both to finish..."

wait $PID_A
STATUS_A=$?
echo "[Group A] Finished with status ${STATUS_A}"

wait $PID_B
STATUS_B=$?
echo "[Group B] Finished with status ${STATUS_B}"

# Merge weight files
echo ""
echo "Merging weight files..."
$PYTHON -c "
import torch, os, json

output_dir = '${OUTPUT_DIR}'
group_a = '${OUTPUT_DIR}/group_a'
group_b = '${OUTPUT_DIR}/group_b'

all_weights = {}

# Load group A weights
for fname in ['w1.pth', 'w2.pth']:
    path = os.path.join(group_a, fname)
    if os.path.exists(path):
        d = torch.load(path, map_location='cpu')
        all_weights.update(d)
        print(f'  Loaded {path}: {len(d)} tensors')

# Load group B weights
for fname in ['w1.pth', 'w2.pth', 'w3.pth']:
    path = os.path.join(group_b, fname)
    if os.path.exists(path):
        d = torch.load(path, map_location='cpu')
        all_weights.update(d)
        print(f'  Loaded {path}: {len(d)} tensors')

# Split into w1/w2/w3 (layers 0-7, 8-15, 16-23)
num_groups = 15  # 4-way: 60/4 = 15 groups
for chunk_id, (start, end) in enumerate([(0, 8), (8, 16), (16, 24)]):
    chunk_dict = {}
    for lid in range(start, end):
        for gid in range(num_groups):
            for wtype in ['gate_proj', 'up_proj', 'down_proj']:
                key = f'{lid}_{gid}_{wtype}.weight'
                if key in all_weights:
                    chunk_dict[key] = all_weights[key]
    if chunk_dict:
        out_path = os.path.join(output_dir, f'w{chunk_id + 1}.pth')
        torch.save(chunk_dict, out_path)
        print(f'  Saved {out_path} ({len(chunk_dict)} tensors)')

# Merge grouping maps
all_grouping = {'num_experts': 60, 'num_groups': 15, 'way': 4, 'layers': {}}
for grp_dir in [group_a, group_b]:
    gm = os.path.join(grp_dir, 'grouping_map.json')
    if os.path.exists(gm):
        with open(gm) as f:
            data = json.load(f)
        all_grouping['layers'].update(data['layers'])
        print(f'  Merged grouping from {gm}')

with open(os.path.join(output_dir, 'grouping_map.json'), 'w') as f:
    json.dump(all_grouping, f, indent=2)
print(f'  Saved grouping_map.json')

print(f'Done! Final weights in {output_dir}/')
"

echo ""
if [ $STATUS_A -eq 0 ] && [ $STATUS_B -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "WARNING: Some groups had errors (A=${STATUS_A}, B=${STATUS_B})"
fi
