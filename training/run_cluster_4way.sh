#!/bin/bash
# 4-way clustering + distillation training on GPU 0,1
# Single group: all 24 layers sequentially on cuda:0,cuda:1

set -e

PYTHON=python
SCRIPT=train_ununited_experts_grpo.py
MODEL_PATH=/root/hujianmin/LLMs/Qwen1.5-MoE-A2.7B-Chat
GROUPING_MAP=grouping_4way_sequential.json
CALIB_JSONL=obqa_calib.jsonl
OUTPUT_DIR=4_way_united_experts_obqa-GRPO
LOG_DIR=${OUTPUT_DIR}/logs

mkdir -p ${LOG_DIR}

echo "============================================"
echo "4-Way Clustering + Distillation Training"
echo "============================================"
echo "GPU: cuda:0,cuda:1"
echo "Output: ${OUTPUT_DIR}/"
echo "============================================"

CUDA_VISIBLE_DEVICES=0,1 $PYTHON $SCRIPT \
    --grouping_map ${GROUPING_MAP} \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR}/group_a \
    --devices cuda:0,cuda:1 \
    --way 4 \
    --calib_jsonl ${CALIB_JSONL} \
    --calib_max_samples 256 \
    --calib_max_length 256 \
    --distill_steps 2000 \
    --distill_lr 1e-4 \
    --early_stop_patience 300 \
    --early_stop_min_delta 1e-7 \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
    --seed 42 \
    2>&1 | tee ${LOG_DIR}/training.log

echo ""
echo "Merging weight files..."
$PYTHON -c "
import torch, os, json

output_dir = '${OUTPUT_DIR}'
group_dir = '${OUTPUT_DIR}/group_a'

all_weights = {}
for fname in ['w1.pth', 'w2.pth', 'w3.pth']:
    path = os.path.join(group_dir, fname)
    if os.path.exists(path):
        d = torch.load(path, map_location='cpu')
        all_weights.update(d)
        print(f'  Loaded {path}: {len(d)} tensors')

# Split into w1/w2/w3 (layers 0-7, 8-15, 16-23)
num_groups = 15
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

# Copy grouping map
src = '${GROUPING_MAP}'
dst = os.path.join(output_dir, 'grouping_map.json')
with open(src) as f:
    gm = json.load(f)
with open(dst, 'w') as f:
    json.dump(gm, f, indent=2)
print(f'  Saved grouping_map.json')

print(f'Done! Final weights in {output_dir}/')
"

echo "Training completed!"
