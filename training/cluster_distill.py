#!/usr/bin/env python3
"""
Clustering-based expert grouping + United Expert Distillation.

For each MoE layer:
  1. Collect calibration activations from training data
  2. Extract 60 experts' weights (fused gate_up_proj → split gate/up)
  3. Compute expert similarity matrix (output cosine similarity)
  4. Hierarchical clustering → optimal grouping
  5. Full distillation: train united expert for each group (2000 steps, early stopping)
  6. Save grouping map + united expert weights

Usage:
    python train_cluster_distill.py \
        --model_path /root/hujianmin/LLMs/Qwen1.5-MoE-A2.7B-Chat \
        --calib_jsonl obqa_calib.jsonl \
        --routing_stats data/routing_counts.json \
        --output_dir ./8_way_united_experts_obqa-GRPO \
        --devices cuda:0,cuda:1 \
        --way 8 \
        --layers 0,1,2,...,11 \
        --seed 42
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.cluster.hierarchy import fcluster, linkage
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from collections import OrderedDict


# ---------------------------------------------------------------------------
NUM_EXPERTS = 60


def way_to_num_groups(num_experts: int, way: int) -> int:
    return math.ceil(num_experts / way)


# ---------------------------------------------------------------------------
# Target Model: SwiGLU FFN
# ---------------------------------------------------------------------------
class TargetModel(nn.Module):
    def __init__(self, hidden_size=2048, intermediate_size=1408):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Expert forward
# ---------------------------------------------------------------------------
def expert_forward(gate_w, up_w, down_w, x):
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    return F.linear(F.silu(gate) * up, down_w)


# ---------------------------------------------------------------------------
# Collect calibration inputs
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_all_layer_inputs(model, tokenizer, layer_ids, calib_jsonl, max_samples,
                             max_length, device, seed=42):
    random.seed(seed)
    texts = []
    with open(calib_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            if len(texts) >= max_samples:
                break
    random.shuffle(texts)

    layer_stores = {lid: [] for lid in layer_ids}
    temp_stores = {lid: [] for lid in layer_ids}
    handles = []

    for lid in layer_ids:
        moe_module = model.model.layers[lid].mlp
        def make_hook(layer_id):
            def hook_fn(_mod, inp, _out):
                temp_stores[layer_id].append(inp[0].detach().cpu())
            return hook_fn
        handles.append(moe_module.register_forward_hook(make_hook(lid)))

    for i, text in enumerate(texts):
        ids = tokenizer(
            text, return_tensors="pt", max_length=max_length, truncation=True
        ).input_ids.to(device)
        model(ids)
        for lid in layer_ids:
            if temp_stores[lid]:
                layer_stores[lid].append(temp_stores[lid][-1].squeeze(0))
                temp_stores[lid].clear()
        if (i + 1) % 500 == 0:
            print(f"  Calibration: {i+1}/{len(texts)} samples processed", flush=True)

    for h in handles:
        h.remove()

    result = {}
    for lid in layer_ids:
        if layer_stores[lid]:
            cpu_tensors = [t.cpu() for t in layer_stores[lid]]
            del layer_stores[lid]
            t = torch.cat(cpu_tensors, dim=0)
            del cpu_tensors
            result[lid] = t
            print(f"  [Layer {lid}] Calib tokens: {t.shape} (on CPU)", flush=True)
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Get expert weights
# ---------------------------------------------------------------------------
def get_expert_weights(model, layer_id, num_experts, device):
    experts_mod = model.model.layers[layer_id].mlp.experts
    gate_up_w = experts_mod.gate_up_proj.detach().to(device).float()
    down_w = experts_mod.down_proj.detach().to(device).float()

    intermediate_size = down_w.shape[2]
    gate_ws = [gate_up_w[e, :intermediate_size, :] for e in range(num_experts)]
    up_ws = [gate_up_w[e, intermediate_size:, :] for e in range(num_experts)]
    down_ws = [down_w[e, :, :] for e in range(num_experts)]
    return gate_ws, up_ws, down_ws


# ---------------------------------------------------------------------------
# Expert similarity matrix
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_expert_similarity(gate_ws, up_ws, down_ws, calib_x, num_experts):
    device = gate_ws[0].device
    if calib_x.shape[0] > 2048:
        idx = torch.randperm(calib_x.shape[0])[:2048]
        x = calib_x[idx].to(device).float()
    else:
        x = calib_x.to(device).float()

    outputs = []
    for e in range(num_experts):
        out = expert_forward(gate_ws[e], up_ws[e], down_ws[e], x)
        outputs.append(out)
    outputs = torch.stack(outputs)

    flat = outputs.view(num_experts, -1)
    norm = flat.norm(dim=1, keepdim=True) + 1e-8
    flat_normed = flat / norm
    sim = (flat_normed @ flat_normed.T).cpu().numpy()
    return sim


# ---------------------------------------------------------------------------
# Hierarchical clustering with capacity-balanced post-processing
# ---------------------------------------------------------------------------
def clustering_init(similarity, num_groups):
    """Hierarchical clustering + greedy balancing to enforce capacity constraints."""
    num_experts = similarity.shape[0]
    cap = math.ceil(num_experts / num_groups)

    # Step 1: Standard hierarchical clustering
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0)
    distance = (distance + distance.T) / 2
    Z = linkage(distance[np.triu_indices(len(distance), k=1)], method="average")
    labels = fcluster(Z, t=num_groups, criterion="maxclust") - 1
    grouping = labels.tolist()

    # Step 2: Greedy balancing — move experts from oversized to undersized groups
    changed = True
    while changed:
        changed = False
        groups = get_group_members(grouping, num_groups)

        # Find oversized and undersized groups
        oversized = [(gid, len(members)) for gid, members in enumerate(groups) if len(members) > cap]
        undersized = [(gid, len(members)) for gid, members in enumerate(groups) if len(members) < cap]

        if not oversized or not undersized:
            break

        # Pick the largest oversized group and the smallest undersized group
        oversized.sort(key=lambda x: -x[1])
        undersized.sort(key=lambda x: x[1])
        from_gid = oversized[0][0]
        to_gid = undersized[0][0]

        # Find the expert in from_gid most similar to experts in to_gid
        from_members = groups[from_gid]
        to_members = groups[to_gid]

        best_expert = None
        best_sim = -1.0
        for e in from_members:
            # Average similarity to target group members
            if to_members:
                avg_sim = np.mean([similarity[e][t] for t in to_members])
            else:
                # If target group is empty, pick expert with highest avg sim to others in source
                avg_sim = np.mean([similarity[e][t] for t in from_members if t != e])
            if avg_sim > best_sim:
                best_sim = avg_sim
                best_expert = e

        if best_expert is not None:
            grouping[best_expert] = to_gid
            changed = True

    return grouping


def sequential_grouping(num_experts, num_groups):
    cap = math.ceil(num_experts / num_groups)
    return [min(e // cap, num_groups - 1) for e in range(num_experts)]


def get_group_members(grouping, num_groups):
    groups = [[] for _ in range(num_groups)]
    for e, g in enumerate(grouping):
        groups[g].append(e)
    return groups


# ---------------------------------------------------------------------------
# Distill a single united expert for one group
# ---------------------------------------------------------------------------
def distill_united_expert(members, gate_ws, up_ws, down_ws, calib_x, routing_weights,
                          distill_steps, distill_lr, early_stop_patience, early_stop_min_delta):
    device = gate_ws[0].device
    x = calib_x.to(device).float()

    # Routing-weighted target
    member_w = torch.tensor(
        [float(routing_weights[e]) for e in members], device=device
    )
    member_w = member_w / (member_w.sum() + 1e-12)

    with torch.no_grad():
        tgt = None
        for i, e in enumerate(members):
            y = expert_forward(gate_ws[e], up_ws[e], down_ws[e], x)
            tgt = y * member_w[i] if tgt is None else tgt + y * member_w[i]

    # Initialize as parameter average
    united_gate = torch.stack([gate_ws[e] for e in members]).mean(0).clone()
    united_up = torch.stack([up_ws[e] for e in members]).mean(0).clone()
    united_down = torch.stack([down_ws[e] for e in members]).mean(0).clone()

    gate_p = nn.Parameter(united_gate)
    up_p = nn.Parameter(united_up)
    down_p = nn.Parameter(united_down)
    opt = optim.AdamW([gate_p, up_p, down_p], lr=distill_lr, weight_decay=0.0)

    best_mse = float('inf')
    best_state = None
    patience_counter = 0

    for step in range(distill_steps):
        out = expert_forward(gate_p, up_p, down_p, x)
        loss = torch.mean((out - tgt).float().pow(2))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        cur_mse = loss.item()
        if cur_mse < best_mse - early_stop_min_delta:
            best_mse = cur_mse
            best_state = (gate_p.data.clone(), up_p.data.clone(), down_p.data.clone())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            break

    if best_state is not None:
        return best_state, best_mse
    return (gate_p.data.clone(), up_p.data.clone(), down_p.data.clone()), cur_mse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--calib_jsonl", type=str, required=True)
    parser.add_argument("--routing_stats", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:1")
    parser.add_argument("--way", type=int, default=8)
    parser.add_argument("--layers", type=str, required=True)
    parser.add_argument("--calib_max_samples", type=int, default=5507)
    parser.add_argument("--calib_max_length", type=int, default=256)
    parser.add_argument("--distill_steps", type=int, default=2000)
    parser.add_argument("--distill_lr", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=300)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    layer_ids = [int(x) for x in args.layers.split(",")]
    device_list = args.devices.split(",")
    model_device = device_list[1] if len(device_list) > 1 else device_list[0]
    train_device = device_list[0]
    num_groups = way_to_num_groups(NUM_EXPERTS, args.way)

    print("=" * 60, flush=True)
    print("Clustering-based Expert Grouping + Distillation", flush=True)
    print("=" * 60, flush=True)
    print(f"  way={args.way}, num_groups={num_groups}", flush=True)
    print(f"  layers={layer_ids}", flush=True)
    print(f"  model_device={model_device}, train_device={train_device}", flush=True)
    print(f"  calib={args.calib_jsonl} ({args.calib_max_samples} samples)", flush=True)
    print(f"  Distill: steps={args.distill_steps}, patience={args.early_stop_patience}", flush=True)
    print(f"  output={args.output_dir}", flush=True)
    print("=" * 60, flush=True)

    # Load routing weights
    routing_weights = torch.ones(NUM_EXPERTS)
    if args.routing_stats and os.path.exists(args.routing_stats):
        with open(args.routing_stats) as f:
            rs = json.load(f)
        if "layer_level" in rs:
            first_key = list(rs["layer_level"].keys())[0]
            counts = rs["layer_level"][first_key]
            total = sum(counts.values())
            routing_weights = torch.tensor(
                [counts.get(str(i), 0) / total for i in range(NUM_EXPERTS)]
            )
        print(f"  Loaded routing weights (sum={routing_weights.sum():.4f})", flush=True)

    # Load model
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(model_device)
    model.eval()

    # Collect calibration data
    print("Collecting calibration data...", flush=True)
    calib_data = collect_all_layer_inputs(
        model, tokenizer, layer_ids, args.calib_jsonl,
        args.calib_max_samples, args.calib_max_length, model_device, args.seed,
    )
    print(f"Calibration done. Collected data for {len(calib_data)} layers.", flush=True)

    # Free model GPU memory (keep model for expert weight extraction only)
    torch.cuda.empty_cache()

    # Process each layer
    all_weights = {}
    all_grouping = {}

    for layer_id in layer_ids:
        print(f"\n{'=' * 60}", flush=True)
        print(f"Layer {layer_id}", flush=True)
        print(f"{'=' * 60}", flush=True)

        calib_x = calib_data[layer_id]

        # Get expert weights
        gate_ws, up_ws, down_ws = get_expert_weights(
            model, layer_id, NUM_EXPERTS, train_device
        )

        # Compute similarity & clustering
        print(f"  Computing expert similarity matrix...", flush=True)
        sim = compute_expert_similarity(gate_ws, up_ws, down_ws, calib_x, NUM_EXPERTS)
        grouping = clustering_init(sim, num_groups)

        seq_grouping = sequential_grouping(NUM_EXPERTS, num_groups)
        groups = get_group_members(grouping, num_groups)
        print(f"  Clustering grouping: {len(groups)} groups", flush=True)
        for gid, members in enumerate(groups):
            print(f"    Group {gid}: {members}", flush=True)

        # Save grouping
        all_grouping[str(layer_id)] = {str(e): int(g) for e, g in enumerate(grouping)}

        # Distill each group
        for gid, members in enumerate(groups):
            if len(members) <= 1:
                # Single expert: just use its weights directly
                print(f"  Group {gid} ({len(members)} experts): using original weights", flush=True)
                gate_w = gate_ws[members[0]].cpu()
                up_w = up_ws[members[0]].cpu()
                down_w = down_ws[members[0]].cpu()
            else:
                print(f"  Group {gid} ({len(members)} experts): distilling...", flush=True)
                (gate_w, up_w, down_w), mse = distill_united_expert(
                    members, gate_ws, up_ws, down_ws, calib_x, routing_weights,
                    args.distill_steps, args.distill_lr,
                    args.early_stop_patience, args.early_stop_min_delta,
                )
                gate_w = gate_w.cpu()
                up_w = up_w.cpu()
                down_w = down_w.cpu()
                print(f"    MSE={mse:.8f}", flush=True)

            all_weights[f"{layer_id}_{gid}_gate_proj.weight"] = gate_w
            all_weights[f"{layer_id}_{gid}_up_proj.weight"] = up_w
            all_weights[f"{layer_id}_{gid}_down_proj.weight"] = down_w

        # Free GPU memory for this layer
        del gate_ws, up_ws, down_ws
        torch.cuda.empty_cache()

    # Save weights: per-layer directory + {layer_id}_{start_expert}.pth
    os.makedirs(args.output_dir, exist_ok=True)

    for layer_id in layer_ids:
        layer_dir = os.path.join(args.output_dir, f"layer_{layer_id}")
        os.makedirs(layer_dir, exist_ok=True)
        groups = get_group_members(
            [all_grouping[str(layer_id)][str(e)] for e in range(NUM_EXPERTS)],
            num_groups,
        )
        for gid, members in enumerate(groups):
            start_expert = members[0]  # first expert in group
            sd = OrderedDict()
            sd["gate_proj.weight"] = all_weights[f"{layer_id}_{gid}_gate_proj.weight"]
            sd["up_proj.weight"] = all_weights[f"{layer_id}_{gid}_up_proj.weight"]
            sd["down_proj.weight"] = all_weights[f"{layer_id}_{gid}_down_proj.weight"]
            out_path = os.path.join(layer_dir, f"{layer_id}_{start_expert}.pth")
            torch.save(sd, out_path)
        print(f"  Saved layer_{layer_id}/ ({num_groups} groups)", flush=True)

    # Save grouping map
    grouping_data = {
        "num_experts": NUM_EXPERTS,
        "num_groups": num_groups,
        "way": args.way,
        "layers": all_grouping,
    }
    with open(os.path.join(args.output_dir, "grouping_map.json"), "w") as f:
        json.dump(grouping_data, f, indent=2)
    print(f"  Saved grouping_map.json", flush=True)

    print(f"\nDone! All outputs in {args.output_dir}/", flush=True)


if __name__ == "__main__":
    main()
