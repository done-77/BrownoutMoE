#!/usr/bin/env python3
"""
Train United Experts using GRPO Grouping Map.

Reads the GRPO grouping output JSON (expert→group mapping),
then trains each united expert via MSE distillation against
the weighted average of its group members' outputs.

Outputs weights in the same format as the original train.py
expects (w1.pth, w2.pth, w3.pth) with keys:
  {layer_id}_{group_id}_up_proj.weight
  {layer_id}_{group_id}_gate_proj.weight
  {layer_id}_{group_id}_down_proj.weight

Usage:
    python train_united_experts_grpo.py \
        --grouping_map runs/true_grpo/grouping_all.json \
        --model_path /root/llm-resource/Models/Qwen1.5-MoE-A2.7B-Chat \
        --output_dir ./8_way_united_experts_grpo \
        --devices cuda:0,cuda:1,cuda:2,cuda:3 \
        --way 8 \
        --distill_steps 2000 \
        --save_every 200
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


from rl_grouping_true_grpo import way_to_num_groups


# ---------------------------------------------------------------------------
# Target Model: SwiGLU FFN (same architecture as train.py)
# ---------------------------------------------------------------------------
class TargetModel(nn.Module):
    def __init__(self, hidden_size=2048, intermediate_size=1408):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Expert forward (no nn.Module,# ---------------------------------------------------------------------------
def expert_forward(gate_w, up_w, down_w, x):
    """Single SwiGLU expert forward using raw weight tensors."""
    gate = torch.nn.functional.linear(x, gate_w)
    up = torch.nn.functional.linear(x, up_w)
    return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down_w)


# ---------------------------------------------------------------------------
# Load GRPO grouping map
# ---------------------------------------------------------------------------
def load_grouping_map(path: str) -> Dict[int, List[int]]:
    """Load grouping map from GRPO output JSON.

    Returns: {layer_id: [group_assignment per expert]}
    grouping[layer_id][e] = group_id for expert e
    """
    with open(path, "r") as f:
        data = json.load(f)
    layers = data["layers"]
    way = data.get("way", data.get("meta", {}).get("way", 4))
    num_experts = data.get("num_experts", 60)
    num_groups = data.get("num_groups", way_to_num_groups(num_experts, way))

    result = {}
    for lid_str, mapping in layers.items():
        lid = int(lid_str)
        result[lid] = [int(mapping[str(e)]) for e in range(num_experts)]
    return result


def get_group_members(grouping: List[int], num_groups: int) -> List[List[int]]:
    """Get list of expert IDs in each group."""
    groups = [[] for _ in range(num_groups)]
    for e, g in enumerate(grouping):
        groups[g].append(e)
    return groups


# ---------------------------------------------------------------------------
# Collect calibration inputs for a layer
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_layer_inputs(
    model,
    tokenizer,
    layer_id: int,
    calib_jsonl: str,
    max_samples: int,
    max_length: int,
    device: str,
    seed: int = 42,
) -> torch.Tensor:
    """Run calibration data through model and hook the MoE layer input."""
    import random
    random.seed(seed)
    texts = []
    with open(calib_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            if len(texts) >= max_samples:
                break
    random.shuffle(texts)

    all_x = []
    hook_store = []
    moe_module = model.model.layers[layer_id].mlp

    def hook_fn(_mod, inp, _out):
        hook_store.append(inp[0].detach())

    handle = moe_module.register_forward_hook(hook_fn)
    for text in texts:
        ids = tokenizer(
            text, return_tensors="pt", max_length=max_length, truncation=True
        ).input_ids.to(device)
        model(ids)
        if hook_store:
            all_x.append(hook_store[-1].squeeze(0))  # [1, seq_len, H] -> [seq_len, H]
            hook_store.clear()
    handle.remove()

    result = torch.cat(all_x, dim=0)  # [total_tokens, H]
    print(f"  Calib data for layer {layer_id}: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Get expert weights for a layer
# ---------------------------------------------------------------------------
def get_expert_weights(model, layer_id: int, num_experts: int, device: str):
    """Extract gate/up/down weight tensors for all experts in a layer.

    Qwen2MoE uses fused gate_up_proj [num_experts, 2*intermediate, hidden],
    so we split it into gate_proj and up_proj each [intermediate, hidden].
    """
    experts_mod = model.model.layers[layer_id].mlp.experts
    gate_up_w = experts_mod.gate_up_proj.detach().to(device).float()  # [E, 2*I, H]
    down_w = experts_mod.down_proj.detach().to(device).float()        # [E, H, I]

    intermediate_size = down_w.shape[2]  # 1408
    gate_ws = [gate_up_w[e, :intermediate_size, :] for e in range(num_experts)]
    up_ws   = [gate_up_w[e, intermediate_size:, :] for e in range(num_experts)]
    down_ws = [down_w[e, :, :] for e in range(num_experts)]
    return gate_ws, up_ws, down_ws


# ---------------------------------------------------------------------------
# Train one layer's united experts
# ---------------------------------------------------------------------------
def train_layer(
    layer_id: int,
    grouping: List[int],
    gate_ws: List[torch.Tensor],
    up_ws: List[torch.Tensor],
    down_ws: List[torch.Tensor],
    calib_x: torch.Tensor,
    num_groups: int,
    num_experts: int,
    train_device: str,
    args,
) -> Dict[str, torch.Tensor]:
    """Train united experts for one layer given a grouping map.

    Returns: dict of {key: tensor} for saving.
    """
    groups = get_group_members(grouping, num_groups)

    # Move everything to train_device
    gate_ws = [w.to(train_device) for w in gate_ws]
    up_ws = [w.to(train_device) for w in up_ws]
    down_ws = [w.to(train_device) for w in down_ws]
    x = calib_x.to(train_device).float()

    weight_dict = {}
    pbar = tqdm(groups, desc=f"  Layer {layer_id} groups")

    for gid, members in enumerate(pbar):
        if len(members) <= 1:
            # Single expert: just copy its weights directly
            e = members[0]
            weight_dict[f"{layer_id}_{gid}_gate_proj.weight"] = gate_ws[e].clone()
            weight_dict[f"{layer_id}_{gid}_up_proj.weight"] = up_ws[e].clone()
            weight_dict[f"{layer_id}_{gid}_down_proj.weight"] = down_ws[e].clone()
            continue

        # Compute target: mean of member experts' outputs (x already on train_device)
        x = x.float()
        with torch.no_grad():
            tgt = None
            for e in members:
                y = expert_forward(gate_ws[e], up_ws[e], down_ws[e], x)
                if tgt is None:
                    tgt = y / len(members)
                else:
                    tgt = tgt + y / len(members)

        # Initialize united expert
        target_model = TargetModel().to(train_device).float()

        # Initialize weights as average of member experts
        with torch.no_grad():
            target_model.gate_proj.weight.copy_(
                torch.stack([gate_ws[e] for e in members]).mean(0)
            )
            target_model.up_proj.weight.copy_(
                torch.stack([up_ws[e] for e in members]).mean(0)
            )
            target_model.down_proj.weight.copy_(
                torch.stack([down_ws[e] for e in members]).mean(0)
            )

        optimizer = optim.Adam(target_model.parameters(), lr=args.distill_lr)
        criterion = nn.MSELoss()

        # Training loop with early stopping
        target_model.train()
        best_loss = float("inf")
        best_state = None
        no_improve_count = 0

        for step in range(1, args.distill_steps + 1):
            output = target_model(x)
            loss = criterion(output, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss - args.early_stop_min_delta:
                best_loss = loss_val
                best_state = {k: v.clone() for k, v in target_model.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1

            if step % 200 == 0 or step == 1:
                print(f"    group {gid} (experts {members}): step {step}/{args.distill_steps}, "
                      f"loss={loss_val:.6f}, best={best_loss:.6f}, no_improve={no_improve_count}")

            # Early stopping
            if no_improve_count >= args.early_stop_patience:
                print(f"    group {gid} (experts {members}): EARLY STOP at step {step}, "
                      f"no improvement for {no_improve_count} steps")
                break

        # Save final best weights
        if best_state is not None:
            target_model.load_state_dict(best_state)

        target_model.eval()
        with torch.no_grad():
            final_out = target_model(x)
            final_loss = criterion(final_out, tgt).item()
        print(f"    group {gid} (experts {members}): final_loss={final_loss:.6f}, "
              f"best_loss={best_loss:.6f}, stopped_at_step={step}")

        # Save as fp16
        sd = target_model.half().state_dict()
        weight_dict[f"{layer_id}_{gid}_gate_proj.weight"] = sd["gate_proj.weight"]
        weight_dict[f"{layer_id}_{gid}_up_proj.weight"] = sd["up_proj.weight"]
        weight_dict[f"{layer_id}_{gid}_down_proj.weight"] = sd["down_proj.weight"]

        del target_model, optimizer
        torch.cuda.empty_cache()

    return weight_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Train United Experts using GRPO Grouping Map"
    )
    ap.add_argument("--grouping_map", required=True,
        help="Path to GRPO grouping map JSON")
    ap.add_argument("--model_path", required=True,
        help="Path to Qwen1.5-MoE-A2.7B-Chat model")
    ap.add_argument("--output_dir", required=True,
        help="Output directory for weight files (w1.pth, w2.pth, w3.pth)")
    ap.add_argument("--devices", required=True,
        help="Comma-separated GPU devices for training (e.g. cuda:0,cuda:1)")
    ap.add_argument("--way", type=int, required=True, choices=[2, 4, 8],
        help="Grouping way: 2, 4, or 8")
    ap.add_argument("--calib_jsonl", required=True,
        help="Calibration data in JSONL format")
    ap.add_argument("--distill_steps", type=int, default=2000,
        help="Distillation training steps per united expert")
    ap.add_argument("--distill_lr", type=float, default=1e-4,
        help="Distillation learning rate")
    ap.add_argument("--calib_max_samples", type=int, default=256,
        help="Max calibration samples")
    ap.add_argument("--calib_max_length", type=int, default=256,
        help="Max token length per sample")
    ap.add_argument("--save_every", type=int, default=200,
        help="Save best weights every N steps")
    ap.add_argument("--layers", type=str, default="",
        help="Comma-separated layer IDs to train (default: all layers with grouping data)")
    ap.add_argument("--early_stop_patience", type=int, default=300,
        help="Early stop after N steps without improvement (0=disabled)")
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-7,
        help="Minimum loss improvement to reset patience counter")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    devices = [d.strip() for d in args.devices.split(",")]
    train_device = devices[0]

    # Load grouping map
    grouping_map = load_grouping_map(args.grouping_map)

    if args.layers:
        target_layers = [int(x) for x in args.layers.split(",")]
    else:
        target_layers = sorted(grouping_map.keys())

    num_groups = way_to_num_groups(60, args.way)
    print(f"way={args.way}, num_groups={num_groups}")
    print(f"Layers to train: {target_layers}")
    print(f"Devices: {devices}")

    # Load model (on first device)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = model.to(devices[-1])  # Load on last device to save train device memory
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )

    # Collect calibration inputs once (share across layers)
    # We collect per-layer later; for now just set up
    torch.manual_seed(args.seed)

    all_weight_dict = {}

    for layer_id in target_layers:
        if layer_id not in grouping_map:
            print(f"Skipping layer {layer_id}: no grouping data")
            continue

        grouping = grouping_map[layer_id]
        print(f"\n{'='*60}")
        print(f"Training layer {layer_id} ({num_groups} groups, way={args.way})")
        print(f"{'='*60}")

        # Free memory: delete previous layer's expert weights if any
        torch.cuda.empty_cache()

        # Collect calibration inputs for this layer
        calib_x = collect_layer_inputs(
            model, tokenizer, layer_id,
            args.calib_jsonl, args.calib_max_samples, args.calib_max_length,
            devices[-1], args.seed,
        )

        # Get expert weights
        gate_ws, up_ws, down_ws = get_expert_weights(
            model, layer_id, 60, devices[-1]
        )

        # Train
        wdict = train_layer(
            layer_id, grouping, gate_ws, up_ws, down_ws,
            calib_x, num_groups, 60, train_device, args,
        )
        all_weight_dict.update(wdict)

        del gate_ws, up_ws, down_ws
        torch.cuda.empty_cache()

    # Split into w1/w2/w3 and save
    # layer 0-7 → w1, layer 8-15 → w2, layer 16-23 → w3
    train_layers_set = set(target_layers)
    os.makedirs(args.output_dir, exist_ok=True)

    for chunk_id, (start, end) in enumerate([(0, 8), (8, 16), (16, 24)]):
        chunk_dict = {}
        for lid in range(start, end):
            if lid not in train_layers_set:
                continue
            for gid in range(num_groups):
                for wtype in ["gate_proj", "up_proj", "down_proj"]:
                    key = f"{lid}_{gid}_{wtype}.weight"
                    if key in all_weight_dict:
                        chunk_dict[key] = all_weight_dict[key]
        if chunk_dict:
            out_path = os.path.join(args.output_dir, f"w{chunk_id + 1}.pth")
            torch.save(chunk_dict, out_path)
            print(f"Saved {out_path} ({len(chunk_dict)} weight tensors)")

    print(f"\nDone! Weights saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
