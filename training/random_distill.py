#!/usr/bin/env python3
"""
Distill united experts under a RANDOM expert grouping (RL-Group ablation baseline).

This is the random-grouping counterpart of train_cluster_distill.py. It
REUSES the pure functions from train_cluster_distill (get_expert_weights,
distill_united_expert, collect_all_layer_inputs, get_group_members,
way_to_num_groups) but replaces the clustering_init() call with a grouping
loaded from a pre-generated random grouping map (see gen_random_grouping.py).

Everything else — calibration activation collection, routing-weighted
distillation target, weight saving layout (layer_{lid}/{lid}_{start}.pth +
grouping_map.json) — is identical to the clustering pipeline, so the random
baseline is directly comparable.

Usage (mirror of run_cluster_distill.sh, layers split across 2 GPU groups):
    python train_random_distill.py \
        --model_path /root/hujianmin/LLMs/Qwen1.5-MoE-A2.7B-Chat \
        --calib_jsonl obqa_calib.jsonl \
        --routing_stats data/routing_counts.json \
        --grouping_map grouping_8way_random.json \
        --output_dir ./8_way_united_experts_random \
        --devices cuda:0,cuda:1 --way 8 --layers 0,1,...,11 --seed 42
"""
import argparse
import json
import os

import torch

# Reuse pure functions from the clustering pipeline (same directory)
from train_cluster_distill import (
    NUM_EXPERTS,
    way_to_num_groups,
    get_expert_weights,
    get_group_members,
    distill_united_expert,
    collect_all_layer_inputs,
)
from collections import OrderedDict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--calib_jsonl", type=str, required=True)
    ap.add_argument("--routing_stats", type=str, default="")
    ap.add_argument(
        "--grouping_map", type=str, required=True,
        help="Pre-generated random grouping map JSON (from gen_random_grouping.py)",
    )
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--devices", type=str, default="cuda:0,cuda:1")
    ap.add_argument("--way", type=int, default=8)
    ap.add_argument("--layers", type=str, required=True)
    ap.add_argument("--calib_max_samples", type=int, default=5507)
    ap.add_argument("--calib_max_length", type=int, default=256)
    ap.add_argument("--distill_steps", type=int, default=2000)
    ap.add_argument("--distill_lr", type=float, default=1e-4)
    ap.add_argument("--early_stop_patience", type=int, default=300)
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    layer_ids = [int(x) for x in args.layers.split(",")]
    device_list = args.devices.split(",")
    model_device = device_list[1] if len(device_list) > 1 else device_list[0]
    train_device = device_list[0]
    num_groups = way_to_num_groups(NUM_EXPERTS, args.way)

    print("=" * 60, flush=True)
    print("RANDOM Grouping + Distillation (RL-Group baseline)", flush=True)
    print("=" * 60, flush=True)
    print(f"  way={args.way}, num_groups={num_groups}", flush=True)
    print(f"  grouping_map={args.grouping_map}", flush=True)
    print(f"  layers={layer_ids}", flush=True)
    print(f"  output={args.output_dir}", flush=True)
    print("=" * 60, flush=True)

    # --- Load random grouping map ---
    with open(args.grouping_map) as f:
        gm = json.load(f)
    assert int(gm["num_experts"]) == NUM_EXPERTS
    assert int(gm["num_groups"]) == num_groups
    assert int(gm["way"]) == args.way
    print(f"  Loaded random grouping for {len(gm['layers'])} layers", flush=True)

    # --- Load routing weights (uniform if absent) ---
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

    # --- Load model ---
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(model_device)
    model.eval()

    # --- Collect calibration data ---
    print("Collecting calibration data...", flush=True)
    calib_data = collect_all_layer_inputs(
        model, tokenizer, layer_ids, args.calib_jsonl,
        args.calib_max_samples, args.calib_max_length, model_device, args.seed,
    )
    print(f"Calibration done. Collected data for {len(calib_data)} layers.", flush=True)
    torch.cuda.empty_cache()

    # --- Process each layer ---
    all_weights = {}
    all_grouping = {}

    for layer_id in layer_ids:
        print(f"\n{'=' * 60}\nLayer {layer_id}\n{'=' * 60}", flush=True)
        calib_x = calib_data[layer_id]

        # Inject the random grouping (instead of clustering_init)
        layer_map = gm["layers"][str(layer_id)]
        grouping = [int(layer_map[str(e)]) for e in range(NUM_EXPERTS)]
        groups = get_group_members(grouping, num_groups)
        print(f"  Random grouping: {len(groups)} groups", flush=True)

        all_grouping[str(layer_id)] = {str(e): int(g) for e, g in enumerate(grouping)}

        gate_ws, up_ws, down_ws = get_expert_weights(
            model, layer_id, NUM_EXPERTS, train_device
        )

        for gid, members in enumerate(groups):
            if len(members) <= 1:
                print(f"  Group {gid} ({len(members)}): using original weights", flush=True)
                gate_w = gate_ws[members[0]].cpu()
                up_w = up_ws[members[0]].cpu()
                down_w = down_ws[members[0]].cpu()
            else:
                print(f"  Group {gid} ({len(members)}): distilling...", flush=True)
                (gate_w, up_w, down_w), mse = distill_united_expert(
                    members, gate_ws, up_ws, down_ws, calib_x, routing_weights,
                    args.distill_steps, args.distill_lr,
                    args.early_stop_patience, args.early_stop_min_delta,
                )
                gate_w, up_w, down_w = gate_w.cpu(), up_w.cpu(), down_w.cpu()
                print(f"    MSE={mse:.8f}", flush=True)

            all_weights[f"{layer_id}_{gid}_gate_proj.weight"] = gate_w
            all_weights[f"{layer_id}_{gid}_up_proj.weight"] = up_w
            all_weights[f"{layer_id}_{gid}_down_proj.weight"] = down_w

        del gate_ws, up_ws, down_ws
        torch.cuda.empty_cache()

    # --- Save weights (same layout as clustering pipeline) ---
    # NOTE: all_weights is already keyed "{lid}_{gid}_{gate/up/down}_proj.weight",
    # which matches the deployed flat-shard format exactly. We save BOTH:
    #   (a) per-layer dirs  {lid}_{start}.pth   (mirrors clustering pipeline)
    #   (b) one flat shard  flat_weights.pth    (for trivial, key-exact merging)
    os.makedirs(args.output_dir, exist_ok=True)
    for layer_id in layer_ids:
        layer_dir = os.path.join(args.output_dir, f"layer_{layer_id}")
        os.makedirs(layer_dir, exist_ok=True)
        groups = get_group_members(
            [all_grouping[str(layer_id)][str(e)] for e in range(NUM_EXPERTS)],
            num_groups,
        )
        for gid, members in enumerate(groups):
            start_expert = members[0]
            sd = OrderedDict()
            sd["gate_proj.weight"] = all_weights[f"{layer_id}_{gid}_gate_proj.weight"]
            sd["up_proj.weight"] = all_weights[f"{layer_id}_{gid}_up_proj.weight"]
            sd["down_proj.weight"] = all_weights[f"{layer_id}_{gid}_down_proj.weight"]
            torch.save(sd, os.path.join(layer_dir, f"{layer_id}_{start_expert}.pth"))
        print(f"  Saved layer_{layer_id}/ ({num_groups} groups)", flush=True)

    # (b) flat shard — keys already in deployed format
    torch.save(all_weights, os.path.join(args.output_dir, "flat_weights.pth"))
    print(f"  Saved flat_weights.pth ({len(all_weights)} tensors)", flush=True)

    grouping_data = {
        "num_experts": NUM_EXPERTS,
        "num_groups": num_groups,
        "way": args.way,
        "layers": all_grouping,
        "meta": {"method": "random"},
    }
    with open(os.path.join(args.output_dir, "grouping_map.json"), "w") as f:
        json.dump(grouping_data, f, indent=2)
    print(f"  Saved grouping_map.json", flush=True)
    print(f"\nDone! All outputs in {args.output_dir}/", flush=True)


if __name__ == "__main__":
    main()
