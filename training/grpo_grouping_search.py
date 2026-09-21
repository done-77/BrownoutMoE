#!/usr/bin/env python3
"""
True GRPO Expert Grouping + United Expert Distillation Training.

For each MoE layer:
  1. Collect calibration activations from all training data (ONE forward pass for ALL layers)
  2. Extract 60 experts' weights (fused gate_up_proj → split gate/up)
  3. Compute expert similarity matrix (output cosine similarity)
  4. Hierarchical clustering for warm-start initialization
  5. GRPO optimization loop: sample → perturb → distill reward → group-relative advantage → PPO update
  6. Final distillation: train united expert for each group (2000 steps, early stopping)
  7. Save grouping map + united expert weights (w1.pth/w2.pth/w3.pth)

Usage:
    python train_grpo_united.py \
        --model_path /root/hujianmin/LLMs/Qwen1.5-MoE-A2.7B-Chat \
        --calib_jsonl obqa_calib.jsonl \
        --output_dir ./8_way_united_experts_obqa-GRPO \
        --devices cuda:0,cuda:1 \
        --way 8 \
        --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
        --seed 42
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.cluster.hierarchy import fcluster, linkage
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_EXPERTS = 60
WAY_SWAP_DEFAULTS = {
    2: (1, 2),
    4: (1, 3),
    8: (2, 4),
}


def way_to_num_groups(num_experts: int, way: int) -> int:
    return math.ceil(num_experts / way)


# ---------------------------------------------------------------------------
# Target Model: SwiGLU FFN (for distillation)
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
# Expert forward (raw weight tensors)
# ---------------------------------------------------------------------------
def expert_forward(gate_w, up_w, down_w, x):
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    return F.linear(F.silu(gate) * up, down_w)


# ---------------------------------------------------------------------------
# Collect calibration inputs for ALL layers at once (single forward pass)
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_all_layer_inputs(
    model, tokenizer, layer_ids: List[int],
    calib_jsonl: str, max_samples: int, max_length: int,
    device: str, seed: int = 42,
) -> Dict[int, torch.Tensor]:
    """Run calibration data once, hook ALL target layers' MLP inputs."""
    random.seed(seed)
    texts = []
    with open(calib_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            if len(texts) >= max_samples:
                break
    random.shuffle(texts)

    layer_stores: Dict[int, List[torch.Tensor]] = {lid: [] for lid in layer_ids}
    temp_stores: Dict[int, List[torch.Tensor]] = {lid: [] for lid in layer_ids}
    handles = []

    for lid in layer_ids:
        moe_module = model.model.layers[lid].mlp
        def make_hook(layer_id):
            def hook_fn(_mod, inp, _out):
                temp_stores[layer_id].append(inp[0].detach().cpu())  # Move to CPU immediately
            return hook_fn
        handles.append(moe_module.register_forward_hook(make_hook(lid)))

    for i, text in enumerate(texts):
        ids = tokenizer(
            text, return_tensors="pt", max_length=max_length, truncation=True
        ).input_ids.to(device)
        model(ids)
        for lid in layer_ids:
            if temp_stores[lid]:
                layer_stores[lid].append(temp_stores[lid][-1].squeeze(0))  # Already on CPU
                temp_stores[lid].clear()
        if (i + 1) % 500 == 0:
            print(f"  Calibration: {i+1}/{len(texts)} samples processed", flush=True)

    for h in handles:
        h.remove()

    result = {}
    for lid in layer_ids:
        if layer_stores[lid]:
            # Move to CPU first to avoid OOM on GPU
            cpu_tensors = [t.cpu() for t in layer_stores[lid]]
            del layer_stores[lid]  # Free GPU memory
            t = torch.cat(cpu_tensors, dim=0)  # Cat on CPU
            del cpu_tensors
            result[lid] = t
            print(f"  [Layer {lid}] Calib tokens: {t.shape} (on CPU)", flush=True)
        else:
            print(f"  [Layer {lid}] WARNING: no activations collected!", flush=True)
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Get expert weights (fused gate_up_proj → split)
# ---------------------------------------------------------------------------
def get_expert_weights(model, layer_id: int, num_experts: int, device: str):
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
def compute_expert_similarity(gate_ws, up_ws, down_ws, calib_x, num_experts, device=None):
    # Determine device from expert weights
    if device is None:
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
    outputs = torch.stack(outputs)

    flat = outputs.view(num_experts, -1)
    norm = flat.norm(dim=1, keepdim=True) + 1e-8
    flat_normed = flat / norm
    sim = (flat_normed @ flat_normed.T).cpu().numpy()
    return sim


def clustering_init(similarity: np.ndarray, num_groups: int) -> List[int]:
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0)
    distance = (distance + distance.T) / 2
    Z = linkage(distance[np.triu_indices(len(distance), k=1)], method="average")
    labels = fcluster(Z, t=num_groups, criterion="maxclust") - 1
    return labels.tolist()


def sequential_grouping(num_experts: int, num_groups: int) -> List[int]:
    cap = math.ceil(num_experts / num_groups)
    return [min(e // cap, num_groups - 1) for e in range(num_experts)]


def get_group_members(grouping: List[int], num_groups: int) -> List[List[int]]:
    groups = [[] for _ in range(num_groups)]
    for e, g in enumerate(grouping):
        groups[g].append(e)
    return groups


# ---------------------------------------------------------------------------
# GRPO Policy
# ---------------------------------------------------------------------------
class Policy:
    def __init__(self, num_experts: int, num_groups: int, device: str):
        self.E = num_experts
        self.G = num_groups
        self.cap = math.ceil(num_experts / num_groups)
        self.device = device
        self.logits = torch.zeros(
            (num_experts, num_groups), dtype=torch.float32, device=device,
            requires_grad=True,
        )

    def init_from_grouping(self, grouping: List[int], temperature: float = 2.0):
        with torch.no_grad():
            self.logits.zero_()
            for e, g in enumerate(grouping):
                self.logits[e, g] = temperature

    def _masked_logits(self, expert_idx, counts, logits_src):
        mask = torch.tensor(
            [counts[g] < self.cap for g in range(self.G)],
            dtype=torch.bool, device=self.device,
        )
        logits_e = logits_src[expert_idx].clone()
        logits_e[~mask] = -1e9
        return logits_e

    def _step_distribution(self, expert_idx, counts, logits_src):
        logits_e = self._masked_logits(expert_idx, counts, logits_src)
        probs = F.softmax(logits_e, dim=0)
        return torch.distributions.Categorical(probs=probs)

    def sample(self, logits_override=None):
        logits_src = self.logits if logits_override is None else logits_override
        counts = [0] * self.G
        out = [-1] * self.E
        lp = torch.zeros((), device=self.device, dtype=torch.float32)
        entropy = torch.zeros((), device=self.device, dtype=torch.float32)
        for e in range(self.E):
            dist = self._step_distribution(e, counts, logits_src)
            a = dist.sample()
            lp = lp + dist.log_prob(a)
            entropy = entropy + dist.entropy()
            out[e] = int(a.item())
            counts[out[e]] += 1
        return out, lp, entropy

    def evaluate(self, grouping, logits_override=None, ref_logits=None):
        logits_src = self.logits if logits_override is None else logits_override
        counts = [0] * self.G
        lp = torch.zeros((), device=self.device, dtype=torch.float32)
        entropy = torch.zeros((), device=self.device, dtype=torch.float32)
        kl_to_ref = torch.zeros((), device=self.device, dtype=torch.float32)
        for e, gid in enumerate(grouping):
            dist = self._step_distribution(e, counts, logits_src)
            action = torch.tensor(int(gid), device=self.device)
            lp = lp + dist.log_prob(action)
            entropy = entropy + dist.entropy()
            if ref_logits is not None:
                ref_dist = self._step_distribution(e, counts, ref_logits)
                kl_to_ref = kl_to_ref + torch.distributions.kl.kl_divergence(dist, ref_dist)
            counts[int(gid)] += 1
        return lp, entropy, kl_to_ref

    def greedy(self) -> List[int]:
        counts = [0] * self.G
        out = [-1] * self.E
        for e in range(self.E):
            dist = self._step_distribution(e, counts, self.logits)
            out[e] = int(dist.probs.argmax().item())
            counts[out[e]] += 1
        return out


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------
def perturb_grouping(base_grouping, num_groups, num_swaps=2):
    E = len(base_grouping)
    result = list(base_grouping)
    groups: Dict[int, List[int]] = {}
    for e in range(E):
        groups.setdefault(result[e], []).append(e)
    for _ in range(num_swaps):
        non_empty = [g for g in range(num_groups) if len(groups.get(g, [])) > 0]
        if len(non_empty) < 2:
            break
        g1, g2 = random.sample(non_empty, 2)
        e1 = random.choice(groups[g1])
        e2 = random.choice(groups[g2])
        result[e1] = g2
        result[e2] = g1
        groups[g1].remove(e1); groups[g1].append(e2)
        groups[g2].remove(e2); groups[g2].append(e1)
    return result


# ---------------------------------------------------------------------------
# Short distillation reward
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_distill_reward(
    grouping, gate_ws, up_ws, down_ws, calib_x, routing_weights,
    num_groups, num_experts, distill_steps=50, distill_lr=1e-3,
):
    groups = [[] for _ in range(num_groups)]
    for e in range(num_experts):
        groups[grouping[e]].append(e)

    total_mse = 0.0
    # Ensure x is on the same device as expert weights
    device = gate_ws[0].device
    x = calib_x.to(device).float()

    for gid in range(num_groups):
        members = groups[gid]
        if len(members) <= 1:
            continue
        member_weights = torch.tensor(
            [float(routing_weights[e]) for e in members], device=x.device,
        )
        member_weights = member_weights / (member_weights.sum() + 1e-12)

        with torch.no_grad():
            tgt = None
            for i, e in enumerate(members):
                y = expert_forward(gate_ws[e], up_ws[e], down_ws[e], x)
                tgt = y * member_weights[i] if tgt is None else tgt + y * member_weights[i]

        united_gate = torch.stack([gate_ws[e] for e in members]).mean(0).clone()
        united_up = torch.stack([up_ws[e] for e in members]).mean(0).clone()
        united_down = torch.stack([down_ws[e] for e in members]).mean(0).clone()

        united_gate_p = nn.Parameter(united_gate)
        united_up_p = nn.Parameter(united_up)
        united_down_p = nn.Parameter(united_down)
        opt = torch.optim.AdamW(
            [united_gate_p, united_up_p, united_down_p], lr=distill_lr, weight_decay=0.0
        )

        with torch.enable_grad():
            for _ in range(distill_steps):
                out = expert_forward(united_gate_p, united_up_p, united_down_p, x)
                loss = torch.mean((out - tgt).float().pow(2))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        with torch.no_grad():
            out = expert_forward(united_gate_p, united_up_p, united_down_p, x)
            final_mse = torch.mean((out - tgt).float().pow(2)).item()

        group_weight = sum(float(routing_weights[e]) for e in members)
        total_mse += group_weight * final_mse

    return -total_mse


# ---------------------------------------------------------------------------
# GRPO grouping search
# ---------------------------------------------------------------------------
def grpo_grouping_search(
    layer_id, gate_ws, up_ws, down_ws, calib_x, routing_weights,
    num_groups, device, args,
):
    # Move calibration data to device
    calib_x = calib_x.to(device).float()
    routing_weights = routing_weights.to(device)
    gate_ws = [w.to(device) for w in gate_ws]
    up_ws = [w.to(device) for w in up_ws]
    down_ws = [w.to(device) for w in down_ws]

    print(f"  [Layer {layer_id}] Computing expert similarity matrix...", flush=True)
    sim = compute_expert_similarity(gate_ws, up_ws, down_ws, calib_x, NUM_EXPERTS)
    cluster_grouping = clustering_init(sim, num_groups)

    cluster_reward = compute_distill_reward(
        cluster_grouping, gate_ws, up_ws, down_ws, calib_x,
        routing_weights, num_groups, NUM_EXPERTS,
        args.grpo_distill_steps, args.grpo_distill_lr,
    )
    seq_g = sequential_grouping(NUM_EXPERTS, num_groups)
    seq_reward = compute_distill_reward(
        seq_g, gate_ws, up_ws, down_ws, calib_x,
        routing_weights, num_groups, NUM_EXPERTS,
        args.grpo_distill_steps, args.grpo_distill_lr,
    )
    print(f"  [Layer {layer_id}] Seq reward={seq_reward:.6f}, Cluster reward={cluster_reward:.6f}", flush=True)

    if cluster_reward > seq_reward:
        init_grouping = cluster_grouping
        init_reward = cluster_reward
        print(f"  [Layer {layer_id}] Using clustering init (better by "
              f"{(cluster_reward - seq_reward)/abs(seq_reward)*100:.1f}%)", flush=True)
    else:
        init_grouping = seq_g
        init_reward = seq_reward
        print(f"  [Layer {layer_id}] Using sequential init", flush=True)

    policy = Policy(NUM_EXPERTS, num_groups, device)
    policy.init_from_grouping(init_grouping, temperature=3.0)
    reference_logits = policy.logits.detach().clone()

    best_reward = init_reward
    best_grouping = list(init_grouping)
    optimizer = torch.optim.Adam([policy.logits], lr=args.policy_lr)
    B, G = args.batch_size, args.group_size
    swaps_min, swaps_max = WAY_SWAP_DEFAULTS.get(args.way, (2, 4))
    no_improve_steps = 0

    print(f"  [Layer {layer_id}] GRPO: steps={args.grpo_steps}, B={B}, G={G}, "
          f"swaps=[{swaps_min},{swaps_max}]", flush=True)

    for step in range(1, args.grpo_steps + 1):
        old_logits = policy.logits.detach().clone()

        base_groupings = []
        for _ in range(B):
            grp, _lp, _ = policy.sample(logits_override=old_logits)
            base_groupings.append(grp)

        all_groupings_flat = []
        all_old_lps_flat = []
        group_reward_tensors = []
        improved = False

        for b in range(B):
            base = base_groupings[b]
            perts = []
            for _ in range(G):
                n_swaps = random.randint(swaps_min, swaps_max)
                perts.append(perturb_grouping(base, num_groups, n_swaps))
            group = [base] + perts

            group_rewards = []
            group_old_lps = []
            for i, grp in enumerate(group):
                ds = args.grpo_distill_steps if i == 0 else args.perturb_distill_steps
                r = compute_distill_reward(
                    grp, gate_ws, up_ws, down_ws, calib_x,
                    routing_weights, num_groups, NUM_EXPERTS,
                    ds, args.grpo_distill_lr,
                )
                lp, _, _ = policy.evaluate(grp, logits_override=old_logits)
                group_rewards.append(r)
                group_old_lps.append(lp.detach())
                if r > best_reward + args.grpo_min_delta:
                    best_reward = r
                    best_grouping = list(grp)
                    improved = True

            all_groupings_flat.extend(group)
            all_old_lps_flat.extend(group_old_lps)
            group_reward_tensors.append(torch.tensor(group_rewards, device=device, dtype=torch.float32))

        # Group-relative advantages
        all_advantages = []
        for b in range(B):
            r_group = group_reward_tensors[b]
            r_mean = r_group.mean()
            r_std = r_group.std(unbiased=False)
            all_advantages.append((r_group - r_mean) / (r_std + 1e-8))
        advantages_t = torch.cat(all_advantages)
        old_lp_t = torch.stack(all_old_lps_flat)

        # PPO update
        for _epoch in range(args.grpo_epochs):
            new_lps, entropies, ref_kls = [], [], []
            for grouping in all_groupings_flat:
                new_lp, ent, rkl = policy.evaluate(grouping, ref_logits=reference_logits)
                new_lps.append(new_lp)
                entropies.append(ent)
                ref_kls.append(rkl)

            new_lp_stack = torch.stack(new_lps)
            entropy_stack = torch.stack(entropies)
            ref_kl_stack = torch.stack(ref_kls)

            ratios = torch.exp(new_lp_stack - old_lp_t)
            clipped = torch.clamp(ratios, 1.0 - args.grpo_clip_eps, 1.0 + args.grpo_clip_eps)
            policy_loss = -torch.min(ratios * advantages_t.detach(), clipped * advantages_t.detach()).mean()
            entropy_bonus = entropy_stack.mean() / float(NUM_EXPERTS)
            ref_kl = ref_kl_stack.mean() / float(NUM_EXPERTS)
            loss = policy_loss - args.entropy_coef * entropy_bonus + args.ref_kl_coef * ref_kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([policy.logits], 1.0)
            optimizer.step()

        if step == 1 or step % 10 == 0 or step == args.grpo_steps:
            r_mean = float(torch.cat(group_reward_tensors).mean().item())
            imp = (best_reward - seq_reward) / abs(seq_reward) * 100 if seq_reward != 0 else 0
            print(f"  [Layer {layer_id}] GRPO step={step}/{args.grpo_steps} "
                  f"r_mean={r_mean:.6f} best={best_reward:.6f} improv={imp:.1f}%", flush=True)

        if improved:
            no_improve_steps = 0
        else:
            no_improve_steps += 1
        if args.grpo_patience > 0 and no_improve_steps >= args.grpo_patience:
            print(f"  [Layer {layer_id}] GRPO early stop at step {step}", flush=True)
            break

    greedy_g = policy.greedy()
    greedy_r = compute_distill_reward(
        greedy_g, gate_ws, up_ws, down_ws, calib_x,
        routing_weights, num_groups, NUM_EXPERTS,
        args.grpo_distill_steps, args.grpo_distill_lr,
    )
    if greedy_r > best_reward:
        best_reward = greedy_r
        best_grouping = greedy_g

    final_imp = (best_reward - seq_reward) / abs(seq_reward) * 100 if seq_reward != 0 else 0
    print(f"  [Layer {layer_id}] GRPO done: best_MSE={-best_reward:.6f}, "
          f"seq_MSE={-seq_reward:.6f}, improv={final_imp:.1f}%", flush=True)
    return best_grouping, best_reward


# ---------------------------------------------------------------------------
# Final distillation
# ---------------------------------------------------------------------------
def distill_united_experts(
    layer_id, grouping, gate_ws, up_ws, down_ws, calib_x, num_groups, device, args,
):
    groups = get_group_members(grouping, num_groups)
    gate_ws = [w.to(device) for w in gate_ws]
    up_ws = [w.to(device) for w in up_ws]
    down_ws = [w.to(device) for w in down_ws]
    x = calib_x.to(device).float()

    weight_dict = {}
    pbar = tqdm(groups, desc=f"  [Layer {layer_id}] Distilling")

    for gid, members in enumerate(pbar):
        if len(members) <= 1:
            e = members[0]
            weight_dict[f"{layer_id}_{gid}_gate_proj.weight"] = gate_ws[e].clone()
            weight_dict[f"{layer_id}_{gid}_up_proj.weight"] = up_ws[e].clone()
            weight_dict[f"{layer_id}_{gid}_down_proj.weight"] = down_ws[e].clone()
            continue

        with torch.no_grad():
            tgt = None
            for e in members:
                y = expert_forward(gate_ws[e], up_ws[e], down_ws[e], x)
                tgt = y / len(members) if tgt is None else tgt + y / len(members)

        target_model = TargetModel().to(device).float()
        with torch.no_grad():
            target_model.gate_proj.weight.copy_(torch.stack([gate_ws[e] for e in members]).mean(0))
            target_model.up_proj.weight.copy_(torch.stack([up_ws[e] for e in members]).mean(0))
            target_model.down_proj.weight.copy_(torch.stack([down_ws[e] for e in members]).mean(0))

        optimizer = optim.Adam(target_model.parameters(), lr=args.distill_lr)
        criterion = nn.MSELoss()

        target_model.train()
        best_loss = float("inf")
        best_state = None
        no_improve = 0

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
                no_improve = 0
            else:
                no_improve += 1

            if step % 200 == 0 or step == 1:
                print(f"    [L{layer_id}] G{gid} (experts {members}): step {step}/{args.distill_steps}, "
                      f"loss={loss_val:.6f}, best={best_loss:.6f}", flush=True)

            if no_improve >= args.early_stop_patience:
                print(f"    [L{layer_id}] G{gid}: EARLY STOP at step {step}", flush=True)
                break

        if best_state is not None:
            target_model.load_state_dict(best_state)

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
    ap = argparse.ArgumentParser(description="True GRPO Expert Grouping + United Expert Distillation")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--calib_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--devices", required=True, help="e.g. cuda:0,cuda:1")
    ap.add_argument("--way", type=int, required=True, choices=[2, 4, 8])
    ap.add_argument("--routing_stats", default="")
    ap.add_argument("--layers", type=str, default="")
    ap.add_argument("--calib_max_samples", type=int, default=5507)
    ap.add_argument("--calib_max_length", type=int, default=256)
    ap.add_argument("--grpo_steps", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--policy_lr", type=float, default=0.01)
    ap.add_argument("--grpo_distill_steps", type=int, default=50)
    ap.add_argument("--perturb_distill_steps", type=int, default=20)
    ap.add_argument("--grpo_distill_lr", type=float, default=1e-3)
    ap.add_argument("--grpo_clip_eps", type=float, default=0.2)
    ap.add_argument("--grpo_epochs", type=int, default=3)
    ap.add_argument("--grpo_patience", type=int, default=100)
    ap.add_argument("--grpo_min_delta", type=float, default=0.0)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--ref_kl_coef", type=float, default=0.02)
    ap.add_argument("--distill_steps", type=int, default=2000)
    ap.add_argument("--distill_lr", type=float, default=1e-4)
    ap.add_argument("--early_stop_patience", type=int, default=300)
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    devices = [d.strip() for d in args.devices.split(",")]
    model_device = devices[-1]
    train_device = devices[0]
    num_groups = way_to_num_groups(NUM_EXPERTS, args.way)

    if args.layers:
        target_layers = [int(x) for x in args.layers.split(",")]
    else:
        target_layers = list(range(24))

    print("=" * 60, flush=True)
    print("True GRPO Expert Grouping + United Expert Distillation", flush=True)
    print("=" * 60, flush=True)
    print(f"  way={args.way}, num_groups={num_groups}", flush=True)
    print(f"  layers={target_layers}", flush=True)
    print(f"  model_device={model_device}, train_device={train_device}", flush=True)
    print(f"  calib={args.calib_jsonl} ({args.calib_max_samples} samples)", flush=True)
    print(f"  GRPO: steps={args.grpo_steps}, B={args.batch_size}, G={args.group_size}", flush=True)
    print(f"  Distill: steps={args.distill_steps}, patience={args.early_stop_patience}", flush=True)
    print(f"  output={args.output_dir}", flush=True)
    print("=" * 60, flush=True)

    # Load model
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    model = model.to(model_device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )

    # Load routing stats
    routing_weights_dict = {}
    if args.routing_stats and os.path.exists(args.routing_stats):
        with open(args.routing_stats) as f:
            stats = json.load(f)
        for lid in target_layers:
            counts = [int(x) for x in stats["routing_counts"][str(lid)]]
            total = sum(counts)
            routing_weights_dict[lid] = torch.tensor(
                [c / total for c in counts], dtype=torch.float32, device=train_device
            )
    else:
        for lid in target_layers:
            routing_weights_dict[lid] = torch.ones(
                NUM_EXPERTS, dtype=torch.float32, device=train_device
            ) / NUM_EXPERTS

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Collect calibration inputs in batches to avoid OOM
    # Hooking all 12 layers at once causes OOM, so we collect 3 layers at a time
    BATCH_SIZE = 3
    all_calib_x = {}
    print(f"\nCollecting calibration inputs in batches of {BATCH_SIZE} layers...", flush=True)
    for batch_start in range(0, len(target_layers), BATCH_SIZE):
        batch_layers = target_layers[batch_start:batch_start + BATCH_SIZE]
        print(f"  Batch: layers {batch_layers}...", flush=True)
        batch_data = collect_all_layer_inputs(
            model, tokenizer, batch_layers,
            args.calib_jsonl, args.calib_max_samples, args.calib_max_length,
            model_device, args.seed,
        )
        all_calib_x.update(batch_data)
    print(f"Calibration done. Collected data for {len(all_calib_x)} layers.", flush=True)

    all_weight_dict = {}
    all_grouping_map = {
        "num_experts": NUM_EXPERTS, "num_groups": num_groups,
        "way": args.way, "layers": {},
    }
    os.makedirs(args.output_dir, exist_ok=True)

    for layer_id in target_layers:
        print(f"\n{'='*60}", flush=True)
        print(f"Layer {layer_id} / {max(target_layers)}", flush=True)
        print(f"{'='*60}", flush=True)

        torch.cuda.empty_cache()

        if layer_id not in all_calib_x:
            print(f"  Skipping layer {layer_id}: no calibration data", flush=True)
            continue

        calib_x = all_calib_x[layer_id]
        gate_ws, up_ws, down_ws = get_expert_weights(model, layer_id, NUM_EXPERTS, model_device)
        routing_weights = routing_weights_dict[layer_id]

        # Phase 1: GRPO grouping search
        best_grouping, best_reward = grpo_grouping_search(
            layer_id, gate_ws, up_ws, down_ws, calib_x,
            routing_weights, num_groups, train_device, args,
        )

        all_grouping_map["layers"][str(layer_id)] = {
            str(e): int(best_grouping[e]) for e in range(NUM_EXPERTS)
        }

        # Phase 2: Final distillation
        print(f"\n  [Layer {layer_id}] Starting final distillation...", flush=True)
        wdict = distill_united_experts(
            layer_id, best_grouping, gate_ws, up_ws, down_ws,
            calib_x, num_groups, train_device, args,
        )
        all_weight_dict.update(wdict)

        del gate_ws, up_ws, down_ws
        torch.cuda.empty_cache()

        # Save intermediate progress
        grouping_path = os.path.join(args.output_dir, "grouping_map.json")
        with open(grouping_path, "w") as f:
            json.dump(all_grouping_map, f, indent=2)

    # Save final grouping map
    grouping_path = os.path.join(args.output_dir, "grouping_map.json")
    with open(grouping_path, "w") as f:
        json.dump(all_grouping_map, f, indent=2)
    print(f"\nSaved grouping map: {grouping_path}", flush=True)

    # Split into w1/w2/w3
    train_layers_set = set(target_layers)
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
            print(f"Saved {out_path} ({len(chunk_dict)} weight tensors)", flush=True)

    print(f"\nDone! All outputs in {args.output_dir}/", flush=True)


if __name__ == "__main__":
    main()
