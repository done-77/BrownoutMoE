# BrownoutMoE

**Learning Structure-Aware Expert Grouping for Adaptive Mixture-of-Experts Model Serving**

BrownoutMoE is a structure-aware optimization framework for efficient and accurate Mixture-of-Experts (MoE) large language model serving. MoE routing is highly imbalanced in practice: a few hot experts process most tokens while many cold experts are rarely activated, which fragments kernels and underutilizes GPU parallelism. Instead of only optimizing runtime execution, BrownoutMoE reorganizes the expert structure itself: it learns which experts can share a **united expert** so that utilization and throughput improve while answer quality is preserved.

## Overview

BrownoutMoE separates the offline structure problem from the online control problem:

```
                 offline (per checkpoint)                     online (per request)
  calibration -> similarity -> clustering warm start      router unchanged
        -> GRPO grouping search  ->  united-expert         brownout threshold τ adapted
             distillation        ->  grouping map          by SLO analyzer from TTFT /
                                                          TPOT / tail latency
```

- **GRPO grouping search** (`training/grpo_grouping_search.py`): collects calibration activations, computes an expert output-similarity matrix, warm-starts from hierarchical clustering, and searches layer-wise groupings with Group Relative Policy Optimization. Each candidate is scored by short-horizon, routing-weighted united-expert distillation error (the true post-distillation MSE), and the policy is updated with a clipped surrogate objective.
- **Grouping-consistent distillation** (`training/cluster_distill.py`, `training/distill_from_map.py`): trains one student per group with a routing-weighted mixture target and exports a deployable checkpoint together with the grouping map.
- **SLO-aware brownout serving** (`brownoutmoe/`): an inference engine with continuous batching, paged attention, expert weight pools, and a brownout controller that lowers the substitution threshold under load pressure and recovers it when the load subsides.

## Repository structure

```
BrownoutMoE/
├── brownoutmoe/                    # online serving engine
│   ├── scheduler.py             # continuous-batching request scheduler
│   ├── slo_analyzer.py          # SLO-aware brownout threshold controller
│   ├── expert_loader.py         # original / united-expert weight pools
│   ├── generation.py            # LLM entry point
│   ├── model.py / weight.py     # Qwen1.5-MoE model and weight loading
│   ├── gpu_manager.py           # GPU memory management
│   ├── kernels/                 # fused MoE, paged attention, etc.
│   └── server/                  # API server and client
├── training/                    # offline grouping search and distillation
│   ├── grpo_grouping_search.py  # GRPO expert-grouping search (core)
│   ├── cluster_distill.py       # clustering-based grouping baseline
│   ├── random_distill.py        # random grouping baseline
│   ├── distill_from_map.py      # distill united experts from a saved map
│   ├── united_experts.py        # united-expert student model
│   ├── weight.py                # expert weight extraction utilities
│   ├── run_grpo_8way.sh         # example launch scripts (4-way / 8-way)
│   ├── run_grpo_4way.sh
│   └── run_cluster_4way.sh
├── examples/                    # serving and benchmark examples
│   ├── slo_control.py           # SLO-aware serving with adaptive threshold
│   ├── throughput_benchmark.py  # throughput under brownout configurations
│   ├── latency_benchmark.py     # per-second prefill / decode latency
│   ├── scalability_benchmark.py # multi-GPU scalability sweep
│   ├── online_serving.py        # general online serving entry
│   ├── basic_inference.py       # minimal inference example
│   └── chatbot.py               # interactive chat demo
├── setup.py
├── requirements.txt
└── LICENSE
```

## Installation

```bash
conda create -n brownoutmoe python=3.10 -y
conda activate brownoutmoe
pip install -r requirements.txt
pip install -e .
```

A CUDA GPU with at least 40 GB memory is recommended for the default Qwen1.5-MoE-A2.7B-Chat configuration; multi-GPU setups are supported for layer-parallel grouping search.

## Quick start

### 1. GRPO expert-grouping search (offline)

```bash
cd training
python grpo_grouping_search.py \
    --model_path /path/to/Qwen1.5-MoE-A2.7B-Chat \
    --calib_jsonl your_calibration.jsonl \
    --output_dir ./8_way_united_experts \
    --devices cuda:0,cuda:1 \
    --way 8 \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
    --seed 42
```

The script exports the grouping map and united-expert weights (`w1.pth` / `w2.pth` / `w3.pth`) for each MoE layer. See `training/run_grpo_8way.sh` for a multi-GPU launch example.

### 2. Serve with SLO-aware brownout control (online)

```bash
cd examples
python slo_control.py --model_path /path/to/Qwen1.5-MoE-A2.7B-Chat \
    --united_dir /path/to/8_way_united_experts \
    --target_slo 0.5
```

### 3. Benchmarks

```bash
python examples/throughput_benchmark.py   # throughput vs. brownout configuration
python examples/latency_benchmark.py      # per-second prefill / decode latency
python examples/scalability_benchmark.py  # multi-GPU scalability
```

## Results highlights

On Qwen1.5-MoE-A2.7B with 2/4/8-way grouping, BrownoutMoE:

- reduces accuracy degradation by up to **71.4%** relative to sequential grouping at the 8-way setting,
- improves unfused MoE throughput by up to **2.24x** over the baseline,
- keeps saturated decode latency about **38% lower** than a none-fused vLLM-style baseline under increasing ShareGPT load through adaptive threshold control.

## Citation

The manuscript is under review. If you use BrownoutMoE in your research, please cite:

```bibtex
@article{ding2026brownoutmoe,
  title   = {BrownoutMoE: Learning Structure-Aware Expert Grouping for Adaptive Mixture-of-Experts Model Serving},
  author  = {Ding, Yi and Xu, Minxian and Fang, Zhengxin and Ye, Kejiang and Xu, Chengzhong},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
