<p align="center">
  <img src="assets/dark-text-logo.png" width="85%" alt="BrownoutMoE">
</p>

**BrownoutMoE: Learning Structure-Aware Expert Grouping for Adaptive Mixture-of-Experts Model Serving**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![Model](https://img.shields.io/badge/model-Qwen1.5--MoE--A2.7B-orange)

## ✨ Introduction

Mixture-of-Experts (MoE) large language models are increasingly deployed in online LLM services, where inference must be both accurate and responsive under bursty demand. In practice, MoE routing is highly imbalanced: a few hot experts process most routed tokens while many cold experts are rarely activated, which fragments kernels and underutilizes GPU parallelism.

Existing serving systems mainly optimize runtime execution (scheduling, communication overlap, kernel fusion) while preserving the original expert organization. **BrownoutMoE** instead reorganizes the expert structure itself. Inspired by the brownout paradigm in service computing, it learns which experts can share a **united expert**, so that expert utilization and serving efficiency improve while answer quality is preserved:

- **Offline (per checkpoint)**: calibration → expert output-similarity matrix → hierarchical-clustering warm start → **GRPO grouping search** scored by short-horizon, routing-weighted united-expert distillation error → grouping map + deployable distilled checkpoint.
- **Online (per request)**: the router is unchanged; an SLO-aware brownout controller adapts the substitution threshold from TTFT / TPOT / tail latency, sending eligible cold-expert calls to united experts under load pressure.

## 🔥 Features

### Core Features

- **Structure-aware expert grouping** — layer-wise expert grouping learned from behavior instead of expert indices
- **GRPO grouping search** — policy optimization over discrete groupings with capacity constraints, group-relative advantages from swap perturbations, and post-distillation MSE as the reward
- **Grouping-consistent distillation** — one student per group trained against a routing-weighted mixture target, exported with the grouping map
- **SLO-aware brownout control** — adaptive threshold that protects tail latency under bursts and recovers accuracy when the load subsides

### Integrated Optimizations

- Continuous batching scheduler with FCFS queueing
- PagedAttention-style block manager and prefill/decode kernels
- FlashAttention and fused MoE kernels
- Original / united-expert dual weight pools with GPU memory management

### Performance Highlights

- Reduces accuracy degradation by up to **71.4%** relative to sequential grouping (8-way, Qwen1.5-MoE-A2.7B)
- Improves unfused MoE throughput by up to **2.24x** over the baseline
- Keeps saturated decode latency about **38% lower** than a none-fused vLLM-style baseline under increasing ShareGPT load

## 🧰 Current Limitations

- Only Qwen1.5-MoE-A2.7B-Chat (60 routed experts, top-4 routing) is supported currently
- Documentation and unit tests are still under development

## 🚀 Installation

### Prerequisites

- Linux with CUDA GPUs (>= 40 GB memory recommended; multi-GPU supported)
- Python 3.10+, PyTorch 2.1+, CUDA 12.x

### Installation Steps

```bash
conda create -n brownoutmoe python=3.10 -y
conda activate brownoutmoe
pip install -r requirements.txt
pip install -e .
```

### Note

Pretrained weights of Qwen1.5-MoE-A2.7B-Chat should be downloaded separately, e.g. from [HuggingFace](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat) or ModelScope. The grouping search consumes a small calibration set in JSONL format.

## 🎯 Quick Start

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

See `training/run_grpo_8way.sh` for a multi-GPU launch example. The script exports the grouping map and united-expert weights for each MoE layer.

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

## 📚 Repository Structure

```
BrownoutMoE/
├── assets/                      # project logo
├── brownoutmoe/                 # online serving engine
│   ├── scheduler.py             # continuous-batching request scheduler
│   ├── slo_analyzer.py          # SLO-aware brownout threshold controller
│   ├── expert_loader.py         # original / united-expert weight pools
│   ├── generation.py            # LLM entry point
│   ├── kernels/                 # fused MoE, paged attention, etc.
│   └── server/                  # API server and client
├── training/                    # offline grouping search and distillation
│   ├── grpo_grouping_search.py  # GRPO expert-grouping search (core)
│   ├── cluster_distill.py       # clustering-based grouping baseline
│   ├── random_distill.py        # random grouping baseline
│   ├── distill_from_map.py      # distill united experts from a saved map
│   └── run_*.sh                 # launch scripts
├── examples/                    # serving and benchmark examples
├── setup.py
└── requirements.txt
```

## 📖 Citation

The manuscript is under review. If you use BrownoutMoE in your research, please cite:

```bibtex
@article{ding2026brownoutmoe,
  title   = {BrownoutMoE: Learning Structure-Aware Expert Grouping for Adaptive Mixture-of-Experts Model Serving},
  author  = {Ding, Yi and Xu, Minxian and Fang, Zhengxin and Ye, Kejiang and Xu, Chengzhong},
  year    = {2026}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).
