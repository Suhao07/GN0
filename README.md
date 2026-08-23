<div align="center">

<img src="assets/GN0.gif" alt="GN0 teaser" width="92%">

<p>
  <a href="https://telehuman-gn0.github.io"><img src="assets/badges/project.svg" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2606.03682"><img src="assets/badges/paper.svg" alt="GN0 Paper - arXiv"></a>
  <a href="https://huggingface.co/datasets/TeleEmbodied/GN-Matrix"><img src="assets/badges/dataset.svg" alt="GN-Matrix Dataset"></a>
  <a href="https://github.com/TeleHuman/GN0-VLN-CE"><img src="assets/badges/bench.svg" alt="GN0-VLN-CE"></a>
</p>

</div>

## 🏠 Introduction

GN0 is a unified research framework for **Generation**, **Evaluation**, and
**Policy Learning** in Vision-and-Language Navigation (VLN). Built on 3D
Gaussian Splatting (3DGS), GN0 connects realistic scene construction,
high-fidelity embodied simulation, and navigation-policy evaluation in visually
grounded indoor environments.

This repository provides the **GN-Bench** evaluation workflow and the
**Break-and-Establish (BAE)** navigation policy. The current release focuses on
InteriorGS and offers a reproducible pipeline for local and remote policy
evaluation.

### Highlights

- **3DGS-native navigation benchmark.** GN-Bench evaluates agents directly in high-fidelity 3D Gaussian Splatting scenes.
- **Unified GN0 ecosystem.** The repository connects GN-Matrix data, GN-Bench simulation, and GN-BAE policy evaluation.
- **InteriorGS evaluation workflow.** Unified entry points support instruction-following evaluation on InteriorGS scenes.
- **Scalable episode splitting.** Deterministic worker sharding supports multi-GPU and multi-process evaluation.
- **Lightweight metric analysis.** Evaluation logs can be summarized into TL, NE, OS, SR, and SPL with a single script.

## 📖 News

- `[2026-08]` We integrated BAE 2.0 into `gammanav`.
- `[2026-07]` We released the GN-Matrix InteriorGS test trajectories.
- `[2026-06]` We released the GN0-VLN-CE evaluation workflow.
- `[2026-06]` We released the GN-Bench InteriorGS evaluation workflow.

---

## 📋 Table of Contents

- [🏠 Introduction](#-introduction)
- [📖 News](#-news)
- [📦 Overview](#-overview)
- [📚 Getting Started](#-getting-started)
- [🎥 Dataset Rendering](#-dataset-rendering)
- [🔌 Remote Client](#-remote-client)
- [🧪 Evaluation](#-evaluation)
- [🧭 GN0-VLN-CE](#-gn0-vln-ce)
- [🔗 Citation](#-citation)
- [👏 Acknowledgements](#-acknowledgements)
- [📄 License](#-license)

## 📦 Overview

### 🧩 GN0 Components

- **[GN-Matrix](https://huggingface.co/datasets/TeleEmbodied/GN-Matrix)** —
  Large-scale navigation trajectories grounded in 3D Gaussian Splatting scenes.
- **GN-Bench** — An interactive benchmark and simulator for high-fidelity VLN
  evaluation.
- **[GN-BAE](https://huggingface.co/TeleEmbodied/GN-BAE/tree/main)** — A
  navigation foundation model for map-based and map-free policy learning.

### 🤗 Model Zoo & Datasets

- [GN-BAE checkpoint](https://huggingface.co/TeleEmbodied/GN-BAE/tree/main) — CC BY-NC 4.0
- [GN-Matrix dataset](https://huggingface.co/datasets/TeleEmbodied/GN-Matrix) — CC BY-NC 4.0
- [InteriorGS dataset](https://huggingface.co/datasets/spatialverse/InteriorGS) — separate dataset terms; access approval is required

For the InteriorGS dataset, scenes are rendered from `3dgs_compressed.ply` by default. If disk space is sufficient and higher 3DGS quality is preferred, decompress the Gaussian splats with:

```bash
npm install -g @playcanvas/splat-transform
splat-transform 3dgs_compressed.ply 3dgs_decompressed.ply
```

## 📚 Getting Started

Follow the [installation guide](docs/INSTALLATION.md) for the complete environment setup,
including PyTorch, the CUDA rendering backend, GN-Bench-Tools, and BAE.

BAE policy code lives under `gammanav.vln.model.bae`. GN-Bench datasets,
metrics, simulator adapters, and evaluation clients remain in GN0 and are not
part of the policy package.

After installation, prepare datasets and checkpoints with the following layout:

```text
GN0/
├── gammanav/                     # BAE policy implementation
├── gn0/                          # Benchmark adapters, agents, collectors, runners
├── configs/gn_bench/interiorgs/  # Split-based simulator/evaluation configs
├── scripts/evaluation/           # BAE, remote-policy, and render launchers
├── data/                         # Dataset files
│   ├── datasets/
│   │   └── GN_Matrix/
│   │       └── InteriorGS/
│   └── scene_datasets/
│       └── InteriorGS/
├── GN-Bench-Tools/               # Benchmark tools (kept outside gammanav)
└── model_zoo/
    └── GN-BAE/                   # Pretrained GN-BAE model weights
```

Run the InteriorGS evaluation:

```bash
bash scripts/evaluation/eval_bae.sh \
  --model-path model_zoo/GN-BAE \
  --exp-config configs/gn_bench/interiorgs/test_seen.yaml \
  --num-gpus 1 \
  --procs-per-gpu 1 \
  --save-media none \
  --result-dir tmp/bae_test_seen
```

`--save-media` accepts `none` (default), `images`, or `video`. Local DAgger
collection requires `--save-media images` because its records reference the
persisted model inputs.

Monitor evaluation progress:

```bash
watch -n 1 python -m gn0.analysis.results --path tmp/bae_test_seen
```

Terminate active evaluation workers if needed:

```bash
bash scripts/evaluation/kill_bae_eval.sh
```

## 🎥 Dataset Rendering

GN-Bench provides a dedicated rendering entry point that converts the trajectory
split selected by `--exp-config` into egocentric MP4 videos. Dataset rendering
does not load or execute the BAE policy.

```bash
bash scripts/evaluation/render_dataset.sh \
  --exp-config configs/gn_bench/interiorgs/train_seen.yaml \
  --start-idx 0 --end-idx 5 \
  --result-dir tmp/train_seen
```

Videos are written by scene and trajectory id:

```text
<result-dir>/<scene>/<trajectory>.mp4
```

`--start-idx` is inclusive and `--end-idx` is exclusive. Omit both limits to
render the complete split. Use `--num-gpus` to assign one renderer per GPU, and
increase `--procs-per-gpu` only when additional workers fit in GPU memory.

Rendering options are configured in the YAML passed with `--exp-config`,
including RGB resolution, video frame rate, codec, path resampling, and
trajectory smoothing.

The renderer first simplifies the pixel-space A\* path with line-of-sight
smoothing, resamples it in world space, and then renders each pose with a
smoothed look-ahead heading.

## 🔌 Remote Client

GN-Bench can evaluate navigation policies hosted outside the GN-Bench environment through a lightweight WebSocket client. The remote policy server receives RGB observations and language instructions, then returns a chunk of actions.

Run GN-Bench workers against remote servers:

```bash
bash scripts/evaluation/eval_remote.sh \
  --exp-config configs/gn_bench/interiorgs/test_seen.yaml \
  --num-gpus 8 \
  --action-format auto \
  --result-dir tmp/remote_eval \
  --server-port 8000
```

Run `bash scripts/evaluation/eval_remote.sh --help` for the complete remote
evaluation interface, including worker scheduling, media output, DAgger, stall
recovery, and remote action controls.

Terminate active remote evaluation workers if needed:

```bash
bash scripts/evaluation/kill_remote_eval.sh
```

The kill script is scoped to the current user and checkout. It does not stop the
remote policy server.

Remote servers receive msgpack-numpy payloads with these fields:

```python
{
    "endpoint": "infer",
    "images": current_or_new_rgb_hwc_or_thwc_uint8,
    "instruction": instruction_text,
    "session_id": episode_id,
}
```

Recommended response formats:

```python
{"action_type": "discrete", "actions": [1, 1, 2, 3]}
{"action_type": "nav_delta", "actions": [[dx, dy, dyaw], ...]}
```

Discrete actions follow the VLN-CE convention: `0=STOP`, `1=MOVE_FORWARD`,
`2=TURN_LEFT`, and `3=TURN_RIGHT`. `nav_delta` actions are relative physical
deltas `[dx_forward, dy_left, dyaw]`, with translation in meters and yaw in
radians. Translation is interpreted in the chunk-start frame by default; use
`--remote-translation-frame current` to make each delta relative to the current
agent frame.

## 🧪 Evaluation

### 📊 Metrics

`python -m gn0.analysis.results` reads JSON logs under the selected result
directory and reports standard VLN metrics in an aligned terminal summary.
Use `--format json` when machine-readable results are required.

- **TL** — Average trajectory length.
- **NE ↓** — Navigation error.
- **OS ↑** — Oracle success.
- **SR ↑** — Success rate.
- **SPL ↑** — Success weighted by path length.

## 🧭 GN0-VLN-CE

[GN0-VLN-CE](https://github.com/TeleHuman/GN0-VLN-CE) is a separate companion repository for evaluating our GN-BAE navigation model on the standard VLN-CE benchmark. It connects GN0-style policy learning with established VLN-CE evaluation protocols, while the current repository focuses on the GN-Bench InteriorGS evaluation workflow.

## 🔗 Citation

If GN0 is useful for your research, please cite our paper:

```bibtex
@article{li2026gn0,
  title={GN0: Toward a Unified Paradigm for Generation, Evaluation, and Policy Learning in Visual-Language Navigation},
  author={Li, Xinhai and Zhang, Xiaotao and Huang, Yuehao and Dong, Jiankun and Wang, Tianhang and Zhou, Sunyao and Wu, Yunzi and Sun, Chengnuo and Ge, Yunfei and Weng, Qizhen and others},
  journal={arXiv preprint arXiv:2606.03682},
  year={2026}
}
```

GitHub users can also cite this release through [`CITATION.cff`](CITATION.cff).

## 👏 Acknowledgements

GN-Bench-Tools is adapted from
[Habitat-Lab](https://github.com/facebookresearch/habitat-lab) and extended for
3D Gaussian Splatting-based navigation. We thank the Habitat-Lab developers,
the [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS)
authors, and the broader Embodied
AI and 3DGS communities for making this research infrastructure possible.

## 📄 License

GN0 source code is released under the [Apache License 2.0](LICENSE). The
Habitat-Lab-derived portions of `GN-Bench-Tools/` remain under the MIT License.
See [Third-Party Notices](docs/THIRD_PARTY_NOTICES.md) for attribution and scope.

Model checkpoints and datasets are distributed separately and are not covered
by the source-code license. Review the terms on the corresponding download page
before use or redistribution.
