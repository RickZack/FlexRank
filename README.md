<p align="center">
  <a href="https://rickzack.github.io/FlexRank/">
    <img src="https://rickzack.github.io/FlexRank/static/images/FlexRank_logo.png" alt="FlexRank logo" width="240">
  </a>
</p>

<h1 align="center">FlexRank</h1>

<p align="center">
  <strong>Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment</strong>
</p>

<p align="center">
  <a href="https://rickzack.github.io/FlexRank/">Project page</a> |
  <a href="https://openreview.net/forum?id=DK0kvnNelx">Paper</a> |
  <a href="https://arxiv.org/abs/2602.02680">arXiv</a>
</p>

## Overview

This is the official repository for **FlexRank: Nested Low-Rank Knowledge Decomposition for
Adaptive Model Deployment**.

FlexRank converts a pretrained model into a family of nested low-rank submodels that share the
same weights. At deployment time, the active rank profile can be selected according to the target
parameter budget, making one consolidated model usable across multiple resource regimes.

Authors: Riccardo Zaccone, Stefanos Laskaridis, Marco Ciccone, and Samuel Horvath.

## Repository Structure

The codebase is split into two installable packages:

- `flexrank`: the independent core package. It contains ordered low-rank layers, SVD/DataSVD
  decomposition utilities, Gauge-Aligned Reparametrization helpers, Gram-matrix collection,
  dynamic-programming profile search, samplers, and lightweight trainer abstractions.
- `flextrain`: the experiment and training package. It builds on `flexrank` with Hugging Face
  model/dataset loaders, Hydra configs, Accelerate launch settings, W&B logging, callbacks,
  evaluation, checkpointing, and save/load support.

Important paths:

- `flexrank/src/flexrank/`: core layers, profile search, samplers, trainers, shared types, and
  utilities.
- `flexrank/tests/`: tests for the core package without `flextrain` dependencies.
- `flextrain/src/flextrain/`: training configs, dataset/model loading, callbacks, and training
  utilities.
- `flextrain/tests/`: integration tests for training-side model serialization.
- `train.py`: Hydra entrypoint for FlexRank knowledge-consolidation training.
- `finetune.py`: Hydra entrypoint for baseline fine-tuning.
- `scripts/`: launch scripts for NLP and computer-vision experiment families.
- `notebooks/`: release notebooks for save/load, pruning, and toy linear-model analysis.

## Installation

Use Python 3.11 or newer. From the repository root, create a fresh virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the runtime packages and dependencies through the top-level requirements file:

```bash
python -m pip install -r requirements.txt
```

For tests, linting, coverage, release builds, and secret scanning:

```bash
python -m pip install -r requirements-dev.txt
```

Runtime dependencies are declared in each package `pyproject.toml`. Development-only tools live
under each package's `dev` optional dependency section; `requirements-dev.txt` is the repository
convenience wrapper around those extras.

## Package Map

### `flexrank`

- `flexrank.layers`: ordered low-rank linear and convolutional layers, SVD/DataSVD decomposition,
  Gauge-Aligned Reparametrization, and Gram-matrix utilities.
- `flexrank.profiles`: profile containers and dynamic-programming rank-profile search.
- `flexrank.samplers`: all-layer, single-layer, and predefined-profile sampling strategies.
- `flexrank.trainers`: trainer interfaces and concrete utilities for standard, SVD, and FlexRank
  training loops.
- `flexrank.types` and `flexrank.utils`: decomposition enums, distributed metadata, logging,
  model traversal, and training helpers.

### `flextrain`

- `flextrain.config`: Hydra groups for models, datasets, decomposition, DP profile search,
  samplers, training, evaluation, Accelerate, lm-eval, and logging.
- `flextrain.dataset`: Hugging Face dataset loading for language-modeling and
  image-classification experiments.
- `flextrain.model`: Hugging Face model loading, DINOv3 image-classification wrappers, Conv1D
  replacement, and `FlexRankModel` save/load support.
- `flextrain.callbacks`: checkpointing, evaluation, sampler updates, lm-eval, and post-training
  profile generation.
- `flextrain.utils`: structured config dataclasses, trainer helpers, FlexRank initialization,
  W&B setup, distributed utilities, plotting, and export helpers.

## Running Experiments

The main FlexRank training entrypoint is Hydra-based:

```bash
python train.py
```

The baseline fine-tuning entrypoint uses a separate Hydra config:

```bash
python finetune.py
```

Release launch scripts are provided for the two experiment families:

```bash
bash scripts/flexrank_nlp.sh model=gpt2 dataset=finewebedu_10bt
bash scripts/flexrank_cv.sh model=vit_base_16 dataset=imagenet1k pretrain_vision_path=/path/to/pretrained_vision_models
```

The NLP script requires `model` and `dataset`. The CV script also requires
`pretrain_vision_path`, pointing to a local directory with pretrained vision checkpoints.
Additional Hydra `key=value` overrides can be appended to either command.

Common Hydra config groups live under `flextrain/src/flextrain/config/`:

- `model`: GPT-2, Llama-family, and ViT-family presets.
- `dataset`: FineWeb-Edu, CIFAR-100, and ImageNet-1k presets.
- `decomposition`: SVD and DataSVD decomposition settings.
- `profile_algo`: DP, fast-DP, and no-search profile modes.
- `sampler`: all-layer, single-layer, and predefined-profile samplers.
- `train`, `eval`, `accelerate`, `lm_eval`, `logger`: runtime, evaluation, distributed,
  downstream-eval, and logging settings.

## Validation

After activating the virtual environment:

```bash
python -m compileall -q flexrank/src flextrain/src train.py finetune.py flexrank/tests flextrain/tests
python -m ruff check .
python -m pylint ./
python -m pytest -ra
```

Package builds can be checked with:

```bash
python -m build --no-isolation flexrank
python -m build --no-isolation flextrain
python -m twine check flexrank/dist/* flextrain/dist/*
```

For a release-oriented secret scan, exclude ignored local caches and known notebook image payloads:

```bash
python -m detect_secrets scan \
  --exclude-files '^(\\.pytest_cache|\\.ruff_cache|notebooks/\\.ipynb_checkpoints)/' \
  --exclude-lines 'image/png|facebook/dinov3-vith16plus-pretrain-lvd1689m'
```

## Citation

```bibtex
@inproceedings{
  zaccone2026flexrank,
  title={FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment},
  author={Zaccone, Riccardo and Laskaridis, Stefanos and Ciccone, Marco and Horvath, Samuel},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=DK0kvnNelx}
}
```
