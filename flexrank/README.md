<p align="center">
  <a href="https://rickzack.github.io/FlexRank/">
    <img src="https://rickzack.github.io/FlexRank/static/images/FlexRank_logo.png" alt="FlexRank logo" width="180">
  </a>
</p>

# flexrank

Core package for **FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model
Deployment**.

`flexrank` is independent of the training stack. It provides ordered low-rank layers,
SVD/DataSVD decomposition utilities, Gauge-Aligned Reparametrization helpers, Gram-matrix
collection, dynamic-programming profile search, samplers, and lightweight trainer abstractions.

Links:

- Project page: https://rickzack.github.io/FlexRank/
- Paper: https://openreview.net/forum?id=DK0kvnNelx
- arXiv: https://arxiv.org/abs/2602.02680
- Full repository README: `../README.md`

Install from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the core package tests from the repository root:

```bash
python -m pytest flexrank/tests -q
```

Authors: Riccardo Zaccone, Stefanos Laskaridis, Marco Ciccone, and Samuel Horvath.
