<p align="center">
  <a href="https://rickzack.github.io/FlexRank/">
    <img src="https://rickzack.github.io/FlexRank/static/images/FlexRank_logo.png" alt="FlexRank logo" width="180">
  </a>
</p>

# flextrain

Training and experiment package for **FlexRank: Nested Low-Rank Knowledge Decomposition for
Adaptive Model Deployment**.

`flextrain` depends on `flexrank` and adds Hugging Face model/dataset loaders, Hydra
configuration, Accelerate launch settings, W&B integration, callbacks, evaluation, checkpointing,
and `FlexRankModel` save/load support.

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

Run the training-side tests from the repository root:

```bash
python -m pytest flextrain/tests -q
```

Authors: Riccardo Zaccone, Stefanos Laskaridis, Marco Ciccone, and Samuel Horvath.
