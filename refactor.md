# Release Preparation Plan

## Summary

Prepare this repository for a clean GitHub/tagged release without changing runtime behavior. Create a local virtual environment for validation:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Use `--no-build-isolation` with the install commands in restricted environments that cannot fetch build dependencies during editable local-package setup.

Current baseline from the release-prep pass:

- `compileall` passes.
- `pytest` collects 40 tests.
- Full test suite result: 39 passed, 1 skipped.
- `flexrank` and `flextrain` import successfully from the repository.
- Ruff is configured and passes with the release-blocker rule set.
- `flexrank` and `flextrain` build successfully with `--no-isolation`.
- `twine check` passes for both generated wheels and source distributions.
- Runtime installation uses the top-level `requirements.txt`.
- Development tools are declared in package `dev` extras and installed through `requirements-dev.txt`.
- Notebook outputs are intentionally retained and should not be cleared.
- `detect-secrets` reports no raw values, but it flags high-entropy notebook output and one model config entry for manual review.
- The pre-existing worktree changes to README/script files and tracked deletions are intentional.
- Local generated data/model artifacts remain ignored by default.

## Implementation Checklist

- [x] Add this release plan as `refactor.md`.
- [x] Ignore local/generated ML artifacts so datasets, checkpoints, notebook data, model weights, and experiment outputs do not enter the release by accident.
- [x] Add stable pytest discovery configuration.
- [x] Add conservative Ruff lint configuration for future cleanup.
- [x] Add development requirements for release validation tools under package `dev` extras.
- [x] Fix release-blocking Ruff findings.
- [x] Add package README metadata so distribution checks pass cleanly.
- [x] Confirm whether existing deletions of tracked files are intentional:
  - `flextrain/requirements/requirements.txt`
  - `flextrain/utils/torch_compile.py`
- [x] Review existing README and script edits and keep only intentional release changes.
- [x] Populate README files with meaningful repository and package descriptions.
- [x] Run a secret scan that reports file paths/rule IDs only, never secret values.
- [ ] Manually review `detect-secrets` high-entropy findings in tracked notebooks and the model config entry.
- [ ] Fix only release-blocking warnings/errors or low-risk correctness issues.
- [x] Build and inspect release artifacts after the repository is clean.

## Sensitive-Data Policy

- Do not commit API keys, passwords, auth headers, access tokens, cloud credentials, private keys, machine-specific absolute paths, or personal config.
- Treat notebooks, YAML configs, shell scripts, model cards/configs, and training utilities as high-risk surfaces.
- Report sensitive-data findings by category and file path only. Do not paste discovered values into issues, commits, PRs, logs, or chat.
- Generated binary artifacts such as `*.pth`, `*.pkl`, `*.safetensors`, and local dataset archives are excluded from release by default.

## Validation Commands

Run all validation after activating the virtual environment:

```bash
python -m compileall -q flexrank/src flextrain/src train.py finetune.py flexrank/tests flextrain/tests
python -m pytest -q
python - <<'PY'
import importlib

for name in ("flexrank", "flextrain"):
    importlib.import_module(name)
    print(f"{name}: ok")
PY
```

When Ruff is installed in the environment, run lint without auto-fixes first:

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
```

Run a path-only sensitive-term sweep before staging release changes:

```bash
rg -l -i '(api[_-]?key|secret|token|password|passwd|private[_-]?key|aws_access|aws_secret|huggingface|hf_token|wandb|openai|github_token|authorization|bearer|/home/|/Users/|C:\\)' \
  --glob '!*.pth' \
  --glob '!*.pkl' \
  --glob '!*.safetensors' \
  --glob '!*.tar.gz' \
  --glob '!data/**' \
  --glob '!notebooks/data/**' \
  --glob '!notebooks/notebook_data/**' \
  --glob '!notebooks/outputs/**' \
  .
```

## Release Gates

- Full test suite passes with no new skips.
- `flexrank` and `flextrain` import from the repository.
- No generated datasets, checkpoints, generated notebook data, or experiment logs are staged.
- Tracked notebook outputs are intentionally retained.
- Secret scan has no committed credentials or sensitive local paths.
- `git status --short` contains only intentional release-prep changes.
- Public APIs, module names, training defaults, and runtime behavior remain unchanged unless a specific release blocker requires a targeted fix.
