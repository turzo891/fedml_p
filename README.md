# Federated Bias-Aware Training (Flower + PyTorch)

This repo demonstrates a simple federated learning simulation that measures fairness and adapts training using a learned loss-weight policy. It uses Flower for orchestration and PyTorch for modeling.

- Binary classifier on MNIST (0 vs non-0)
- Per-client fairness metric: BIAS = |AUC(group0) − AUC(group1)|
- Server learns a policy λ(bias) and broadcasts a scalar λ to clients
- Server can run on GPU; clients run on CPU by default

## Quick Start (Windows PowerShell)

1) Create and activate a virtualenv

- Create: `py -3.10 -m venv .venv`
- Activate: `.\.venv\Scripts\Activate.ps1`

2) Upgrade packaging tools

- `python -m pip install -U pip setuptools wheel`

3) Install PyTorch

Pick ONE of the following:

- GPU (CUDA 11.8 example):
  - `python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio`
  - Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`

- CPU-only:
  - `python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio`

4) Install Flower (simulation) and other deps

- `pip install "flwr[simulation]>=1.7,<2.0" numpy scikit-learn`

5) Run the simulation

- `python run_fed.py`

Notes
- First run downloads MNIST to `./data`.
- Using `start_simulation()` prints a deprecation warning; it’s fine for this demo.

## How It Works

Round loop at a glance:
- Each client computes validation fairness (BIAS) on its local val split.
- Server aggregates BIAS across clients and trains a small policy network λ = f(bias) to reduce mean bias.
- Server broadcasts the current λ to clients; clients scale their local loss by λ during training.

Key files:
- `run_fed.py`: Entry point. Builds data, clients, and the Flower strategy; wires GPU server/CPU clients behavior.
- `client.py`: Implements `fl.client.NumPyClient` with bias-aware local training and metrics.
- `server.py`: Extends Flower FedAvg to aggregate client metrics and update the λ policy; broadcasts λ via `on_fit_config_fn`.
- `policy_network.py`: Tiny MLP mapping bias → λ in (0, 1].
- `client_BIAS.py`: Computes BIAS = |AUC(group0) − AUC(group1)| from model scores.
- `fairness_metrics.py`: Standalone helpers for fairness evaluation.
- `explainability.py`: Optional SHAP explainer (not used in the main loop).

## Configuration

Edit `run_fed.py` to change the following:
- Rounds/clients: `run_fed.py:156` calls `main(rounds=20, num_clients=5)`
- Local epochs per round: set in `fit_config_fn` (`run_fed.py:132`)
- GPU vs CPU:
  - Server device auto-selects CUDA if available (`run_fed.py:119`).
  - Clients are forced to CPU to avoid Ray GPU serialization issues.
- Concurrency/resources (optional): pass `client_resources` to `fl.simulation.start_simulation(...)`, for example:
  - `client_resources={"num_cpus": 1, "num_gpus": 0.0}`

## Expected Logs

- Server prints mean bias and λ-policy loss each round, e.g.:
  - `[Server] Round 1 mean bias: 0.1832`
  - `[Server] λ-policy loss: 0.032451; λ=0.7421`
- Clients report metrics including `bias_score` and `val_accuracy`.

## Dataset & Fairness Caveat

For demonstration, the “sensitive attribute” is derived from the label (0 vs non-0), which makes bias measurement trivial. In real use, replace this with a true sensitive attribute per sample.

- Data wrapper: `BiasedClientDataset` returns `(image, label, sensitive)` and is split into train/val per client.

## Troubleshooting

- Flower deprecation: `start_simulation()` is deprecated; this demo still uses it. For production, prefer the `flwr run` CLI.
- CUDA deserialization error: Ensure you installed a CUDA-enabled PyTorch build matching your driver (see step 3), or run CPU-only.
- Ray/Resource issues: Reduce `num_clients`, set `rounds` lower, or specify `client_resources` to limit concurrency.
- Torch/Torchvision mismatch: Install matching versions from the same index (GPU or CPU).

## Use Case

This pattern suits scenarios where global fairness is a goal but only local (per-client) groups/labels are available. Example: multiple clinics training a model where group membership is sensitive; each clinic computes its own fairness metric, the server learns a policy to nudge training toward a fairer global model via a scalar λ broadcast every round.

## Repo Structure

- `run_fed.py` — main simulation script
- `client.py` — Flower NumPyClient implementation
- `server.py` — Strategy extending FedAvg with fairness-aware aggregation
- `policy_network.py` — bias → λ policy network
- `client_BIAS.py` — BIAS computation helper
- `fairness_metrics.py` — metrics utilities
- `explainability.py` — SHAP-based local explainer (optional)

## Next Steps

- Replace the synthetic sensitive attribute with your own group labels.
- Log fairness metrics over time and add plots.
- Integrate `explainability.py` to analyze per-sample attributions.
- Add CLI flags for rounds/clients and configurable datasets.
