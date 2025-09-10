# Aggregates models *and* aggregates bias statistics, then trains the λ‑policy.
import torch
import flwr as fl
from policy_network import LambdaPolicy
from client_BIAS import BiasTracker
import copy
import numpy as np

class FedServer(fl.server.strategy.FedAvg):
    """
    Extends FedAvg to:
      • compute aggregated BIAS stats
      • train λ‑policy network (λ_policy)
    """
    def __init__(self,
                 policy: LambdaPolicy,
                 device: torch.device,
                 **kwargs):
        super().__init__(**kwargs)
        self.policy = policy
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()
        self.bias_tracker = BiasTracker()

    # ---- 5️⃣1  After aggregation hook -----
    def aggregate_fit(self, rnd: int,
                      results: list[tuple[fl.common.Peer, fl.common.ModelParameters, float]],
                      failures: list[BaseException]) -> tuple[fl.common.ModelParameters | None, dict[str,
fl.common.StrategyEvaluation]]:
        # Standard FedAvg aggregation
        aggregated_params, metrics = super().aggregate_fit(rnd, results, failures)

        # ---- 5️⃣2  Compute BIAS stats across clients -----
        # We ask each client to run its bias_tracker on its local validation set
        # For demonstration, we emulate this by reading a stored bias file
        # (clients should store `bias_score.npy` after training)
        bias_scores = []
        for peer, _, _ in results:
            # Suppose each peer sends its bias in the `extra` dict
            bias_scores.append(peer.extra.get("bias_score", 0.0))

        mean_bias = np.mean(bias_scores)
        print(f"[Server] Round {rnd} mean bias: {mean_bias:.4f}")

        # ---- 5️⃣3  Update λ‑policy to reduce bias -----
        # Goal: map current mean bias → λ that improves fairness.
        # Here we simply train λ to *minimize* the mean bias (MSE loss).
        # In practice you could use a more sophisticated objective.
        target = torch.tensor([0.0], device=self.device)          # 0 bias is perfect fairness
        input_bias = torch.tensor([mean_bias], device=self.device)
        pred_lambda = self.policy(input_bias)
        loss = self.loss_fn(pred_lambda, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"[Server] λ-policy loss: {loss.item():.6f}")

        return aggregated_params, metrics

# > **How it works**
# > * Each client sends its **bias score** in the `extra` dict (e.g., `peer.extra["bias_score"] = computed_bias`).
# > * The server averages those scores, logs them, and then *trains* the λ‑policy to push the λ value towards
# reducing bias.
# > * The λ value returned by the policy is applied locally by the client in `client_update` (see `client.py`).
