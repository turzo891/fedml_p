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
        # Value to broadcast to clients in next round via on_fit_config_fn
        self.current_lambda: float = 1.0

    # ---- 5️⃣1  After aggregation hook -----
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results and update λ‑policy using client bias metrics.

        Compatible with Flower 1.x where `results` is a list of
        (ClientProxy, FitRes) tuples and metrics are in `FitRes.metrics`.
        """
        # Standard FedAvg aggregation
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        # ---- 5️⃣2  Compute BIAS stats across clients -----
        bias_scores = []
        for _client, fit_res in results:
            m = getattr(fit_res, "metrics", None) or {}
            if "bias_score" in m:
                try:
                    bias_scores.append(float(m["bias_score"]))
                except Exception:
                    pass

        # Drop NaNs if any
        bias_scores = [b for b in bias_scores if b == b]  # b==b filters out NaN
        if bias_scores:
            mean_bias = float(np.mean(bias_scores))
            print(f"[Server] Round {server_round} mean bias: {mean_bias:.4f}")

            # ---- 5️⃣3  Update λ‑policy to reduce bias -----
            target = torch.tensor([0.0], device=self.device, dtype=torch.float32)  # perfect fairness
            input_bias = torch.tensor([mean_bias], device=self.device, dtype=torch.float32)
            self.policy.to(self.device)
            pred_lambda = self.policy(input_bias).view(-1)
            loss = self.loss_fn(pred_lambda, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Store scalar lambda (CPU) for broadcasting in next round
            self.current_lambda = float(pred_lambda.detach().cpu().item())
            print(f"[Server] λ-policy loss: {loss.item():.6f}; λ={self.current_lambda:.4f}")
        else:
            print(f"[Server] Round {server_round} received no bias metrics")

        return aggregated_params, metrics

# > **How it works**
# > * Each client sends its **bias score** in the `extra` dict (e.g., `peer.extra["bias_score"] = computed_bias`).
# > * The server averages those scores, logs them, and then *trains* the λ‑policy to push the λ value towards
# reducing bias.
# > * The λ value returned by the policy is applied locally by the client in `client_update` (see `client.py`).
