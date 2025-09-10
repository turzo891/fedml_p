# Replaces the vanilla `local_update` with `client_update` that returns the bias score and applies the λ weight
import torch
from torch import nn, optim
import flwr as fl
from client_BIAS import BiasTracker
from policy_network import LambdaPolicy

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 device: torch.device,
                 policy: LambdaPolicy,
                 bias_tracker: BiasTracker):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.policy = policy
        self.bias_tracker = bias_tracker
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
 # --- 4️⃣1  Client‑specific helper  --------------------------
    def _compute_lambda(self, bias):
        """Map bias → λ via the policy network."""
        bias_tensor = torch.tensor([bias], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            lam = self.policy(bias_tensor).item()
        return lam

    # --- 4️⃣2  Local training with bias‑aware weighting -------
    def client_update(self, parameters, config):
        self.model.load_state_dict(parameters)
        self.model.to(self.device)

        # Forward pass on validation set to compute bias
        bias = self.bias_tracker.compute(self.model, self.val_loader, self.device)

        # λ from policy
        lam = self._compute_lambda(bias)

        # ---------- Local training ----------
        self.model.train()
        for epoch in range(config.get("local_epochs", 1)):
            for X, y, _ in self.train_loader:
                X = X.to(self.device)
                y = y.to(self.device).float()
                self.optimizer.zero_grad()
                logits = self.model(X).squeeze()
                loss = self.criterion(logits, y)
                # Re‑weight loss by λ (encourages fairer models)
                loss = lam * loss
                loss.backward()
                self.optimizer.step()

        return self.model.state_dict(), len(self.train_loader.dataset), {}


# **NOTE** – In a real setting you would also pass the bias score back to the server for aggregation, but for simplicity we only return the updated parameters here.  The server will query the bias locally  
