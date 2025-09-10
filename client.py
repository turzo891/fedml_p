# Replaces the vanilla `local_update` with `client_update` that returns the bias score and applies the λ weight
import torch
from torch import nn, optim
import flwr as fl
from client_BIAS import BiasTracker
from policy_network import LambdaPolicy
from collections import OrderedDict

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 device: torch.device,
                 policy: LambdaPolicy | None,
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
        """Kept for backward-compatibility; not used by Flower wrapper."""
        # Convert a state_dict (if provided) and train; otherwise no-op
        if isinstance(parameters, dict):
            self.model.load_state_dict(parameters)
        self.model.to(self.device)

        bias = self.bias_tracker.compute(self.model, self.val_loader, self.device)
        lam = self._compute_lambda(bias)

        self.model.train()
        for epoch in range(config.get("local_epochs", 1)):
            for X, y, _ in self.train_loader:
                X = X.to(self.device)
                y = y.to(self.device).float()
                self.optimizer.zero_grad()
                logits = self.model(X).squeeze()
                loss = self.criterion(logits, y)
                loss = lam * loss
                loss.backward()
                self.optimizer.step()

        return self.model.state_dict(), len(self.train_loader.dataset), {"bias_score": float(bias), "lambda": float(lam)}

    # ---- NumPyClient required methods ------------------------
    def get_parameters(self, config):
        """Return model weights as a list of NumPy arrays."""
        return [v.detach().cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model weights from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)

        # Compute bias on validation set
        bias = self.bias_tracker.compute(self.model, self.val_loader, self.device)
        # Use λ provided by server (preferred), else local policy (if any), else 1.0
        lam = float(config.get("lambda", 1.0)) if isinstance(config, dict) else 1.0
        if (not isinstance(config, dict) or "lambda" not in config) and self.policy is not None:
            lam = self._compute_lambda(bias)

        # Local training
        self.model.train()
        local_epochs = int(config.get("local_epochs", 1)) if isinstance(config, dict) else 1
        for _ in range(local_epochs):
            for X, y, _ in self.train_loader:
                X = X.to(self.device)
                y = y.to(self.device).float()
                self.optimizer.zero_grad()
                logits = self.model(X).squeeze()
                loss = self.criterion(logits, y)
                loss = lam * loss
                loss.backward()
                self.optimizer.step()

        metrics = {"bias_score": float(bias), "lambda": float(lam)}
        return self.get_parameters(config), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for X, y, _ in self.val_loader:
                X = X.to(self.device)
                y = y.to(self.device).float()
                logits = self.model(X).squeeze()
                loss = self.criterion(logits, y)
                total_loss += loss.item() * X.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y).sum().item()
                total += X.size(0)

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        metrics = {"val_accuracy": float(acc)}
        return float(avg_loss), total, metrics


# **NOTE** – In a real setting you would also pass the bias score back to the server for aggregation, but for simplicity we only return the updated parameters here.  The server will query the bias locally  
