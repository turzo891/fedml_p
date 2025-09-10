# Runs a full round‑by‑round simulation.
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np

from client_BIAS import BiasTracker
from policy_network import LambdaPolicy
from client import FlowerClient
from server import FedServer

# ----------------------------------------------------------
# 6️⃣1  Simple two‑class classifier (CNN)
# ----------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----------------------------------------------------------
# 6️⃣2  Generate a *biased* MNIST split (digit 0 vs 1)
# ----------------------------------------------------------
class BiasedClientDataset(Dataset):
    """Wrap MNIST to return (image, binary_label, sensitive_group)."""

    def __init__(self, base_ds, indices, labels, sensitive):
        self.base_ds = base_ds
        self.indices = np.array(indices)
        self.labels = labels
        self.sensitive = sensitive

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x, _ = self.base_ds[idx]
        y = self.labels[idx].item() if hasattr(self.labels[idx], 'item') else int(self.labels[idx])
        g = self.sensitive[idx].item() if hasattr(self.sensitive[idx], 'item') else int(self.sensitive[idx])
        return x, torch.tensor(y, dtype=torch.long), torch.tensor(g, dtype=torch.long)

def load_biased_mnist(num_clients=5, samples_per_client=1000, seed=42):
    """Create a 0-vs-1 MNIST split and a balanced sensitive attribute.

    - Labels: y=1 for digit==1, else 0 (digit==0)
    - Sensitive attribute g ~ Bernoulli(p), correlated with y but not identical
      (p=0.7 if y==1 else p=0.3) so both groups contain both classes.
    - Each client receives a balanced mix of class 0/1.
    """
    transform = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    digits = ds.targets
    mask01 = (digits == 0) | (digits == 1)
    idx_all = np.nonzero(mask01.numpy())[0]

    # Build full-length label/sensitive arrays (aligned with original indices)
    labels_full = torch.zeros_like(digits).int()
    labels_full[digits == 1] = 1

    rng = np.random.default_rng(seed)
    sens_np = rng.binomial(1, p=(labels_full.numpy() * 0.4 + 0.3))  # p=0.3 if y=0 else 0.7
    sensitive_full = torch.from_numpy(sens_np).int()

    # Separate indices by class for balancing per client
    idx0 = idx_all[digits[idx_all].numpy() == 0]
    idx1 = idx_all[digits[idx_all].numpy() == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    # Allocate per-client
    per_c0 = samples_per_client // 2
    per_c1 = samples_per_client - per_c0
    clients = []
    ptr0 = ptr1 = 0
    for _ in range(num_clients):
        sel0 = idx0[ptr0:ptr0 + per_c0]
        sel1 = idx1[ptr1:ptr1 + per_c1]
        ptr0 += per_c0
        ptr1 += per_c1
        sel = np.concatenate([sel0, sel1])
        rng.shuffle(sel)
        clients.append(BiasedClientDataset(ds, sel, labels_full, sensitive_full))

    return clients

# ----------------------------------------------------------
# 6️⃣3  Build clients
# ----------------------------------------------------------
def build_clients(clients, device, policy):
    client_objs = []
    for client_id, subset in enumerate(clients):
        # 60% train, 40% val split
        n = len(subset)
        train_idx = np.arange(int(0.6 * n))
        val_idx = np.arange(int(0.6 * n), n)

        train_set = Subset(subset, train_idx)
        val_set = Subset(subset, val_idx)

        train_loader = DataLoader(train_set,
                                  batch_size=32,
                                  shuffle=True)
        val_loader = DataLoader(val_set,
                                batch_size=32,
                                shuffle=False)

        bias_tracker = BiasTracker(group_label=0)  # group 0 is reference

        client = FlowerClient(
            model=SimpleCNN(),
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            policy=policy,
            bias_tracker=bias_tracker
        )
        client_objs.append(client)
    return client_objs

# ----------------------------------------------------------
# 6️⃣4  Main experiment
# ----------------------------------------------------------
def main(rounds=10, num_clients=5):
    # Server on GPU if available; clients run on CPU
    server_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = LambdaPolicy().to(server_device)

    # Data
    clients = load_biased_mnist(num_clients=num_clients)

    # Build Flower clients (CPU only; no policy object to avoid GPU serialization)
    client_objs = build_clients(clients, torch.device("cpu"), policy=None)

    # Server strategy
    strategy = FedServer(policy=policy,
                         device=server_device,
                         fraction_fit=1.0,          # prefer using all clients each round
                         min_fit_clients=num_clients,
                         min_available_clients=num_clients,
                         )

    # Provide per-round fit config including the server-computed λ
    def fit_config_fn(server_round: int):
        return {"local_epochs": 2, "lambda": float(getattr(strategy, "current_lambda", 1.0))}

    # Attach config function after instantiation for compatibility
    strategy.on_fit_config_fn = fit_config_fn

    # Run simulation
    fl.simulation.start_simulation(
        client_fn=lambda cid: client_objs[int(cid)].to_client(),   # return Client, not NumPyClient
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

# ----------------------------------------------------------
# 6️⃣5  (Optional) Client evaluator – logs bias each round
# ----------------------------------------------------------
class ClientEvaluator:
    pass

if __name__ == "__main__":
    main(rounds=20, num_clients=5)

# > **What this script does**
# > 1. Generates a toy *biased* MNIST split.
# > 2. Creates a `FlowerClient` for each shard.
# > 3. Spins up a Flower simulation with the custom `FedServer`.
# > 4. Every round:
# >    * Clients compute their bias → λ → train locally.
# >    * Server aggregates weights and λ‑policy.
# >    * λ‑policy learns to reduce mean bias.

## 7️⃣  Evaluation (fairness metrics)
# Below is a simple utility you can import into `run_fed.py` or run as a separate script to compute per‑client fairness metrics **after each round**.
