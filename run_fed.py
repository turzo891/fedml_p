# Runs a full round‑by‑round simulation.
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
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
def load_biased_mnist(num_clients=5, samples_per_client=1000, seed=42):
    transform = T.Compose([T.ToTensor()])
    dataset = torchvision.datasets.MNIST(root="./data",
                                         train=True,
                                         download=True,
                                         transform=transform)

    rng = np.random.default_rng(seed)
    # We encode a binary label: 0 if digit==0 else 1
    labels = (dataset.targets != 0).int()

    # Inject a synthetic sensitive attribute:
    #  - group 0: all digits 0
    #  - group 1: all digits 1
    sensitive = labels.clone()

    # Shuffle and split
    indices = rng.permutation(len(dataset))
    client_indices = np.array_split(indices, num_clients)

    clients = []
    for idx in client_indices:
        subset = Subset(dataset, idx[:samples_per_client])
        subset.targets = labels[idx[:samples_per_client]]
        subset.sensitive = sensitive[idx[:samples_per_client]]
        clients.append(subset)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = LambdaPolicy().to(device)

    # Data
    clients = load_biased_mnist(num_clients=num_clients)

    # Build Flower clients
    client_objs = build_clients(clients, device, policy)

    # Server strategy
    strategy = FedServer(policy=policy,
                         device=device,
                         fraction_fit=1.0,          # use all clients each round
                         fraction_eval=1.0,
                         min_fit_clients=num_clients,
                         min_eval_clients=num_clients,
                         min_available_clients=num_clients,
                         initial_parameters=fl.common.ndarray_to_parameters(
                             [p.detach().cpu().numpy() for p in SimpleCNN().parameters()]
                         ),
                         use_val_loader=True,
                         val_evaluation_fn=ClientEvaluator()  # optional
                         )

    # Run simulation
    fl.simulation.start_simulation(
        client_fn=lambda cid: client_objs[cid],   # id → client
        num_clients=num_clients,
        strategy=strategy,
        config={"rounds": rounds, "local_epochs": 2}
    )

# ----------------------------------------------------------
# 6️⃣5  (Optional) Client evaluator – logs bias each round
# ----------------------------------------------------------
class ClientEvaluator(fl.server.strategy.EvaluateServerFn):
    def __init__(self):
        super().__init__()

    def __call__(self, parameters, config):
        # This function runs on the *server* to evaluate a model.
        # We can compute global fairness metrics here if we have
        # access to a hold‑out fair dataset.
        return None, 0.0, {}

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
