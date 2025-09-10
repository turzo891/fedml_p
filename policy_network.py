import torch
import torch.nn as nn
import torch.nn.functional as F

class LambdaPolicy(nn.Module):
    """
    Maps a client’s bias score to a scalar weight λ ∈ (0, 1].
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()        # ensures output in (0,1)
        )

    def forward(self, bias_score: torch.Tensor) -> torch.Tensor:
        return self.net(bias_score.unsqueeze(1))
