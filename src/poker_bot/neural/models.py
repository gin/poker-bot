import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Policy Network for action selection.
    Input: State feature vector
    Output: Probability distribution over actions
    v2 (NN-IMPROVE-005): deeper (256→128→64) + dropout for Flop/Turn
    """

    def __init__(self, input_dim: int, num_actions: int = 5, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(128, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop(x)
        x = self.fc3(x)
        return self.out(x)  # logits for CrossEntropyLoss


class ValueNetwork(nn.Module):
    """
    Value Network for EV estimation.
    Input: State feature vector
    Output: Scalar estimated EV
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_model_dims(encoder_sample: torch.Tensor):
    """Helper to determine input dimensions from a sample."""
    return encoder_sample.shape[1]


# Action mapping for reference
ACTION_MAP = {0: "fold", 1: "call", 2: "check", 3: "raise", 4: "all-in"}
