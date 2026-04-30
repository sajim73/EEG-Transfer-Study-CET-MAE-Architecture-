import torch.nn as nn


class LinearProbeHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MLPProbeHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_probe_head(head_type: str, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1):
    head_type = head_type.lower()
    if head_type in {"linear", "probe", "linear_probe"}:
        return LinearProbeHead(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
    if head_type in {"mlp", "two_layer", "nonlinear"}:
        return MLPProbeHead(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unsupported head_type: {head_type}")
