import torch
import torch.nn as nn

from src.config import GPTConfig


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.c_fc = nn.Linear(
            in_features=config.n_embd,
            out_features=4 * config.n_embd,
            bias=config.bias,
        )

        self.gelu = nn.GELU()

        self.c_proj = nn.Linear(
            in_features=4 * config.n_embd,
            out_features=config.n_embd,
            bias=config.bias,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]

        return:
            x: [B, T, C]
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x