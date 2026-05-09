import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0, (
            "n_embd must be divisible by n_head"
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.block_size = config.block_size
        self.dropout = config.dropout

        self.c_attn = nn.Linear(
            in_features=config.n_embd,
            out_features=3 * config.n_embd,
            bias=config.bias,
        )

        self.c_proj = nn.Linear(
            in_features=config.n_embd,
            out_features=config.n_embd,
            bias=config.bias,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]

        return:
            y: [B, T, C]
        """
        B, T, C = x.shape

        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = q @ k.transpose(-2, -1)
        att = att / math.sqrt(self.head_size)

        att = att.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float("-inf"),
        )

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y