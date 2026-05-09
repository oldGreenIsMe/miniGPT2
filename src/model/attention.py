import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import GPTConfig


KVCache = Tuple[torch.Tensor, torch.Tensor]


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

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
    ):
        """
        x:
            no cache  : [B, T, C]
            with cache: [B, T_new, C]

        past_kv:
            None or (past_k, past_v)

            past_k: [B, nh, T_past, hs]
            past_v: [B, nh, T_past, hs]

        return:
            y: [B, T, C] or [B, T_new, C]

            present_kv:
                None, if use_cache=False
                (k_all, v_all), if use_cache=True
        """
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv

            k_all = torch.cat([past_k, k], dim=2)
            v_all = torch.cat([past_v, v], dim=2)
        else:
            k_all = k
            v_all = v

        present_kv = (k_all, v_all) if use_cache else None

        att = q @ k_all.transpose(-2, -1)
        att = att / math.sqrt(self.head_size)

        T_key = k_all.size(2)

        if past_kv is None:
            att = att.masked_fill(
                self.causal_mask[:, :, :T, :T_key] == 0,
                float("-inf"),
            )
        else:
            # Decode 阶段通常 T=1，新 token 位于序列最右侧，
            # 它可以看所有 past token 和自己，因此不需要额外 causal mask。
            pass

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v_all

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y, present_kv