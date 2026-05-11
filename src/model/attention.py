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
        return_attn: bool = False,
    ):
        """
        x:
            no cache  : [B, T, C]
            with cache: [B, T_new, C]

        return:
            y: [B, T, C]
            present_kv: None or (k_all, v_all)
            att_to_return:
                None or [B, n_head, T, T_key]
        """
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
        
            # past_k / past_v: [B, nh, T_past, hs]
            # k / v:           [B, nh, T_new, hs]
            #
            # 为了 rolling cache，总长度不能超过 block_size。
            # 当前输入长度是 T，所以历史最多保留 block_size - T 个 token。
            max_past_len = self.block_size - T
        
            if max_past_len < 0:
                raise ValueError(
                    f"Current input length T={T} exceeds block_size={self.block_size}"
                )
        
            if past_k.size(2) > max_past_len:
                past_k = past_k[:, :, -max_past_len:, :]
                past_v = past_v[:, :, -max_past_len:, :]
        
            k_all = torch.cat([past_k, k], dim=2)
            v_all = torch.cat([past_v, v], dim=2)
        else:
            k_all = k
            v_all = v
        
        # 如果 prefill 阶段输入本身超过 block_size，也裁剪到最后 block_size
        if k_all.size(2) > self.block_size:
            k_all = k_all[:, :, -self.block_size:, :]
            v_all = v_all[:, :, -self.block_size:, :]

        present_kv = (k_all, v_all) if use_cache else None

        att = q @ k_all.transpose(-2, -1)
        att = att / math.sqrt(self.head_size)

        T_key = k_all.size(2)

        if past_kv is None:
            att = att.masked_fill(
                self.causal_mask[:, :, :T, :T_key] == 0,
                float("-inf"),
            )

        att = F.softmax(att, dim=-1)

        att_to_return = att.detach().cpu() if return_attn else None

        att = self.attn_dropout(att)

        y = att @ v_all

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y, present_kv, att_to_return