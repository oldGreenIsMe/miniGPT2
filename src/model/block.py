from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.config import GPTConfig
from src.model.attention import CausalSelfAttention, KVCache
from src.model.mlp import MLP


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
        return_attn: bool = False,
    ):
        """
        x: [B, T, C]
    
        return:
            x: [B, T, C]
            present_kv: None or (k_all, v_all)
            att_map: None or [B, n_head, T, T_key]
        """
        attn_out, present_kv, att_map = self.attn(
            self.ln_1(x),
            past_kv=past_kv,
            use_cache=use_cache,
            return_attn=return_attn,
        )
    
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
    
        return x, present_kv, att_map