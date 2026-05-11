from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import GPTConfig
from src.model.attention import KVCache
from src.model.block import Block


PastKeyValues = List[KVCache]


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.n_embd,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=config.block_size,
            embedding_dim=config.n_embd,
        )

        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(
            in_features=config.n_embd,
            out_features=config.vocab_size,
            bias=False,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[PastKeyValues] = None,
        use_cache: bool = False,
        return_attn: bool = False,
    ):
        """
        idx:
            no cache  : [B, T]
            with cache: [B, T_new]

        targets:
            [B, T] or None

        past_key_values:
            None or list of length n_layer
            each item: (past_k, past_v)

        return:
            logits: [B, T, vocab_size]
            loss: scalar or None
            present_key_values: None or list of length n_layer
        """
        B, T = idx.shape

        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            B, T = idx.shape

        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(2)

        # rolling cache 下，past_length 可能已经等于 block_size。
        # 当前输入 T 个 token 时，position 起点应该被压到合法范围内。
        pos_start = min(past_length, self.config.block_size - T)

        tok_emb = self.token_embedding(idx)

        pos = torch.arange(
            pos_start,
            pos_start + T,
            dtype=torch.long,
            device=idx.device,
        )
        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb
        x = self.drop(x)

        present_key_values = [] if use_cache else None
        all_attn_maps = [] if return_attn else None

        for layer_idx, block in enumerate(self.blocks):
            past_kv = None
            if past_key_values is not None:
                past_kv = past_key_values[layer_idx]

            x, present_kv, att_map = block(
                x,
                past_kv=past_kv,
                use_cache=use_cache,
                return_attn=return_attn,
            )

            if return_attn:
                all_attn_maps.append(att_map)

            if use_cache:
                present_key_values.append(present_kv)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss, present_key_values, all_attn_maps

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        原始无 cache 生成。
        idx: [B, T]
        """
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]

            logits, _, _, _ = self(idx_cond)

            logits = logits[:, -1, :]

            logits = self._sample_logits(
                logits=logits,
                temperature=temperature,
                top_k=top_k,
            )

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_with_cache(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Rolling KV cache 版本生成。
    
        idx: [B, T_prompt]
    
        return:
            [B, T_prompt + max_new_tokens]
        """
        self.eval()
    
        # prefill 阶段最多只使用最后 block_size 个 token 建立 cache
        idx_cond = idx[:, -self.config.block_size:]
    
        logits, _, past_key_values, _ = self(
            idx_cond,
            past_key_values=None,
            use_cache=True,
        )
    
        for _ in range(max_new_tokens):
            logits_next = logits[:, -1, :]
    
            logits_next = self._sample_logits(
                logits=logits_next,
                temperature=temperature,
                top_k=top_k,
            )
    
            probs = F.softmax(logits_next, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
    
            idx = torch.cat((idx, idx_next), dim=1)
    
            logits, _, past_key_values, _ = self(
                idx_next,
                past_key_values=past_key_values,
                use_cache=True,
            )
    
        return idx
    
    def _sample_logits(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        logits: [B, vocab_size]
        """
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = v[:, [-1]]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        return logits