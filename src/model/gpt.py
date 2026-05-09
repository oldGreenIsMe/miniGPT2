import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import GPTConfig
from src.model.block import Block


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
        targets: torch.Tensor | None = None,
    ):
        """
        idx:     [B, T]
        targets: [B, T] or None

        return:
            logits: [B, T, vocab_size]
            loss: scalar or None
        """
        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length T={T} exceeds block_size={self.config.block_size}"
            )

        tok_emb = self.token_embedding(idx)

        pos = torch.arange(
            0,
            T,
            dtype=torch.long,
            device=idx.device,
        )
        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        idx: [B, T]

        return:
            idx: [B, T + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            if temperature <= 0:
                raise ValueError("temperature must be greater than 0")

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = v[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx