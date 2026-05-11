import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


class CharTokenizer:
    """
    字符级 tokenizer。

    支持两种初始化方式：
    1. 从 text 自动构建 stoi / itos
    2. 从已有 stoi / itos 恢复 tokenizer
    """
    def __init__(
        self,
        text: str | None = None,
        stoi: Dict[str, int] | None = None,
        itos: Dict[int, str] | None = None,
    ):
        if text is not None:
            chars = sorted(list(set(text)))
            self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
            self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}

        elif stoi is not None and itos is not None:
            self.stoi = stoi
            self.itos = itos

        else:
            raise ValueError("Either text or both stoi/itos must be provided.")

        self.vocab_size: int = len(self.stoi)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join([self.itos[i] for i in ids])

    def save(self, path: str):
        meta = {
            "stoi": self.stoi,
            "itos": self.itos,
            "vocab_size": self.vocab_size,
        }

        with open(path, "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            meta = pickle.load(f)

        return cls(
            stoi=meta["stoi"],
            itos=meta["itos"],
        )


@dataclass
class DatasetInfo:
    vocab_size: int
    total_tokens: int
    train_tokens: int
    val_tokens: int


class CharDataset:
    def __init__(
        self,
        text: str,
        block_size: int,
        train_ratio: float = 0.9,
        device: str = "cpu",
        tokenizer: CharTokenizer | None = None,
    ):
        self.block_size = block_size
        self.device = device

        if tokenizer is None:
            self.tokenizer = CharTokenizer(text=text)
        else:
            self.tokenizer = tokenizer

        token_ids = self.tokenizer.encode(text)
        self.data = torch.tensor(token_ids, dtype=torch.long)

        n = int(train_ratio * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

        self.info = DatasetInfo(
            vocab_size=self.tokenizer.vocab_size,
            total_tokens=len(self.data),
            train_tokens=len(self.train_data),
            val_tokens=len(self.val_data),
        )

    def encode(self, s: str) -> List[int]:
        return self.tokenizer.encode(s)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_block_size(self) -> int:
        return self.block_size

    def save_tokenizer(self, path: str):
        self.tokenizer.save(path)

    @staticmethod
    def load_tokenizer(path: str) -> CharTokenizer:
        return CharTokenizer.load(path)

    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data

        ix = torch.randint(
            low=0,
            high=len(data) - self.block_size - 1,
            size=(batch_size,),
        )

        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])

        x = x.to(self.device)
        y = y.to(self.device)

        return x, y