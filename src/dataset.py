import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple


class CharTokenizer:
    """
    最简单的字符级 tokenizer
    """
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size: int = len(chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]
    
    def decode(self, ids: List[int]) -> str:
        return "".join([self.itos[i] for i in ids])
    

@dataclass
class DatasetInfo:
    """
    方便调试和查看数据集状态
    """
    vocab_size: int
    total_tokens: int
    train_tokens: int
    val_tokens: int


class CharDataset:
    """
    负责：
    1. 读取原始文本
    2. 建立字符级 tokenizer
    3. encode 成 token ids
    4. 划分 train / val
    5. 采样 batch
    """
    def __init__(
            self,
            text: str,
            block_size: int,
            train_ratio: float = 0.9,
            device: str = "cpu",
    ):
        self.block_size = block_size
        self.device = device

        # 1) tokenizer
        self.tokenizer = CharTokenizer(text)

        # 2) 整个文本编码成 token id 序列
        token_ids = self.tokenizer.encode(text)
        self.data = torch.tensor(token_ids, dtype=torch.long)

        # 3) 划分 train / val
        n = int(train_ratio * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

        self.info = DatasetInfo(
            vocab_size=self.tokenizer.vocab_size,
            total_tokens=len(self.data),
            train_tokens=len(self.train_data),
            val_tokens=len(self.val_data)
        )

    def encode(self, s: str) -> List[int]:
        return self.tokenizer.encode(s)
    
    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)
    
    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size
    
    def get_block_size(self) -> int:
        return self.block_size
    
    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
            x: [B, T]
            y: [B, T]

        其中：
            x = token[t : t+block_size]
            y = token[t+1 : t+block_size+1]

        也就是语言模型标准的 next-token prediction
        """
        data = self.train_data if split == "train" else self.val_data

        # 随机选择 batch_size 个起点
        ix = torch.randint(
            low=0,
            high=len(data) - self.block_size - 1,
            size=(batch_size,)
        )

        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])

        x = x.to(self.device)
        y = y.to(self.device)
        return x, y