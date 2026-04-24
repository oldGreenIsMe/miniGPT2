from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    模型结构配置
    """
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True


@dataclass
class TrainerConfig:
    """
    训练流程配置
    """
    batch_size: int = 32
    max_iters: int = 5000
    eval_interval: int = 200
    eval_iters: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    device: str = "cuda"
    log_interval: int = 50
    ckpt_path: str = "checkpoints/latest_model.pt"


@dataclass
class DataConfig:
    """
    数据相关配置
    """
    data_path: str = "data/input.txt"
    train_ratio: float = 0.9
    encoding: str = "utf-8"
    seed: int = 42