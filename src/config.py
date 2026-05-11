from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True


@dataclass
class TrainerConfig:
    batch_size: int = 32
    max_iters: int = 2000
    eval_interval: int = 100
    eval_iters: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    device: str = "cuda"
    log_interval: int = 20

    latest_ckpt_path: str = "checkpoints/latest_model.pt"
    best_ckpt_path: str = "checkpoints/best_model.pt"
    meta_path: str = "checkpoints/meta.pkl"
    log_path: str = "logs/train_log.csv"


@dataclass
class DataConfig:
    data_path: str = "data/input.txt"
    train_ratio: float = 0.9
    encoding: str = "utf-8"
    seed: int = 42