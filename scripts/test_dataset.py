import os
import sys

sys.path.append(os.path.abspath("."))

from src.config import DataConfig, GPTConfig, TrainerConfig
from src.dataset import CharDataset


def main():
    data_cfg = DataConfig()
    gpt_cfg = GPTConfig(vocab_size=0, block_size=16)
    trainer_cfg = TrainerConfig(batch_size=4, device="cpu")

    with open(data_cfg.data_path, "r", encoding=data_cfg.encoding) as f:
        text = f.read()

    dataset = CharDataset(
        text=text,
        block_size=gpt_cfg.block_size,
        train_ratio=data_cfg.train_ratio,
        device=trainer_cfg.device,
    )

    print("===== Dataset Info =====")
    print(f"Vocab size   : {dataset.info.vocab_size}")
    print(f"Total tokens : {dataset.info.total_tokens}")
    print(f"Train tokens : {dataset.info.train_tokens}")
    print(f"Val tokens   : {dataset.info.val_tokens}")

    sample_text = "To be"
    encoded = dataset.encode(sample_text)
    decoded = dataset.decode(encoded)

    print("\n===== Encode / Decode Test =====")
    print(f"Original : {sample_text}")
    print(f"Encoded  : {encoded}")
    print(f"Decoded  : {decoded}")

    x, y = dataset.get_batch(split="train", batch_size=trainer_cfg.batch_size)

    print("\n===== Batch Test =====")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")

    print("\nFirst sample in batch:")
    print("x[0]:", x[0].tolist())
    print("y[0]:", y[0].tolist())

    print("\nDecoded first sample:")
    print("x[0] text:", dataset.decode(x[0].tolist()))
    print("y[0] text:", dataset.decode(y[0].tolist()))


if __name__ == "__main__":
    main()