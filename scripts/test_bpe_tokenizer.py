import os
import sys

sys.path.append(os.path.abspath("."))

from src.config import DataConfig
from src.bpe_tokenizer import SimpleBPETokenizer


def main():
    data_cfg = DataConfig()

    with open(data_cfg.data_path, "r", encoding=data_cfg.encoding) as f:
        text = f.read()

    target_vocab_size = 100

    tokenizer = SimpleBPETokenizer(vocab_size=target_vocab_size)
    tokenizer.train(text)

    sample = "To be or not to be"

    char_tokens = list(sample)
    bpe_tokens = tokenizer.encode_to_tokens(sample)
    bpe_ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(bpe_ids)

    print("=" * 60)
    print("Simple BPE Tokenizer Test")
    print("=" * 60)
    print(f"Target vocab size : {target_vocab_size}")
    print(f"Actual vocab size : {tokenizer.vocab_size}")
    print(f"Num merges        : {len(tokenizer.merges)}")
    print("=" * 60)

    print("\nSample text:")
    print(sample)

    print("\nChar-level tokens:")
    print(char_tokens)
    print(f"Char-level token count: {len(char_tokens)}")

    print("\nBPE tokens:")
    print(bpe_tokens)
    print(f"BPE token count: {len(bpe_tokens)}")

    print("\nBPE ids:")
    print(bpe_ids)

    print("\nDecoded:")
    print(decoded)

    print("\nFirst 20 merges:")
    for i, pair in enumerate(tokenizer.merges[:20]):
        print(f"{i:02d}: {repr(pair[0])} + {repr(pair[1])} -> {repr(pair[0] + pair[1])}")

    save_path = "checkpoints/simple_bpe.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)

    loaded = SimpleBPETokenizer.load(save_path)
    loaded_decoded = loaded.decode(loaded.encode(sample))

    print("\nSave / Load Test:")
    print(f"Saved to       : {save_path}")
    print(f"Loaded decoded : {loaded_decoded}")
    print("=" * 60)


if __name__ == "__main__":
    main()