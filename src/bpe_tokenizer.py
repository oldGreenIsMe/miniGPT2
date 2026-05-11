import pickle
from collections import Counter
from typing import Dict, List, Tuple


class SimpleBPETokenizer:
    """
    教学版 BPE tokenizer。

    注意：
    1. 这是 character-level BPE，不是 GPT-2 byte-level BPE。
    2. 目标是理解 BPE merge 机制。
    3. 适合当前 Mini GPT-2 项目做 tokenizer 实验。
    """

    def __init__(
        self,
        vocab_size: int = 200,
        merges: List[Tuple[str, str]] | None = None,
        stoi: Dict[str, int] | None = None,
        itos: Dict[int, str] | None = None,
    ):
        self.target_vocab_size = vocab_size
        self.merges = merges or []
        self.stoi = stoi or {}
        self.itos = itos or {}

    def train(self, text: str):
        """
        从训练文本中学习 BPE merge rules。
        """
        tokens = list(text)

        vocab = set(tokens)

        while len(vocab) < self.target_vocab_size:
            pair_counts = self._get_pair_counts(tokens)

            if len(pair_counts) == 0:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]

            if best_count < 2:
                break

            tokens = self._merge_pair(tokens, best_pair)

            merged_token = best_pair[0] + best_pair[1]
            vocab.add(merged_token)
            self.merges.append(best_pair)

        final_vocab = sorted(vocab)
        self.stoi = {token: i for i, token in enumerate(final_vocab)}
        self.itos = {i: token for token, i in self.stoi.items()}

    def _get_pair_counts(self, tokens: List[str]) -> Counter:
        pairs = zip(tokens[:-1], tokens[1:])
        return Counter(pairs)

    def _merge_pair(
        self,
        tokens: List[str],
        pair: Tuple[str, str],
    ) -> List[str]:
        merged = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                merged.append(pair[0] + pair[1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1

        return merged

    def encode_to_tokens(self, text: str) -> List[str]:
        """
        把字符串编码成 BPE token 字符串列表。
        """
        tokens = list(text)

        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        把字符串编码成 token id。
        """
        tokens = self.encode_to_tokens(text)

        ids = []
        for token in tokens:
            if token not in self.stoi:
                raise KeyError(
                    f"Token {repr(token)} not found in BPE vocab. "
                    f"This usually means the input contains unseen characters."
                )
            ids.append(self.stoi[token])

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        把 token id 解码回字符串。
        """
        tokens = [self.itos[i] for i in ids]
        return "".join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def save(self, path: str):
        meta = {
            "target_vocab_size": self.target_vocab_size,
            "merges": self.merges,
            "stoi": self.stoi,
            "itos": self.itos,
        }

        with open(path, "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            meta = pickle.load(f)

        return cls(
            vocab_size=meta["target_vocab_size"],
            merges=meta["merges"],
            stoi=meta["stoi"],
            itos=meta["itos"],
        )