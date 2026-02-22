"""
Dataset/dataloader scaffold for Titans LM.
Tokenization is pluggable; here we assume a text file with one sample per line.
"""

from pathlib import Path
from typing import Callable, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class TextWindowDataset(Dataset):
    """
    Memory-safe dataset that tokenizes the corpus once and slices fixed windows
    without duplicating data. Keeps the token buffer as a torch tensor.
    """

    def __init__(
        self,
        path: Path,
        tokenizer: Callable[[str], List[int]],
        seq_len: int,
        max_tokens: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.tokens = self._load_tokens(path, tokenizer, max_tokens)
        if len(self.tokens) <= seq_len:
            raise ValueError(f"Not enough tokens ({len(self.tokens)}) for seq_len={seq_len}")
        self.num_tokens = len(self.tokens)
        self.num_windows = (len(self.tokens) - 1) // seq_len

    def _load_tokens(
        self, path: Path, tokenizer: Callable[[str], List[int]], max_tokens: Optional[int]
    ) -> torch.Tensor:
        tokens: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens.extend(tokenizer(line))
                if max_tokens is not None and len(tokens) >= max_tokens:
                    tokens = tokens[:max_tokens]
                    break
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


def build_dataloader(
    path: str,
    tokenizer: Callable[[str], List[int]],
    seq_len: int,
    batch_size: int,
    shuffle_buffer: int = 100000,  # kept for API compatibility; unused
    num_workers: int = 0,
    shuffle: bool = True,
    max_tokens: Optional[int] = None,
) -> DataLoader:
    ds = TextWindowDataset(Path(path), tokenizer, seq_len, max_tokens=max_tokens)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)

