import ast

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class CustomerDataset(Dataset):
    """Custom dataset class."""

    def __init__(
        self, df: pd.DataFrame, max_len: int, article_id_map: dict[int, int]
    ) -> None:
        self.df = df
        self.max_len = max_len
        self.article_id_map = article_id_map

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        article_ids = self.df.iloc[idx]["articles"]
        if isinstance(article_ids, str):
            article_ids = ast.literal_eval(article_ids)

        articles = [self.article_id_map[article] for article in article_ids]
        # <sos> - 1
        # <eos> - 2
        # <pad> - 0
        articles = (
            [1] + articles + [2] + [0] * (self.max_len - len(articles))
        )  # Padding
        input_seq = torch.tensor(articles[:-1], dtype=torch.long)
        target_seq = torch.tensor(articles[1:], dtype=torch.long)  # 1:
        return input_seq, target_seq
