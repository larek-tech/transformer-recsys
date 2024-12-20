import ast

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class CustomerDataset(Dataset):
    def __init__(self, df, max_len, article_id_map):
        self.df = df
        self.max_len = max_len
        self.article_id_map = article_id_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        articles = self.df.iloc[idx]["articles"]
        if isinstance(articles, str):
            articles = ast.literal_eval(
                articles
            )
        articles = [
            self.article_id_map[article] for article in articles
        ]
        # <sos> - 1
        # <eos> - 2
        # <pad> - 0
        articles = [1]  + articles + [2] + [0] * (self.max_len - len(articles))  # Padding
        input_seq = torch.tensor(articles[:-1], dtype=torch.long)
        target_seq = torch.tensor(articles[1:], dtype=torch.long) # 1:
        return input_seq, target_seq