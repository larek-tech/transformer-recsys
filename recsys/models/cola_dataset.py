import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, pad_list):
        self.texts = texts
        self.tokenizer = tokenizer
        self.pad_list = pad_list

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text_t = self.tokenizer(text)
        text_t = torch.tensor(self.pad_list(text_t), dtype=torch.long)
        return text_t
