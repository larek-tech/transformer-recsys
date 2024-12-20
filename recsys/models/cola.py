import torch
from torch import nn
import math
from transformers import BertTokenizer
from pytorch_lightning import LightningModule

TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

CLS_IDX = TOKENIZER.convert_tokens_to_ids("[CLS]")
PAD_IDX = TOKENIZER.convert_tokens_to_ids("[PAD]")
SEP_IDX = TOKENIZER.convert_tokens_to_ids("[SEP]")


vocab_size = TOKENIZER.vocab_size


def tokenize(text: str):
    raw_tokens = TOKENIZER.encode(text)
    return raw_tokens


def pad_list(
    list_integers, context_size: int = 90, pad_val: int = PAD_IDX, mode="right"
):
    list_integers = list_integers[:context_size]

    if len(list_integers) < context_size:
        if mode == "left":
            list_integers = [pad_val] * (
                context_size - len(list_integers)
            ) + list_integers
        else:
            list_integers = list_integers + [pad_val] * (
                context_size - len(list_integers)
            )

    return list_integers


class PositionalEncoding(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0:, :, 0::2] = torch.sin(position * div_term)
        pe[0:, :, 1::2] = torch.cos(position * div_term)
        # позиционное кодирование
        self.register_buffer("pe", pe)

        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        x = x + self.pe[:, : x.size(1)] / math.sqrt(self.d_model)

        return self.dropout(x)


class Cola(LightningModule):
    def __init__(
        self,
        lr=0.001,
        use_pretrained=False,
        dropout=0.2,
        d_model=128,
        n_vocab=30_522,
        smoothing=0.1,
    ):
        super().__init__()
        self.dropout = dropout

        self.lr = lr
        self.d_model = d_model
        self.n_vocab = n_vocab
        self.smoothing = smoothing

        # Text embeddings and encoder
        self.item_embeddings = torch.nn.Embedding(self.n_vocab, self.d_model)
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model, dropout=self.dropout
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dropout=self.dropout, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Output layer to project to vocabulary size
        self.output_layer = torch.nn.Linear(self.d_model, self.n_vocab)

        self.save_hyperparameters()

    def encode_text(self, x):
        x = self.item_embeddings(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.output_layer(x)  # Add projection to vocab size

        return x  # Return full sequence output for language modeling

    def forward(self, x):
        x = self.item_embeddings(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.output_layer(x)  # Project to vocab size
        return x
