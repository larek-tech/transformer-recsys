from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from loguru import logger
from transformers import AutoModel, AutoTokenizer

from recsys.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, device

app = typer.Typer()


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("bert-base-cased")
model = model.to(device)


def filter_transactions(
    transaction_path: Path | str,
    customer_path: Path | str,
    articles_path: Path | str,
    processed_transactions_path: Path | str = PROCESSED_DATA_DIR / "customers.csv",
    processed_customers_path: Path | str = PROCESSED_DATA_DIR / "transactions.csv",
    processed_articles_path: Path | str = PROCESSED_DATA_DIR / "articles.csv",
    top_articles_cnt: int = 1000,
) -> None:
    # 1 filter tx to limit amount of data
    try:
        transactions: pd.DataFrame = pd.read_csv(filepath_or_buffer=transaction_path)
        customers: pd.DataFrame = pd.read_csv(filepath_or_buffer=customer_path)
        articles = pd.read_csv(filepath_or_buffer=articles_path)
    except Exception as e:
        logger.error(f"unable to load csv {e}")
    # 2 group by customer
    logger.info(f"loaded {len(transactions)} transactions")
    logger.info(f"loaded {len(customers)} customers")
    top_article_ids_counts = transactions["article_id"].value_counts()[
        :top_articles_cnt
    ]
    top_article_ids = top_article_ids_counts.to_dict().keys()
    transactions = transactions[transactions["article_id"].isin(top_article_ids)]
    logger.info(
        f"filtered transactions  to {len(transactions)}  with top {top_articles_cnt}"
    )

    customers = customers[customers["customer_id"].isin(transactions["customer_id"])]
    logger.info(f"filtered customers to {len(customers)}")

    logger.info(f"saving transactions into {processed_transactions_path}")
    transactions.to_csv(processed_transactions_path)

    logger.info(f"saving customers into {processed_customers_path}")
    customers.to_csv(processed_customers_path)

    # Cut articles
    articles = articles[articles["article_id"].isin(top_article_ids)]

    logger.info(f"saving articles into {processed_articles_path}")
    articles.to_csv(processed_articles_path)


def test_train_split(transactions_path: Path | str, split_output_path: Path) -> None:
    df = pd.read_csv(transactions_path)
    test_size = 0.2
    num_samples = df.shape[0]
    num_test_samples = int(num_samples * test_size)
    random_state = 42

    # Перемешивание индексов
    np.random.seed(random_state)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Разделение индексов на обучающую и тестовую выборки
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]

    X_train = df.iloc[train_indices]
    X_test = df.iloc[test_indices]

    X_train.to_csv(split_output_path / "x_train_ids.csv")
    X_test.to_csv(split_output_path / "x_test_ids.csv")
    logger.info("splitted into test and train split")


def group_transactions(
    transactions_path: Path | str,
    grouped_transaction_path: Path | str,
) -> None:
    transactions = pd.read_csv(transactions_path)

    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions = transactions.sort_values(by="t_dat")
    grouped_transactions = (
        transactions.groupby("customer_id")["article_id"].apply(list).reset_index()
    )
    grouped_transactions.columns = ["customer_id", "articles"]

    grouped_transactions["sequence_length"] = grouped_transactions["articles"].apply(
        lambda x: len(x)
    )

    min_sequence_length = 2
    min_sequence_mask = grouped_transactions["sequence_length"] < min_sequence_length

    max_sequence_length = 10
    max_sequence_mask = grouped_transactions["sequence_length"] > max_sequence_length

    grouped_transactions = grouped_transactions[~min_sequence_mask & ~max_sequence_mask]
    # grouped_transactions[max_sequence_mask].shape, grouped_transactions[
    #     max_sequence_mask
    # ].shape[0] / grouped_transactions.shape[0] * 100 # type: ignore
    grouped_transactions.to_csv(grouped_transaction_path, index=False)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_customers_path: Path = RAW_DATA_DIR / "customers.csv",
    input_transactions_path: Path = RAW_DATA_DIR / "transactions_train.csv",
    input_articles_path: Path = RAW_DATA_DIR / "articles.csv",
    output_customers_path: Path = PROCESSED_DATA_DIR / "customers.csv",
    output_transactions_path: Path = PROCESSED_DATA_DIR / "transactions_train.csv",
    output_articles_path: Path = PROCESSED_DATA_DIR / "articles.csv",
    output_grouped_transactions_path: Path = PROCESSED_DATA_DIR
    / "grouped_transactions.csv",
    top_k_articles: int = 1000,
    split: bool = False,
    # ----------------------------------------------
) -> None:
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    filter_transactions(
        transaction_path=input_transactions_path,
        customer_path=input_customers_path,
        articles_path=input_articles_path,
        processed_transactions_path=output_transactions_path,
        processed_customers_path=output_customers_path,
        processed_articles_path=output_articles_path,
        top_articles_cnt=top_k_articles,
    )
    logger.success("Processing dataset complete.")
    # -----------------------------------------
    if split:
        group_transactions(
            output_transactions_path,
            output_grouped_transactions_path,
        )
        logger.success("grouped transactions by users")
        test_train_split(
            output_grouped_transactions_path,
            PROCESSED_DATA_DIR,
        )
        logger.success("train split complete.")


def get_embeddings(text: str):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
    )
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    sentence_embeddings = outputs[0]
    sentence_embeddings = sentence_embeddings.mean(dim=1)
    sentence_embeddings = sentence_embeddings.cpu().numpy()
    return sentence_embeddings[0]


def process_articles(
    input_articles_path: Path = PROCESSED_DATA_DIR / "articles.csv",
    output_articles_path: Path = PROCESSED_DATA_DIR / "articles.pkl",
) -> None:
    try:
        df: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=input_articles_path,
            nrows=None,
            dtype={
                "article_id": str,
            },
        )
    except Exception as e:
        logger.error(f"unable to load csv {e}")

    logger.info(f"loaded {len(df)} articles")
    df["text"] = df.apply(
        lambda x: " ".join(
            [
                str(x["prod_name"]),
                str(x["product_type_name"]),
                str(x["product_group_name"]),
                str(x["graphical_appearance_name"]),
                str(x["colour_group_name"]),
                str(x["perceived_colour_value_name"]),
                str(x["index_name"]),
                str(x["section_name"]),
                str(x["detail_desc"]),
            ]
        ),
        axis=1,
    )
    tokens_df = pd.DataFrame(
        data={
            "article_id": [0, 1, 2],
            "text": ["<pad>", "<sos>", "<eos>"],
        }
    )
    df = pd.concat(
        objs=[df, tokens_df],
        ignore_index=True,
    )
    df["Embed_comb_text"] = df["text"].apply(lambda x: get_embeddings(x))
    df.to_pickle(output_articles_path)


@app.command("embeddings")
def embeddings(
    input_articles_path: Path = PROCESSED_DATA_DIR / "articles.csv",
    output_articles_path: Path = PROCESSED_DATA_DIR / "articles.pkl",
) -> None:
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing embeddings...")
    process_articles(
        input_articles_path=input_articles_path,
        output_articles_path=output_articles_path,
    )
    logger.success("Processing embeddings complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
