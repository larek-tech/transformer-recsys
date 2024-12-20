from pathlib import Path

import typer
import pandas as pd
from loguru import logger

from recsys.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def filter_transactions(
    transaction_path: Path | str,
    customer_path: Path | str,
    processed_transactions_path: Path | str = PROCESSED_DATA_DIR / "customers.csv",
    processed_customers_path: Path | str = PROCESSED_DATA_DIR / "transactions.csv",
    top_articles_cnt: int = 1000,
) -> None:
    # 1 filter tx to limit amount of data
    try:
        transactions: pd.DataFrame = pd.read_csv(filepath_or_buffer=transaction_path)
        customers: pd.DataFrame = pd.read_csv(filepath_or_buffer=customer_path)
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


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_customers_path: Path = RAW_DATA_DIR / "customers.csv",
    input_transactions_path: Path = RAW_DATA_DIR / "transactions_train.csv",
    output_customers_path: Path = PROCESSED_DATA_DIR / "customers.csv",
    output_transactions_path: Path = PROCESSED_DATA_DIR / "transactions_train.csv",
    top_k_articles: int = 1000,
    # ----------------------------------------------
) -> None:
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    filter_transactions(
        transaction_path=input_transactions_path,
        customer_path=input_customers_path,
        processed_transactions_path=output_transactions_path,
        processed_customers_path=output_customers_path,
        top_articles_cnt=top_k_articles,
    )
    logger.success("Processing dataset complete.")
    # -----------------------------------------


def process_articles(
    input_articles_path: Path = RAW_DATA_DIR / "articles.csv",
    output_articles_path: Path = PROCESSED_DATA_DIR / "articles.csv",
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

    df.to_csv(path_or_buf=output_articles_path)


@app.command("embeddings")
def embeddings(
    input_articles_path: Path = RAW_DATA_DIR / "articles.csv",
    output_articles_path: Path = PROCESSED_DATA_DIR / "articles.csv",
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
