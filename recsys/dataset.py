from pathlib import Path

import typer
import pandas as pd
from loguru import logger
from tqdm import tqdm

from recsys.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def filter_transactions(
    transaction_path: Path | str,
    customer_path: Path | str,
    processed_transactions_path: Path | str = PROCESSED_DATA_DIR / "customers.csv",
    processed_customers_path: Path | str = PROCESSED_DATA_DIR / "transactions.csv",
    top_articles_cnt: int = 1000,
):
    # 1 filter tx to limit amount of data
    try:
        transactions = pd.read_csv(transaction_path)
        customers = pd.read_csv(customer_path)
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
    customers.to_csv(processed_transactions_path)

    logger.info(f"saving customers into {processed_customers_path}")
    customers.to_csv(processed_customers_path)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_customers_path: Path = RAW_DATA_DIR / "customers.csv",
    input_transactions_path: Path = RAW_DATA_DIR / "transactions_train.csv",
    output_customers_path: Path = PROCESSED_DATA_DIR / "customers.csv",
    output_transactions_path: Path = PROCESSED_DATA_DIR / "transactions_train.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    filter_transactions(
        input_transactions_path,
        input_customers_path,
        output_transactions_path,
        output_customers_path,
    )
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
