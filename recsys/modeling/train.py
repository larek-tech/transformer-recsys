import ast
import datetime as dt
from pathlib import Path

import pandas as pd
import torch
import typer
import yaml
from loguru import logger
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from recsys import config
from recsys.models import transformer as tf
from recsys.models import transformer_dataset as dataset
from transformers import get_linear_schedule_with_warmup


app = typer.Typer()
device = config.device


class ModelConfig:
    """Config for transformer training."""

    def __init__(self, path: Path | str) -> None:
        with open(path, "r") as cfg_data:
            cfg = yaml.safe_load(cfg_data)

        self.lr = float(cfg["lr"])
        self.epochs = int(cfg["epochs"])
        self.batch_size = int(cfg["batch_size"])
        self.seq_max_len = int(cfg["seq_max_len"])
        self.hidden_size = int(cfg["hidden_size"])
        self.num_heads = int(cfg["num_heads"])
        self.num_layers = int(cfg["num_layers"])
        self.num_embeds = int(cfg["num_embeds"])


def _get_unique_articles(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> dict[int, int]:
    unique_articles_train = set()
    unique_articles_test = set()

    for articles in df_train["articles"]:
        if isinstance(articles, str):
            articles = ast.literal_eval(articles)
        unique_articles_train.update(articles)

    for articles in df_test["articles"]:
        if isinstance(articles, str):
            articles = ast.literal_eval(articles)
        unique_articles_test.update(articles)

    unique_articles = unique_articles_train.union(unique_articles_test)

    article_id_map = {
        article_id: idx for idx, article_id in enumerate(unique_articles, start=3)
    }

    article_id_map[0] = 0
    article_id_map[1] = 1
    article_id_map[2] = 2

    return article_id_map


def _prepare_data(cfg: ModelConfig) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
    """Load datasets and create dataloaders and embeddings for training the transformer."""
    try:
        # Load train/test dataframes
        df_train = pd.read_csv(config.PROCESSED_DATA_DIR / "x_train_ids.csv")
        df_test = pd.read_csv(config.PROCESSED_DATA_DIR / "x_test_ids.csv")
    except Exception as e:
        logger.error(f"unable to load csv {e}")
        logger.error(
            "Please run the preprocessing script first:\n python3 -m recsys.dataset main --split"
        )
    # Get uniques articles map <id, count>
    article_id_map = _get_unique_articles(df_train, df_test)

    # Load embeds
    embedings_df: pd.DataFrame = pd.read_pickle(
        config.PROCESSED_DATA_DIR / "articles.pkl"
    )
    embedings_df["article_id"] = embedings_df["article_id"].map(article_id_map)

    # Create training and testing datasets
    df_train = pd.read_csv(config.PROCESSED_DATA_DIR / "x_train_ids.csv")
    df_test = pd.read_csv(config.PROCESSED_DATA_DIR / "x_test_ids.csv")

    # Create data loaders for the training and testing datasets
    dataset_train = dataset.CustomerDataset(df_train, cfg.seq_max_len, article_id_map)
    dataset_test = dataset.CustomerDataset(df_test, cfg.seq_max_len, article_id_map)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        # num_workers=8,
        drop_last=True,
    )
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=cfg.batch_size,
        shuffle=True,  # num_workers=8
    )
    return data_loader_train, data_loader_test, embedings_df


def train_transformer(cfg: ModelConfig) -> tuple[list[float], list[float]]:
    data_loader_train, _, embeds = _prepare_data(cfg)

    # Создание модели
    tf_generator = tf.Transformer(
        embedings_df=embeds,
        num_emb=cfg.num_embeds,
        num_layers=cfg.num_layers,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
    ).to(tf.device)

    # Инициализация оптимизатора с вышеуказанными параметрами
    optimizer = optim.Adam(tf_generator.parameters(), lr=cfg.lr)
    t_total = len(data_loader_train) * cfg.epochs
    warmup_steps = int(0.1 * t_total)  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Масштабировщик для обучения с смешанной точностью
    scaler = torch.cuda.amp.GradScaler()

    # Определение функции потерь
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    num_model_params = 0
    for param in tf_generator.parameters():
        num_model_params += param.flatten().shape[0]

    logger.info(
        f"This Model Has {num_model_params} (Approximately {num_model_params//1e6} Million) Parameters!"
    )

    # Инициализация логгера потерь при обучении и логгера энтропии
    training_loss_logger = []
    entropy_logger = []

    for _ in trange(0, cfg.epochs, leave=False, desc="Эпоха"):
        tf_generator.train()

        for index, text in enumerate(tqdm(data_loader_train)):
            # Преобразование текста в токенизированный ввод

            input_text, output_text = text
            input_text = input_text.to(tf.device)
            output_text = output_text.to(tf.device)
            # Генерация предсказаний
            with torch.cuda.amp.autocast():
                pred = tf_generator(input_text)

            # Вычисление потерь с маскированной кросс-энтропией
            mask = (output_text != 0).float()
            loss = (
                loss_fn(pred.transpose(1, 2), output_text) * mask
            ).sum() / mask.sum()

            # Обратное распространение ошибки
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if index % 20 == 0:
                logger.info(f"train_loss {loss} на {index}/{len(data_loader_train)}")
            # Логирование потерь при обучении и энтропии
            training_loss_logger.append(loss.item())
            with torch.no_grad():
                dist = Categorical(logits=pred)
                entropy_logger.append(dist.entropy().mean().item())
        scheduler.step()

    file_path = config.MODELS_DIR / f"model_{cfg.epochs}_epoch.pth"
    logger.success(f"Training complete. saving model into {file_path}")

    def save_model(model, file_path):
        torch.save(model.state_dict(), file_path)
        logger.info(f"Model has been saved into {file_path}")

    save_model(tf_generator, file_path)

    return training_loss_logger, entropy_logger


def save_report(training_loss_logger, entropy_logger):
    report = pd.DataFrame(
        {
            "training_loss": training_loss_logger,
            "entropy": entropy_logger,
        }
    )
    report_path = (
        config.REPORTS_DIR
        / f"training_report_{dt.datetime.now().strftime('%d_%m_%y_%H_%M')}.csv"
    )

    report.to_csv(report_path, index=False)
    logger.success(f"Report has been saved into {report_path}")


@app.command()
def main(path: str = "model_config.yaml", plot: bool = False):
    cfg = ModelConfig(path)
    training_loss, entropy = train_transformer(cfg)
    if plot:
        save_report(training_loss, entropy)


if __name__ == "__main__":
    app()
