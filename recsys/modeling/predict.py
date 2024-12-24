from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import torch 
from torch.distributions import Categorical
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from recsys.config import MODELS_DIR, PROCESSED_DATA_DIR
from recsys.models.transformer_dataset import CustomerDataset
from recsys.modeling.train import ModelConfig, _get_unique_articles, _prepare_data
from recsys.models import transformer as tf

app = typer.Typer()

loss_fn = nn.CrossEntropyLoss(reduction="none")

def evaluate_model(tf_generator, data_loader_test, device):
    tf_generator.eval()
    total_loss = 0
    total_entropy = 0
    num_samples = 0
    correct_predictions_at_1 = 0
    correct_predictions_at_5 = 0
    correct_predictions_at_10 = 0
    total_positive = 0

    with torch.no_grad():
        for index, text in enumerate(tqdm(data_loader_test)):
            input_seq, target_seq = text
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            pred = tf_generator(input_seq)
            mask = (target_seq != 0).float()
            loss = (loss_fn(pred.transpose(1, 2), target_seq) * mask).sum() / mask.sum()
            total_loss += loss.item()

            dist = Categorical(logits=pred)
            total_entropy += dist.entropy().mean().item()

            # Получаем предсказания
            pred_labels = pred.argmax(dim=-1)  # Предсказание с наивысшей вероятностью
            top5_labels = pred.topk(5, dim=-1).indices  # 5 наивысших предсказаний
            top10_labels = pred.topk(10, dim=-1).indices  # 10 наивысших предсказаний

            correct_predictions_at_1 += (pred_labels == target_seq).sum().item()
            target_seq_expanded = target_seq.unsqueeze(2)

            correct_predictions_at_5 += ((top5_labels == target_seq_expanded).any(dim=1)).sum().item()
            correct_predictions_at_10 += ((top10_labels == target_seq_expanded).any(dim=1)).sum().item()

            total_positive += (target_seq != 0).sum().item()  # Считаем количество положительных примеров

            num_samples += 1

    avg_loss = total_loss / num_samples
    avg_entropy = total_entropy / num_samples

    # Рассчитываем recall@1, recall@5 и recall@10
    recall_at_1 = correct_predictions_at_1 / total_positive if total_positive > 0 else 0
    recall_at_5 = correct_predictions_at_5 / total_positive if total_positive > 0 else 0
    recall_at_10 = correct_predictions_at_10 / total_positive if total_positive > 0 else 0

    logger.success(f"Test Loss: {avg_loss:.4f}, Test Entropy: {avg_entropy:.4f}, "
          f"Recall@1: {recall_at_1:.4f}, Recall@5: {recall_at_5:.4f}, Recall@10: {recall_at_10:.4f}")



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    df_test_path: Path  = PROCESSED_DATA_DIR / "x_test_ids.csv",
    df_train_path: Path = PROCESSED_DATA_DIR / "x_train_ids.csv",
    model_path: Path = MODELS_DIR / "model_160_epoch.pth",
    cfg_path: Path = "model_config.yaml",
    # -----------------------------------------
):
    cfg = ModelConfig(cfg_path)
    data_loader_train, _, embeds = _prepare_data(cfg)

    model = tf.Transformer(embedings_df=embeds, num_emb=cfg.num_embeds, num_layers=cfg.num_layers, 
                           hidden_size=cfg.hidden_size, num_heads=cfg.num_heads).to(tf.device)
    model.load_state_dict(torch.load(model_path))

    df_test = pd.read_csv(df_test_path)    
    df_train = pd.read_csv(df_train_path)    
    article_id_map = _get_unique_articles(df_test, df_train)


    dataset_test = CustomerDataset(df_test, cfg.seq_max_len , article_id_map)
    data_loader_test = DataLoader(dataset_test, batch_size=cfg.batch_size, shuffle=True, num_workers=8)

    evaluate_model(model, data_loader_test, tf.device)


if __name__ == "__main__":
    app()
