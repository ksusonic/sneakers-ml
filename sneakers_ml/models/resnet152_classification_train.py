import csv
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet152_Weights, resnet152
from tqdm import tqdm

from sneakers_ml.models.onnx_utils import save_torch_model


class ResNet152Classifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        weights = ResNet152_Weights.DEFAULT
        backbone = resnet152(weights=weights)
        num_filters = backbone.fc.in_features
        backbone.fc = nn.Linear(num_filters, self.num_classes)
        extractor_layers = list(backbone.children())[:-3]
        trainable_bottleneck_layers = list(backbone.children())[-3:-1]
        classifier_layer = list(backbone.children())[-1]
        self.feature_extractor = nn.Sequential(*extractor_layers)
        self.feature_extractor.eval()

        self.trainable_bottleneck = nn.Sequential(*trainable_bottleneck_layers)
        self.classifier = nn.Sequential(classifier_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = self.trainable_bottleneck(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def calculate_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int
) -> tuple[float, float, float, float]:
    f1_macro = multiclass_f1_score(y_pred, y_true, num_classes=num_classes, average="macro")
    f1_micro = multiclass_f1_score(y_pred, y_true, num_classes=num_classes, average="micro")
    f1_weighted = multiclass_f1_score(y_pred, y_true, num_classes=num_classes, average="weighted")
    accuracy = multiclass_accuracy(y_pred, y_true, num_classes=num_classes, average="micro")
    return f1_macro.item(), f1_micro.item(), f1_weighted.item(), accuracy.item()


def train_epoch(
    model: nn.Module, train_dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: str
) -> float:
    running_loss = 0.0

    model.trainable_bottleneck.train()
    model.classifier.train()

    for data in tqdm(train_dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_dataloader)


def eval_epoch(
    model: nn.Module, val_dataloader: DataLoader, criterion: nn.Module, device: str
) -> tuple[float, float, float, float, float]:
    running_loss = 0.0
    y_true = []
    y_pred = []

    model.trainable_bottleneck.eval()
    model.classifier.eval()

    with torch.inference_mode():
        for data in tqdm(val_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            y_true.append(labels.cpu())
            y_pred.append(predicted.cpu())

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        f1_macro, f1_micro, f1_weighted, accuracy = calculate_metrics(y_pred, y_true, model.num_classes)

        return running_loss / len(val_dataloader), f1_macro, f1_micro, f1_weighted, accuracy


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    val_dataloader: DataLoader,
    num_epochs: int,
    device: str,
    log_wandb: bool,
) -> None:
    for _ in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss, f1_macro, f1_micro, f1_weighted, accuracy = eval_epoch(model, val_dataloader, criterion, device)
        if log_wandb:
            wandb.log(
                {
                    "val_f1_macro": f1_macro,
                    "val_f1_micro": f1_micro,
                    "val_f1_weighted": f1_weighted,
                    "val_accuracy": accuracy,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                }
            )


@hydra.main(version_base=None, config_path="../../config", config_name="cfg_dl")
def train_resnet152_and_save(cfg: DictConfig) -> None:

    weights = ResNet152_Weights.DEFAULT
    preprocess = weights.transforms()
    torch.set_float32_matmul_precision("medium")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = ImageFolder(cfg.data.splits.train, transform=preprocess)
    val_dataset = ImageFolder(cfg.data.splits.val, transform=preprocess)
    test_dataset = ImageFolder(cfg.data.splits.test, transform=preprocess)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.models.resnet152.dataloader.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.models.resnet152.dataloader.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.models.resnet152.dataloader.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.models.resnet152.dataloader.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.models.resnet152.dataloader.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.models.resnet152.dataloader.num_workers,
    )

    save_path = Path(cfg.models.resnet152.idx_to_classes)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    class_to_idx = train_dataset.class_to_idx
    with save_path.open("wb") as save_file:
        np.save(save_file, np.array(list(class_to_idx.items())), allow_pickle=False)

    num_classes = len(train_dataset.classes)
    model = ResNet152Classifier(num_classes)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [{"params": model.trainable_bottleneck.parameters()}, {"params": model.classifier.parameters()}],
        lr=cfg.models.resnet152.optimizer.lr,
    )

    if cfg.log_wandb:
        wandb.init(project="sneakers_ml")

    train(
        model,
        train_dataloader,
        criterion,
        optimizer,
        val_dataloader,
        cfg.models.resnet152.num_epochs,
        device,
        cfg.log_wandb,
    )

    if cfg.log_wandb:
        wandb.finish()

    loss, f1_macro, f1_micro, f1_weighted, accuracy = eval_epoch(model, test_dataloader, criterion, device)
    print(
        {
            "test_f1_macro": f1_macro,
            "test_f1_micro": f1_micro,
            "test_f1_weighted": f1_weighted,
            "test_accuracy": accuracy,
            "test_loss": loss,
        }
    )

    if cfg.models.resnet152.save_onnx:
        model.eval()
        model.to("cpu")
        torch_input = torch.randn(1, 3, 224, 224)
        path = f"{cfg.paths.models_save}/{cfg.models.resnet152.name}.onnx"
        save_torch_model(model, torch_input, path)

        metrics = [f1_macro, f1_micro, f1_weighted, accuracy]
        for i in range(len(metrics)):
            metrics[i] = round(metrics[i], 2)
        results_save_path = Path(cfg.paths.results)
        with results_save_path.open("a", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([cfg.models.resnet152.name, *metrics])


if __name__ == "__main__":
    train_resnet152_and_save()  # pylint: disable=no-value-for-parameter
