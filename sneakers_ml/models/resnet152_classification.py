from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from hydra import compose, initialize
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from scipy.special import softmax
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet152_Weights, resnet152
from tqdm import tqdm

from sneakers_ml.models.base import DLClassifier, log_metrics
from sneakers_ml.models.onnx_utils import get_device, predict, save_torch_model


class CustomResNet152(nn.Module):
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
        return self.classifier(x)  # type: ignore[no-any-return]


class ResNet152ClassificationTrainer:

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = get_device(cfg.models.resnet152.device)

        weights = ResNet152_Weights.DEFAULT
        self.preprocess = weights.transforms()

        self.load_dataset()
        self.save_class_to_idx()
        self.load_model()
        self.set_training_args()

    def get_dataloader(self, dataset: torch.utils.data.Dataset[Any]) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=self.cfg.models.resnet152.dataloader.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.cfg.models.resnet152.dataloader.num_workers,
        )

    def load_dataset(self) -> None:
        self.train_dataset = ImageFolder(self.cfg.data.splits.train, transform=self.preprocess)
        self.val_dataset = ImageFolder(self.cfg.data.splits.val, transform=self.preprocess)
        self.test_dataset = ImageFolder(self.cfg.data.splits.test, transform=self.preprocess)

        self.train_dataloader = self.get_dataloader(self.train_dataset)
        self.val_dataloader = self.get_dataloader(self.val_dataset)
        self.test_dataloader = self.get_dataloader(self.test_dataset)

    def save_class_to_idx(self) -> None:
        save_path = Path(self.cfg.models.resnet152.class_to_idx)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        class_to_idx = self.train_dataset.class_to_idx
        with save_path.open("wb") as save_file:
            np.save(save_file, np.array(list(class_to_idx.items())), allow_pickle=False)

    def load_model(self) -> None:
        num_classes = len(self.train_dataset.classes)
        self.model = CustomResNet152(num_classes)
        self.model.to(self.device)

    def save_to_onnx(self) -> None:
        self.model.eval()
        self.model.to("cpu")
        torch_input = torch.randn(1, 3, 224, 224)
        save_torch_model(self.model, torch_input, self.cfg.models.resnet152.onnx_path)

    def set_training_args(self) -> None:
        torch.set_float32_matmul_precision("medium")
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            [{"params": self.model.trainable_bottleneck.parameters()}, {"params": self.model.classifier.parameters()}],
            lr=self.cfg.models.resnet152.optimizer.lr,
        )

    @staticmethod
    def calculate_metrics(
        y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int
    ) -> tuple[float, float, float, float]:
        f1_macro = multiclass_f1_score(y_pred, y_true, num_classes=num_classes, average="macro")
        f1_micro = multiclass_f1_score(y_pred, y_true, num_classes=num_classes, average="micro")
        f1_weighted = multiclass_f1_score(y_pred, y_true, num_classes=num_classes, average="weighted")
        accuracy = multiclass_accuracy(y_pred, y_true, num_classes=num_classes, average="micro")
        return f1_macro.item(), f1_micro.item(), f1_weighted.item(), accuracy.item()

    def train_epoch(self) -> float:
        running_loss = 0.0

        self.model.trainable_bottleneck.train()
        self.model.classifier.train()

        for data in tqdm(self.train_dataloader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_dataloader)

    def eval_epoch(self) -> tuple[float, float, float, float, float]:
        running_loss = 0.0
        y_true = []
        y_pred = []

        self.model.trainable_bottleneck.eval()
        self.model.classifier.eval()

        with torch.inference_mode():
            for data in tqdm(self.val_dataloader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                y_true.append(labels.cpu())
                y_pred.append(predicted.cpu())

            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)
            f1_macro, f1_micro, f1_weighted, accuracy = self.calculate_metrics(y_pred, y_true, self.model.num_classes)

            return running_loss / len(self.val_dataloader), f1_macro, f1_micro, f1_weighted, accuracy

    def train(self) -> None:
        if self.cfg.log_wandb:
            wandb.init(project="sneakers_ml")

        for _ in range(self.cfg.models.resnet152.num_epochs):
            train_loss = self.train_epoch()
            val_loss, f1_macro, f1_micro, f1_weighted, accuracy = self.eval_epoch()
            if self.cfg.log_wandb:
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

        if self.cfg.log_wandb:
            wandb.finish()

        _, f1_macro, f1_micro, f1_weighted, accuracy = self.eval_epoch()
        metrics = {
            "test_f1_macro": f1_macro,
            "test_f1_micro": f1_micro,
            "test_f1_weighted": f1_weighted,
            "test_accuracy": accuracy,
        }

        logger.info(str(metrics))

        self.save_to_onnx()
        log_metrics(metrics, self.cfg.paths.results, self.cfg.models.resnet152.name)


class Resnet152Classifier(DLClassifier):
    def __init__(self, onnx_path: str, class_to_idx_path: str) -> None:
        super().__init__(onnx_path=onnx_path, class_to_idx_path=class_to_idx_path)

        self.weights = ResNet152_Weights.DEFAULT
        self.preprocess = self.weights.transforms()

    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image)  # type: ignore[no-any-return]

    def predict(self, images: Sequence[Image.Image]) -> list[str]:
        preprocessed_images = torch.stack([self.apply_transforms(image) for image in images])
        pred = predict(self.onnx_session, preprocessed_images)
        softmax_pred = softmax(pred, axis=1)
        predictions = np.argmax(softmax_pred, axis=1)
        string_predictions = np.vectorize(self.idx_to_class.get)(predictions)
        return string_predictions.tolist()  # type: ignore[no-any-return]


if __name__ == "__main__":
    # train
    with initialize(version_base=None, config_path="../../config", job_name="resnet152-train"):
        cfg_dl = compose(config_name="cfg_dl")
        ResNet152ClassificationTrainer(cfg_dl).train()

    # predict
    with initialize(version_base=None, config_path="../../config", job_name="resnet152-predict"):
        cfg_dl = compose(config_name="cfg_dl")
        test_image = Image.open("tests/static/newbalance574.jpg")
        print(
            Resnet152Classifier(
                cfg_dl.models.resnet152.onnx_path,
                cfg_dl.models.resnet152.class_to_idx,
            ).predict([test_image])
        )
        print(
            Resnet152Classifier(
                cfg_dl.models.resnet152.onnx_path,
                cfg_dl.models.resnet152.class_to_idx,
            ).predict([test_image, test_image, test_image])
        )
