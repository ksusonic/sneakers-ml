from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from PIL import Image
from scipy.special import softmax
from torchvision.models import ResNet152_Weights

from sneakers_ml.models.onnx_utils import get_session, predict


class Resnet152Classifier:
    def __init__(self, config: DictConfig) -> None:
        self.config = config

        with Path(cfg.models.resnet152.idx_to_classes).open("rb") as file:
            class_to_idx_numpy = np.load(file, allow_pickle=False)
            self.class_to_idx = dict(zip(class_to_idx_numpy[:, 1].astype(int), class_to_idx_numpy[:, 0]))

        self.weights = ResNet152_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.onnx_session = get_session(f"{cfg.paths.models_save}/{cfg.models.resnet152.name}.onnx", "cpu")

    def _apply_transforms(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image)  # type: ignore[no-any-return]

    def predict(self, images: Sequence[Image.Image]) -> tuple[list, list]:

        preprocessed_images = torch.stack([self._apply_transforms(image) for image in images])
        pred = predict(self.onnx_session, preprocessed_images)
        softmax_pred = softmax(pred, axis=1)
        predictions = np.argmax(softmax_pred, axis=1)
        string_predictions = np.vectorize(self.class_to_idx.get)(predictions)
        return predictions.tolist(), string_predictions.tolist()


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../config", job_name="resnet152-predict"):
        cfg = compose(config_name="cfg_dl")
        image = Image.open("data/training/brands-classification/train/adidas/1.jpeg")
        print(Resnet152Classifier(cfg).predict([image]))
        print(Resnet152Classifier(cfg).predict([image, image, image]))
