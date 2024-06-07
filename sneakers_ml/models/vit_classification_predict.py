from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from PIL import Image
from scipy.special import softmax
from transformers import ViTImageProcessor

from sneakers_ml.models.onnx_utils import get_session, predict


class ViTClassifier:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        with Path(self.cfg.models.vit_transformer.idx_to_classes).open("rb") as file:
            class_to_idx_numpy = np.load(file, allow_pickle=False)
            self.class_to_idx = dict(zip(class_to_idx_numpy[:, 1].astype(int), class_to_idx_numpy[:, 0]))

        self.preprocess = ViTImageProcessor.from_pretrained(self.cfg.models.vit_transformer.hf_name)
        self.onnx_session = get_session(self.cfg.models.vit_transformer.onnx_path, "cpu")

    def _apply_transforms(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image, return_tensors="pt")["pixel_values"].squeeze(0)  # type: ignore[no-any-return]

    def predict(self, images: Sequence[Image.Image]) -> tuple[list, list]:
        preprocessed_images = torch.stack([self._apply_transforms(image) for image in images])
        pred = predict(self.onnx_session, preprocessed_images)
        softmax_pred = softmax(pred, axis=1)
        predictions = np.argmax(softmax_pred, axis=1)
        string_predictions = np.vectorize(self.class_to_idx.get)(predictions)
        return string_predictions.tolist()


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../config", job_name="vit-predict"):
        cfg_dl = compose(config_name="cfg_dl")
        test_image = Image.open("data/training/brands-classification/train/adidas/1.jpeg")
        print(ViTClassifier(cfg_dl).predict([test_image]))
        print(ViTClassifier(cfg_dl).predict([test_image, test_image, test_image]))
