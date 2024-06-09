from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sneakers_ml.models.onnx_utils import get_session


class DLClassifier(ABC):
    def __init__(self, onnx_path: str, class_to_idx_path: str) -> None:
        self.onnx_path = onnx_path
        self.class_to_idx_path = class_to_idx_path

        self.onnx_session = get_session(self.onnx_path)
        self.idx_to_class = self.load_class_to_idx(self.class_to_idx_path)

    @staticmethod
    def load_class_to_idx(class_to_idx_path: str) -> dict[int, str]:
        with Path(class_to_idx_path).open("rb") as file:
            class_to_idx_numpy = np.load(file, allow_pickle=False)
            return dict(zip(class_to_idx_numpy[:, 1].astype(int), class_to_idx_numpy[:, 0]))

    @abstractmethod
    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def predict(self, images: Sequence[Image.Image]) -> list[str]:
        raise NotImplementedError
