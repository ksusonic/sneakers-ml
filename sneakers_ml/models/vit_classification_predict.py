from collections.abc import Sequence

import numpy as np
import torch
from hydra import compose, initialize
from PIL import Image
from scipy.special import softmax
from transformers import ViTImageProcessor

from sneakers_ml.models.dl_classification_base import DLClassifier
from sneakers_ml.models.onnx_utils import predict


class ViTClassifier(DLClassifier):
    def __init__(self, onnx_path: str, class_to_idx_path: str, base_model: str) -> None:
        super().__init__(onnx_path=onnx_path, class_to_idx_path=class_to_idx_path)

        self.base_model = base_model
        self.preprocess = ViTImageProcessor.from_pretrained(self.base_model)

    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image, return_tensors="pt")["pixel_values"].squeeze(0)  # type: ignore[no-any-return]

    def predict(self, images: Sequence[Image.Image]) -> tuple[list, list]:
        preprocessed_images = torch.stack([self.apply_transforms(image) for image in images])
        pred = predict(self.onnx_session, preprocessed_images)
        softmax_pred = softmax(pred, axis=1)
        predictions = np.argmax(softmax_pred, axis=1)
        string_predictions = np.vectorize(self.idx_to_class.get)(predictions)
        return string_predictions.tolist()


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../config", job_name="vit-predict"):
        cfg_dl = compose(config_name="cfg_dl")
        test_image = Image.open("data/training/brands-classification/train/adidas/1.jpeg")
        print(
            ViTClassifier(
                cfg_dl.models.vit_transformer.onnx_path,
                cfg_dl.models.vit_transformer.class_to_idx,
                cfg_dl.models.vit_transformer.hf_name,
            ).predict([test_image])
        )
        print(
            ViTClassifier(
                cfg_dl.models.vit_transformer.onnx_path,
                cfg_dl.models.vit_transformer.class_to_idx,
                cfg_dl.models.vit_transformer.hf_name,
            ).predict([test_image, test_image, test_image])
        )
