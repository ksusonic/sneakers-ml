from collections.abc import Sequence

import numpy as np
import torch
from hydra import compose, initialize
from PIL import Image
from scipy.special import softmax
from torchvision.models import ResNet152_Weights

from sneakers_ml.models.dl_classification_base import DLClassifier
from sneakers_ml.models.onnx_utils import predict


class Resnet152Classifier(DLClassifier):
    def __init__(self, onnx_path: str, class_to_idx_path: str) -> None:
        super().__init__(onnx_path=onnx_path, class_to_idx_path=class_to_idx_path)

        self.weights = ResNet152_Weights.DEFAULT
        self.preprocess = self.weights.transforms()

    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image)  # type: ignore[no-any-return]

    def predict(self, images: Sequence[Image.Image]) -> tuple[list, list]:
        preprocessed_images = torch.stack([self.apply_transforms(image) for image in images])
        pred = predict(self.onnx_session, preprocessed_images)
        softmax_pred = softmax(pred, axis=1)
        predictions = np.argmax(softmax_pred, axis=1)
        string_predictions = np.vectorize(self.idx_to_class.get)(predictions)
        return string_predictions.tolist()


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../config", job_name="resnet152-predict"):
        cfg_dl = compose(config_name="cfg_dl")
        test_image = Image.open("data/training/brands-classification/train/adidas/1.jpeg")
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
