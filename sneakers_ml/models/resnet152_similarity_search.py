from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import torch
from hydra import compose, initialize
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet152_Weights, resnet152

from sneakers_ml.models.base import SimilaritySearchPredictor, SimilaritySearchTrainer
from sneakers_ml.models.onnx_utils import predict, save_torch_model
from sneakers_ml.models.qdrant import Qdrant


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x


class ResNet152SimilaritySearchTrainer(SimilaritySearchTrainer):
    def __init__(
        self, image_folder: str, onnx_path: str, embeddings_path: str, device: str, qdrant: Optional[Qdrant] = None
    ) -> None:
        super().__init__(
            image_folder=image_folder,
            embeddings_path=embeddings_path,
            onnx_path=onnx_path,
            device=device,
            qdrant=qdrant,
        )
        self.model: Optional[torch.nn.Module] = None
        self.preprocess = None
        self.weights: ResNet152_Weights.IMAGENET1K_V2 = None

    def init_data(self) -> None:
        self.dataset = ImageFolder(self.image_folder, transform=self.preprocess)
        self.dataloader = DataLoader(
            self.dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=6, pin_memory=False
        )

    def init_model(self) -> None:
        self.weights = ResNet152_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.model = self._initialize_torch_resnet()
        self.model.to(self.device)

    def _initialize_torch_resnet(self) -> torch.nn.Module:
        model = resnet152(weights=self.weights)
        model.fc = Identity()
        model.eval()
        return model  # type: ignore[no-any-return]

    def create_onnx_model(self) -> None:
        model = self._initialize_torch_resnet()
        torch_input = torch.randn(1, 3, 224, 224)
        save_torch_model(model, torch_input, self.onnx_path)

    def model_forward(self, data: Sequence[Any]) -> torch.Tensor:
        x = data[0].to(self.device)
        return self.model(x).cpu()  # type: ignore[no-any-return]


class ResNet152SimilaritySearch(SimilaritySearchPredictor):
    def __init__(
        self, embeddings_path: str, onnx_path: str, metadata_path: str, qdrant: Optional[Qdrant] = None
    ) -> None:
        super().__init__(embeddings_path, onnx_path, metadata_path, qdrant=qdrant)

        self.weights = ResNet152_Weights.DEFAULT
        self.preprocess = self.weights.transforms()

    def get_features(self, images: Optional[Sequence[Image.Image]] = None) -> np.ndarray:
        preprocessed_images = torch.stack([self.preprocess(image) for image in images])
        return predict(self.onnx_session, preprocessed_images)

    def predict(self, top_k: int, image: Optional[Image.Image] = None) -> tuple[np.ndarray, np.ndarray]:
        return self.get_similar(self.get_features([image]), top_k)


if __name__ == "__main__":
    # train
    with initialize(version_base=None, config_path="../../config", job_name="resnet152-similarity-search-create"):
        cfg = compose(config_name="cfg_resnet152_similarity_search")
        trainer = ResNet152SimilaritySearchTrainer(cfg.images_path, cfg.model_path, cfg.embeddings_path, cfg.device)
        trainer.train()

    # predict
    with initialize(version_base=None, config_path="../../config", job_name="resnet152-similarity-search-predict"):
        cfg = compose(config_name="cfg_resnet152_similarity_search")
        predictor = ResNet152SimilaritySearch(cfg.embeddings_path, cfg.model_path, cfg.metadata_path)
        test_image = Image.open("tests/static/newbalance574.jpg")
        print(predictor.predict(3, test_image))
