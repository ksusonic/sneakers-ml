from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet152_Weights, resnet152
from tqdm import tqdm

from sneakers_ml.models.onnx_utils import get_device, save_torch_model


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x


class ResNet152SimilaritySearchCreator:
    def __init__(self, device: str, onnx_path: str) -> None:
        self.device = get_device(device)
        self.onnx_path = onnx_path

        self.weights = ResNet152_Weights.DEFAULT
        self.preprocess = self.weights.transforms()

        self.model = self._initialize_torch_resnet()
        self.model.to(self.device)

        self._create_onnx_model(self.onnx_path)

    def _initialize_torch_resnet(self) -> torch.nn.Module:
        model = resnet152(weights=self.weights)
        model.fc = Identity()
        model.eval()
        return model  # type: ignore[no-any-return]

    def _create_onnx_model(self, save_path) -> None:
        model = self._initialize_torch_resnet()
        torch_input = torch.randn(1, 3, 224, 224)
        save_torch_model(model, torch_input, save_path)

    def get_features_folder(self, folder_path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:

        dataset = ImageFolder(folder_path, transform=self.preprocess)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

        features = []
        with torch.inference_mode():
            for data in tqdm(dataloader, desc=folder_path):
                x = data[0].to(self.device)
                prediction = self.model(x)

                features.append(prediction.cpu())

        full_images_features = torch.cat(features, dim=0)
        numpy_features = full_images_features.numpy()
        classes = np.array(dataset.imgs)

        return numpy_features, classes, dataset.class_to_idx

    @staticmethod
    def _save_features(
        path: str, numpy_features: np.ndarray, classes: np.ndarray, class_to_idx: dict[str, int]
    ) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as save_file:
            np.save(save_file, numpy_features, allow_pickle=False)
            np.save(save_file, classes, allow_pickle=False)
            np.save(save_file, np.array(list(class_to_idx.items())), allow_pickle=False)

    @staticmethod
    def _load_features(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        with Path(path).open("rb") as file:
            numpy_features = np.load(file, allow_pickle=False)
            classes = np.load(file, allow_pickle=False)
            class_to_idx_numpy = np.load(file, allow_pickle=False)
            class_to_idx = dict(zip(class_to_idx_numpy[:, 0], class_to_idx_numpy[:, 1].astype(int)))
            return numpy_features, classes, class_to_idx


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../config", job_name="similarity-search-features-create"):
        cfg = compose(config_name="cfg_similarity_search")
        creator = ResNet152SimilaritySearchCreator(cfg.device, cfg.model_path)
        numpy_features, classes, class_to_idx = creator.get_features_folder(cfg.images_path)
        creator._save_features(cfg.embeddings_path, numpy_features, classes, class_to_idx)
