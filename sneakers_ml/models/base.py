import csv
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from sneakers_ml.models.onnx_utils import get_device, get_session
from sneakers_ml.models.quadrant import Qdrant

if TYPE_CHECKING:
    from torchvision.datasets import ImageFolder


class SimilaritySearchBase(ABC):  # noqa: B024
    def __init__(
        self, embeddings_path: str, onnx_path: str, device: str = "cpu", qdrant: Optional[Qdrant] = None
    ) -> None:
        self.embeddings_path = embeddings_path
        self.onnx_path = onnx_path
        self.device = get_device(device)
        self.qdrant = qdrant

    @staticmethod
    def save_features(path: str, numpy_features: np.ndarray, classes: np.ndarray, class_to_idx: dict[str, int]) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as save_file:
            np.save(save_file, numpy_features, allow_pickle=False)
            np.save(save_file, classes, allow_pickle=False)
            np.save(save_file, np.array(list(class_to_idx.items())), allow_pickle=False)

    @staticmethod
    def load_features(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        with Path(path).open("rb") as file:
            numpy_features = np.load(file, allow_pickle=False)
            classes = np.load(file, allow_pickle=False)
            class_to_idx_numpy = np.load(file, allow_pickle=False)
            class_to_idx = dict(zip(class_to_idx_numpy[:, 0], class_to_idx_numpy[:, 1].astype(int)))
            return numpy_features, classes, class_to_idx


class SimilaritySearchPredictor(SimilaritySearchBase):
    def __init__(
        self,
        embeddings_path: str,
        onnx_path: str,
        metadata_path: str,
        device: str = "cpu",
        qdrant: Optional[Qdrant] = None,
    ) -> None:
        super().__init__(embeddings_path, onnx_path, device, qdrant)

        self.metadata_path = metadata_path
        self.metadata_df = self.get_metadata(self.metadata_path)

        if not qdrant:
            self.numpy_features, self.classes, self.class_to_idx = self.load_features(self.embeddings_path)
            self.idx_to_class = {str(v): k for k, v in self.class_to_idx.items()}

        self.onnx_session = get_session(self.onnx_path, self.device)

    @abstractmethod
    def get_features(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict(self, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_metadata(self, metadata_path: str) -> pd.DataFrame:
        metadata_df = pd.read_csv(metadata_path)
        metadata_df = metadata_df.drop(
            ["brand_merge", "images_path", "collection_name", "color", "images_flattened", "title_without_color"],
            axis=1,
        )
        metadata_df["title"] = metadata_df["title"].apply(eval)
        metadata_df["brand"] = metadata_df["brand"].apply(eval)
        metadata_df["price"] = metadata_df["price"].apply(eval)
        metadata_df["pricecurrency"] = metadata_df["pricecurrency"].apply(eval)
        metadata_df["website"] = metadata_df["website"].apply(eval)
        metadata_df["url"] = metadata_df["url"].apply(eval)
        return metadata_df.explode(["title", "brand", "price", "pricecurrency", "url", "website"])

    def get_similar(self, feature: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.qdrant:
            similar_images, similar_models = self.qdrant.get_similar(feature, top_k)
        else:
            similarity_matrix = cosine_similarity(self.numpy_features, feature).flatten()
            top_k_indices = np.argsort(similarity_matrix)[-top_k:][::-1]

            similar_objects = self.classes[top_k_indices]
            similar_images = similar_objects[:, 0]
            similar_models = np.vectorize(self.idx_to_class.get)(similar_objects[:, 1])

        similar_metadata_dump = (
            self.metadata_df[self.metadata_df["title_merge"].isin(np.unique(similar_models).tolist())]
            .groupby(["title", "website"])
            .agg(
                {
                    "title_merge": "first",
                    "brand": "first",
                    "price": lambda x: f"{min(x)} - {max(x)}",
                    "pricecurrency": "first",
                    "url": "first",
                }
            )
            .reset_index()
            .set_index("title_merge")
            .loc[similar_models]
            .reset_index()
            .to_numpy()
        )
        return similar_metadata_dump, similar_images


class SimilaritySearchTrainer(SimilaritySearchBase):
    def __init__(
        self,
        image_folder: str,
        embeddings_path: str,
        onnx_path: str,
        device: str = "cpu",
        qdrant: Optional[Qdrant] = None,
    ) -> None:
        super().__init__(embeddings_path, onnx_path, device, qdrant)
        self.image_folder = image_folder
        self.dataset: ImageFolder = None
        self.dataloader: torch.utils.data.DataLoader[Any] = None

        self.numpy_image_features: np.ndarray = None
        self.image_paths: np.ndarray = None
        self.class_to_idx: dict[str, int] = None

    @abstractmethod
    def create_onnx_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_data(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def model_forward(self, data: Sequence[Any]) -> torch.Tensor:
        raise NotImplementedError

    def get_image_folder_features(self) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        image_features = []
        with torch.inference_mode():
            for data in tqdm(self.dataloader, desc=self.image_folder):
                outputs = self.model_forward(data)
                image_features.append(outputs)

        self.numpy_image_features = torch.cat(image_features, dim=0).numpy()
        self.image_paths = np.array(self.dataset.imgs)
        self.class_to_idx = self.dataset.class_to_idx

        return self.numpy_image_features, self.image_paths, self.class_to_idx

    def train(self) -> None:
        self.init_model()
        self.init_data()
        self.numpy_image_features, self.image_paths, self.class_to_idx = self.get_image_folder_features()
        if self.qdrant:
            self.qdrant.save_features_quadrant(self.numpy_image_features, self.image_paths, self.class_to_idx)
        else:
            self.save_features(self.embeddings_path, self.numpy_image_features, self.image_paths, self.class_to_idx)
        self.create_onnx_model()


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


def log_metrics(metrics: dict[str, float], save_path: str, model_name: str) -> None:
    list_metrics = list(metrics.values())
    for i, metric in enumerate(metrics):
        list_metrics[i] = round(metric, 2)
    results_save_path = Path(save_path)

    if not results_save_path.exists():
        with results_save_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["model_name", "f1_macro", "f1_micro", "f1_weighted", "accuracy"])

    with results_save_path.open("a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([model_name, *list_metrics])
