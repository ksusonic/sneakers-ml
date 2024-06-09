from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch.utils.data
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from sneakers_ml.models.onnx_utils import get_device, get_session


class SimilaritySearchBase(ABC):  # noqa: B024
    """ """

    def __init__(self, embeddings_path: str, onnx_path: str, device: str = "cpu") -> None:
        self.embeddings_path = embeddings_path
        self.onnx_path = onnx_path
        self.device = get_device(device)

    @staticmethod
    def save_features(path: str, numpy_features: np.ndarray, classes: np.ndarray, class_to_idx: dict[str, int]) -> None:
        """

        :param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: param path: str:
        :param numpy_features: np.ndarray:
        :param classes: np.ndarray:
        :param class_to_idx: dict[str:
        :param int: 
        :param path: str: 
        :param numpy_features: np.ndarray: 
        :param classes: np.ndarray: 
        :param class_to_idx: dict[str: 
        :param int]: 

        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as save_file:
            np.save(save_file, numpy_features, allow_pickle=False)
            np.save(save_file, classes, allow_pickle=False)
            np.save(save_file, np.array(list(class_to_idx.items())), allow_pickle=False)

    @staticmethod
    def load_features(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        """

        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str:
        :param path: str: 

        """
        with Path(path).open("rb") as file:
            numpy_features = np.load(file, allow_pickle=False)
            classes = np.load(file, allow_pickle=False)
            class_to_idx_numpy = np.load(file, allow_pickle=False)
            class_to_idx = dict(zip(class_to_idx_numpy[:, 0], class_to_idx_numpy[:, 1].astype(int)))
            return numpy_features, classes, class_to_idx


class SimilaritySearchPredictor(SimilaritySearchBase):
    """ """

    def __init__(self, embeddings_path: str, onnx_path: str, metadata_path: str, device: str = "cpu") -> None:
        super().__init__(embeddings_path, onnx_path, device)

        self.metadata_path = metadata_path
        self.df = self.get_metadata(self.metadata_path)

        self.numpy_features, self.classes, self.class_to_idx = self.load_features(self.embeddings_path)
        self.idx_to_class = {str(v): k for k, v in self.class_to_idx.items()}

        self.onnx_session = get_session(self.onnx_path)

    @abstractmethod
    def get_features(self) -> np.ndarray:
        """ """
        raise NotImplementedError

    @abstractmethod
    def predict(self, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """

        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int:
        :param top_k: int: 

        """
        raise NotImplementedError

    def get_metadata(self, metadata_path: str) -> pd.DataFrame:
        """

        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str:
        :param metadata_path: str: 

        """
        df = pd.read_csv(metadata_path)
        df = df.drop(
            ["brand_merge", "images_path", "collection_name", "color", "images_flattened", "title_without_color"],
            axis=1,
        )
        df["title"] = df["title"].apply(eval)
        df["brand"] = df["brand"].apply(eval)
        df["price"] = df["price"].apply(eval)
        df["pricecurrency"] = df["pricecurrency"].apply(eval)
        df["website"] = df["website"].apply(eval)
        df["url"] = df["url"].apply(eval)
        df = df.explode(["title", "brand", "price", "pricecurrency", "url", "website"])

        return df

    def get_similar(self, feature: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """

        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray:
        :param top_k: int:
        :param feature: np.ndarray: 
        :param top_k: int: 

        """
        similarity_matrix = cosine_similarity(self.numpy_features, feature).flatten()
        top_k_indices = np.argsort(similarity_matrix)[-top_k:][::-1]

        similar_objects = self.classes[top_k_indices]
        similar_images = similar_objects[:, 0]
        similar_models = np.vectorize(self.idx_to_class.get)(similar_objects[:, 1])

        similar_metadata_dump = (
            self.df[self.df["title_merge"].isin(set(similar_models.tolist()))]
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
            .to_numpy()
        )
        return similar_metadata_dump, similar_images


class SimilaritySearchTrainer(SimilaritySearchBase):
    """ """

    def __init__(self, image_folder: str, embeddings_path: str, onnx_path: str, device: str = "cpu") -> None:
        super().__init__(embeddings_path, onnx_path, device)
        self.image_folder = image_folder
        self.dataset: torch.utils.data.Dataset = None
        self.dataloader: torch.utils.data.DataLoader = None

        self.numpy_image_features: np.ndarray = None
        self.image_paths: np.ndarray = None
        self.class_to_idx: dict[str, int] = None

    @abstractmethod
    def create_onnx_model(self) -> None:
        """ """
        raise NotImplementedError

    @abstractmethod
    def init_data(self) -> None:
        """ """
        raise NotImplementedError

    @abstractmethod
    def init_model(self) -> None:
        """ """
        raise NotImplementedError

    @abstractmethod
    def model_forward(self, data: Sequence[Any]) -> torch.Tensor:
        """

        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]: 

        """
        raise NotImplementedError

    def get_image_folder_features(self) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        """ """
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
        """ """
        self.init_model()
        self.create_onnx_model()
        self.init_data()
        self.numpy_image_features, self.image_paths, self.class_to_idx = self.get_image_folder_features()
        self.save_features(self.embeddings_path, self.numpy_image_features, self.image_paths, self.class_to_idx)
