from pathlib import Path
from typing import Union

import numpy as np
import onnxruntime as rt
import torch
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from skl2onnx import to_onnx
from sklearn.base import BaseEstimator


def get_device(device: str) -> str:
    """

    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:
    :param device: str:

    """
    if device.lower().startswith("cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def get_providers(device: str = "cpu") -> list[str]:
    """

    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")
    :param device: str:  (Default value = "cpu")

    """
    return ["CUDAExecutionProvider", "CPUExecutionProvider"
            ] if device == "cuda" else ["CPUExecutionProvider"]


def get_session(model_path: str, device: str = "cpu") -> rt.InferenceSession:
    """

    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")
    :param model_path: str:
    :param device: str:  (Default value = "cpu")

    """
    device = get_device(device)
    providers = get_providers(device)
    return rt.InferenceSession(model_path, providers=providers)


def save_torch_model(model: torch.nn.Module, torch_input_tensor: torch.Tensor,
                     model_path: str) -> None:
    """

    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensor: torch.Tensor:
    :param model_path: str:

    """
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch_input_tensor,
        str(model_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {
                0: "batch_size"
            },
            "output": {
                0: "batch_size"
            }
        },
    )


def save_clip_model(model: torch.nn.Module,
                    torch_input_tensors: tuple[torch.Tensor],
                    model_path: str) -> None:
    """

    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:
    :param model: torch.nn.Module:
    :param torch_input_tensors: tuple[torch.Tensor]:
    :param model_path: str:

    """
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        torch_input_tensors,
        model_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_features"],
        dynamic_axes={
            "input_ids": {
                0: "batch_size",
                1: "sequence_length"
            },
            "attention_mask": {
                0: "batch_size",
                1: "sequence_length"
            },
            "text_features": {
                0: "batch_size"
            },
        },
        opset_version=13,
    )


def save_sklearn_model(model: BaseEstimator, x: np.ndarray, path: str) -> None:
    """

    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:
    :param model: BaseEstimator:
    :param x: np.ndarray:
    :param path: str:

    """
    onx = to_onnx(model, x[:1].astype(np.float32))
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as file:
        file.write(onx.SerializeToString())


def save_catboost_model(model: Union[CatBoostRegressor, CatBoostClassifier],
                        path: str) -> None:
    """

    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier: param path: str:
    :param model: Union[CatBoostRegressor:
    :param CatBoostClassifier]:
    :param path: str:

    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(
        str(path),
        format="onnx",
        export_parameters={
            "onnx_domain": "ai.catboost",
            "onnx_model_version": 1,
            "onnx_doc_string": "Empty",
            "onnx_graph_name": "CatBoostModel",
        },
    )


def save_model(
    model: Union[BaseEstimator, torch.nn.Module, CatBoostRegressor,
                 CatBoostClassifier],
    x: Union[np.ndarray, torch.Tensor],
    path: str,
) -> None:
    """

    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostRegressor: param CatBoostClassifier]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch: nn.Module:
    :param CatBoostClassifier: param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param path: str:
    :param model: Union[BaseEstimator:
    :param torch.nn.Module:
    :param CatBoostClassifier]:
    :param x: Union[np.ndarray:
    :param torch.Tensor]:
    :param path: str:

    """
    if isinstance(model, torch.nn.Module):
        return save_torch_model(model, x, path)
    if isinstance(model, BaseEstimator):
        return save_sklearn_model(model, x, path)
    if isinstance(model, (CatBoostRegressor, CatBoostClassifier)):
        return save_catboost_model(model, path)
    msg = "Unknown model"
    raise ValueError(msg)


def format_inputs(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """

    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param x: Union[np.ndarray:
    :param torch.Tensor]:

    """
    if isinstance(x, torch.Tensor):
        # type: ignore[no-any-return]
        return x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    msg = "Unknown input"
    raise ValueError(msg)


def predict(onnx_session: rt.InferenceSession,
            x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """

    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch: Tensor]:
    :param onnx_session: rt.InferenceSession:
    :param x: Union[np.ndarray:
    :param torch.Tensor]:

    """
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    input_value = format_inputs(x)
    # type: ignore[no-any-return]
    return onnx_session.run([output_name], {input_name: input_value})[0]


def predict_clip(onnx_session: rt.InferenceSession,
                 x: dict[str, np.array]) -> np.ndarray:
    """

    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np: array]:
    :param onnx_session: rt.InferenceSession:
    :param x: dict[str:
    :param np.array]:

    """
    input_name_1 = onnx_session.get_inputs()[0].name
    input_name_2 = onnx_session.get_inputs()[1].name
    output_name = onnx_session.get_outputs()[0].name
    x[input_name_1] = x[input_name_1].astype(np.int64)
    x[input_name_2] = x[input_name_2].astype(np.int64)
    # type: ignore[no-any-return]
    return onnx_session.run([output_name], dict(x))[0]
