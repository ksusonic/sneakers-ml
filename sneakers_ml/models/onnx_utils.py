from pathlib import Path
from typing import Union

import numpy as np
import onnxruntime as rt
import torch
from catboost import CatBoostClassifier, CatBoostRegressor
from skl2onnx import to_onnx
from sklearn.base import BaseEstimator


def get_device(device: str) -> str:
    if device.lower().startswith("cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def get_providers(device: str = "cpu") -> list[str]:
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]


def get_session(model_path: str, device: str = "cpu") -> rt.InferenceSession:
    device = get_device(device)
    providers = get_providers(device)
    return rt.InferenceSession(model_path, providers=providers)


def save_torch_model(
    model: torch.nn.Module, torch_input_tensor: Union[torch.Tensor, tuple[torch.Tensor]], model_path: str
) -> None:
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch_input_tensor,
        str(model_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def save_clip_text_model(model: torch.nn.Module, torch_input_tensors: tuple[torch.Tensor], model_path: str) -> None:
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        torch_input_tensors,
        model_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_features"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "text_features": {0: "batch_size"},
        },
        opset_version=13,
    )


def save_clip_vision_model(model: torch.nn.Module, torch_input_tensors: tuple[torch.Tensor], model_path: str) -> None:
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        torch_input_tensors,
        model_path,
        input_names=["pixel_values"],
        output_names=["image_embeds", "last_hidden_state"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeds": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
        },
        opset_version=13,
    )


def save_sklearn_model(model: BaseEstimator, x: np.ndarray, path: str) -> None:
    onx = to_onnx(model, x[:1].astype(np.float32))
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as file:
        file.write(onx.SerializeToString())


def save_catboost_model(model: Union[CatBoostRegressor, CatBoostClassifier], path: str) -> None:
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
    model: Union[BaseEstimator, torch.nn.Module, CatBoostRegressor, CatBoostClassifier],
    x: Union[np.ndarray, torch.Tensor],
    path: str,
) -> None:
    if isinstance(model, torch.nn.Module):
        return save_torch_model(model, x, path)
    if isinstance(model, BaseEstimator):
        return save_sklearn_model(model, x, path)
    if isinstance(model, (CatBoostRegressor, CatBoostClassifier)):
        return save_catboost_model(model, path)
    msg = "Unknown model"
    raise ValueError(msg)


def format_inputs(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()  # type: ignore[no-any-return]
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    msg = "Unknown input"
    raise ValueError(msg)


def predict(onnx_session: rt.InferenceSession, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    input_value = format_inputs(x)
    return onnx_session.run([output_name], {input_name: input_value})[0]  # type: ignore[no-any-return]


def predict_clip_text(onnx_session: rt.InferenceSession, x: dict[str, np.ndarray]) -> np.ndarray:
    input_name_1 = onnx_session.get_inputs()[0].name
    input_name_2 = onnx_session.get_inputs()[1].name
    output_name = onnx_session.get_outputs()[0].name
    x[input_name_1] = x[input_name_1].astype(np.int64)
    x[input_name_2] = x[input_name_2].astype(np.int64)
    return onnx_session.run([output_name], dict(x))[0]  # type: ignore[no-any-return]


def predict_clip_image(onnx_session: rt.InferenceSession, x: dict[str, np.ndarray]) -> np.ndarray:
    # input_name = onnx_session.get_inputs()[0].name
    output_name_1 = onnx_session.get_outputs()[0].name
    # output_name_2 = onnx_session.get_outputs()[1].name
    return onnx_session.run([output_name_1], dict(x))[0]  # type: ignore[no-any-return]
