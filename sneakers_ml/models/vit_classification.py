from collections.abc import Sequence
from pathlib import Path

import evaluate
import numpy as np
import torch
import transformers
import wandb
from datasets import load_dataset
from hydra import compose, initialize
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from scipy.special import softmax
from transformers import Trainer, TrainingArguments, ViTForImageClassification, ViTImageProcessor

from sneakers_ml.models.base import DLClassifier, log_metrics
from sneakers_ml.models.onnx_utils import predict, save_torch_model


class ViTClassificationTrainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.temp_model_path = "./vit"

        self.load_dataset()
        self.load_model()
        self.save_class_to_idx()
        self.set_training_args()

    def load_dataset(self) -> None:
        self.processor = ViTImageProcessor.from_pretrained(self.cfg.models.vit.hf_name)
        self.dataset = load_dataset("imagefolder", data_dir=self.cfg.data.path)
        self.processed_train = self.dataset["train"].with_transform(self.transform)
        self.processed_val = self.dataset["validation"].with_transform(self.transform)
        self.processed_test = self.dataset["test"].with_transform(self.transform)

    def set_training_args(self) -> None:
        torch.set_float32_matmul_precision("medium")

        self.metric_f1 = evaluate.load("f1")
        self.metric_accuracy = evaluate.load("accuracy")

        self.training_args = TrainingArguments(
            output_dir=self.temp_model_path,
            per_device_train_batch_size=self.cfg.models.vit.dataloader.batch_size,
            per_device_eval_batch_size=self.cfg.models.vit.dataloader.batch_size,
            eval_strategy="steps",
            num_train_epochs=self.cfg.models.vit.num_epochs,
            fp16=True,
            save_steps=500,
            eval_steps=500,
            logging_steps=500,
            learning_rate=self.cfg.models.vit.optimizer.lr,
            weight_decay=self.cfg.models.vit.optimizer.weight_decay,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="wandb" if self.cfg.log_wandb else "none",
            load_best_model_at_end=True,
            dataloader_drop_last=False,
            dataloader_pin_memory=self.cfg.models.vit.dataloader.pin_memory,
            dataloader_num_workers=self.cfg.models.vit.dataloader.num_workers,
        )

    def load_model(self) -> None:
        labels = self.dataset["train"].features["label"].names
        self.id2label = {str(i): c for i, c in enumerate(labels)}
        self.label2id = {c: str(i) for i, c in enumerate(labels)}
        self.model = ViTForImageClassification.from_pretrained(
            self.cfg.models.vit.hf_name,
            num_labels=len(labels),
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def save_to_onnx(self) -> None:
        loaded_model = ViTForImageClassification.from_pretrained(self.temp_model_path)
        loaded_model.eval()
        torch_input = (torch.randn(1, 3, 224, 224),)
        save_torch_model(loaded_model, torch_input, self.cfg.models.vit.onnx_path)

    def save_class_to_idx(self) -> None:
        save_path = Path(self.cfg.models.vit.class_to_idx)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as save_file:
            np.save(save_file, np.array(list(self.label2id.items())), allow_pickle=False)

    def transform(self, example_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs = self.processor(list(example_batch["image"]), return_tensors="pt")
        inputs["labels"] = example_batch["label"]
        return inputs  # type: ignore[no-any-return]

    def collate_fn(self, batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x["labels"] for x in batch]),
        }

    def compute_metrics(self, p: transformers.trainer_utils.EvalPrediction) -> dict[str, float]:
        predictions = np.argmax(p.predictions, axis=1)
        references = p.label_ids

        f1_macro = self.metric_f1.compute(predictions=predictions, references=references, average="macro")["f1"]
        f1_micro = self.metric_f1.compute(predictions=predictions, references=references, average="micro")["f1"]
        f1_weighted = self.metric_f1.compute(predictions=predictions, references=references, average="weighted")["f1"]
        accuracy = self.metric_accuracy.compute(predictions=predictions, references=references)["accuracy"]

        return {"f1_macro": f1_macro, "f1_micro": f1_micro, "f1_weighted": f1_weighted, "accuracy": accuracy}

    def train(self) -> None:
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            train_dataset=self.processed_train,
            eval_dataset=self.processed_val,
            tokenizer=self.processor,
        )
        wandb.init(project="sneakers_ml")

        trainer.train()
        trainer.save_model()
        trainer.save_state()

        eval_metrics = trainer.evaluate(self.processed_test)
        true_metrics = {
            "test_f1_macro": eval_metrics["eval_f1_macro"],
            "test_f1_micro": eval_metrics["eval_f1_micro"],
            "test_f1_weighted": eval_metrics["eval_f1_weighted"],
            "test_accuracy": eval_metrics["eval_accuracy"],
        }

        logger.info(str(true_metrics))
        self.save_to_onnx()
        log_metrics(true_metrics, self.cfg.paths.results, self.cfg.models.vit.name)


class ViTClassifier(DLClassifier):
    def __init__(self, onnx_path: str, class_to_idx_path: str, base_model: str) -> None:
        super().__init__(onnx_path=onnx_path, class_to_idx_path=class_to_idx_path)

        self.base_model = base_model
        self.preprocess = ViTImageProcessor.from_pretrained(self.base_model)

    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image, return_tensors="pt")["pixel_values"].squeeze(0)  # type: ignore[no-any-return]

    def predict(self, images: Sequence[Image.Image]) -> list[str]:
        preprocessed_images = torch.stack([self.apply_transforms(image) for image in images])
        pred = predict(self.onnx_session, preprocessed_images)
        softmax_pred = softmax(pred, axis=1)
        predictions = np.argmax(softmax_pred, axis=1)
        string_predictions = np.vectorize(self.idx_to_class.get)(predictions)
        return string_predictions.tolist()  # type: ignore[no-any-return]


if __name__ == "__main__":
    # train
    with initialize(version_base=None, config_path="../../config", job_name="vit-train"):
        cfg_dl = compose(config_name="cfg_dl")
        ViTClassificationTrainer(cfg_dl).train()

    # predict
    with initialize(version_base=None, config_path="../../config", job_name="vit-predict"):
        cfg_dl = compose(config_name="cfg_dl")
        test_image = Image.open("tests/static/newbalance574.jpg")
        print(
            ViTClassifier(
                cfg_dl.models.vit.onnx_path,
                cfg_dl.models.vit.class_to_idx,
                cfg_dl.models.vit.hf_name,
            ).predict([test_image])
        )
        print(
            ViTClassifier(
                cfg_dl.models.vit.onnx_path,
                cfg_dl.models.vit.class_to_idx,
                cfg_dl.models.vit.hf_name,
            ).predict([test_image, test_image, test_image])
        )
