import os
import typing
from typing import Any, Literal, TypeVar

import torch
from torch.optim import (ASGD, LBFGS, SGD, Adadelta, Adagrad, Adam, Adamax, AdamW, NAdam, RAdam,
                         RMSprop, Rprop, SparseAdam)
from yaml_config_override import add_arguments  # type: ignore

from metrics import EvaluationMetric

OptimizerName = Literal[
    "adadelta",
    "adagrad",
    "adamax",
    "adamw",
    "adam",
    "asgd",
    "lbfgs",
    "nadam",
    "radam",
    "rmsprop",
    "rprop",
    "sgd",
    "sparse_adam"
]

ValidOptimizerNames = typing.get_args(OptimizerName)

OPTIMIZERS = {
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "adamax": Adamax,
    "adamw": AdamW,
    "adam": Adam,
    "asgd": ASGD,
    "lbfgs": LBFGS,
    "nadam": NAdam,
    "radam": RAdam,
    "rmsprop": RMSprop,
    "rprop": Rprop,
    "sgd": SGD,
    "sparse_adam": SparseAdam,
}

ValidEvaluationMetrics = typing.get_args(EvaluationMetric)

class DataConfig:
    train_ratio: float
    test_val_ratio: float
    data_dir: str

    def __init__(self, train_ratio: float, test_val_ratio: float, data_dir: str) -> None:
        if train_ratio <= 0 or train_ratio > 1:
            raise ValueError(f"'train_ratio' of {train_ratio} must be in the range (0, 1]")

        if test_val_ratio <= 0 or test_val_ratio > 1:
            raise ValueError(f"'test_val_ratio' of {test_val_ratio} must be in the range (0, 1]")

        if not os.path.isdir(data_dir):
            raise ValueError(f"'data_dir' of '{data_dir}' must be a valid directory path")

        self.train_ratio = train_ratio
        self.test_val_ratio = test_val_ratio
        self.data_dir = data_dir

class ModelConfig:
    dropout: float

    def __init__(self, dropout: float) -> None:
        if dropout <= 0 or dropout > 1:
            raise ValueError(f"'dropout' of {dropout} must be in the range (0, 1]")

        self.dropout = dropout

class TrainingConfig:
    epochs: int
    batch_size: int
    optimizer: OptimizerName
    base_lr: float
    min_lr: float
    warmup_ratio: float
    checkpoint_dir: str
    model_name: str
    device: torch.device
    evaluation_metric: EvaluationMetric
    best_metric_lower_is_better: bool

    def __init__(
        self,
        epochs: int,
        batch_size: int,
        optimizer: str,
        base_lr: float,
        min_lr: float,
        warmup_ratio: float,
        checkpoint_dir: str,
        model_name: str,
        device_name: str,
        evaluation_metric: str,
        best_metric_lower_is_better: bool
    ) -> None:
        if batch_size <= 1:
            raise ValueError(f"`batch_size` must be greater than 1")

        if optimizer not in ValidOptimizerNames:
            raise ValueError(f"'{optimizer}' must be one of {ValidOptimizerNames}")

        if base_lr <= 0:
            raise ValueError(f"'base_lr' of {base_lr} must be a positive float")

        if min_lr <= 0:
            raise ValueError(f"'min_lr' of {min_lr} must be a positive float")

        if base_lr < min_lr:
            raise ValueError(f"'base_lr' of {base_lr} must be greater than 'min_lr' of {min_lr}")

        if warmup_ratio <= 0 or warmup_ratio > 1:
            raise ValueError(f"'warmup_ratio' of {warmup_ratio} must be in the range (0, 1]")

        if not os.path.isdir(checkpoint_dir):
            raise ValueError(f"'checkpoint_dir' of {checkpoint_dir} must be a valid directory path")

        # Carica il device da utilizzare tra CUDA, MPS e CPU
        match device_name:
            case "cuda" if torch.cuda.is_available():
                device = torch.device("cuda")
            case "mps" if torch.backends.mps.is_available():
                device = torch.device("mps")
            case _:
                device = torch.device("cpu")

        if evaluation_metric not in ValidEvaluationMetrics:
            raise ValueError(f"'evaluation_metric' of '{evaluation_metric}' must be one of {ValidEvaluationMetrics}")

        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer # type: ignore
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_ratio = warmup_ratio
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.device = device
        self.evaluation_metric = evaluation_metric # type: ignore
        self.best_metric_lower_is_better = best_metric_lower_is_better

class PlotConfig:
    metrics: EvaluationMetric | list[EvaluationMetric] | None

    def __init__(self, metrics: str | list[str] | None) -> None:
        def validate_metric(metric: str) -> EvaluationMetric:
            if metric not in ValidEvaluationMetrics:
                raise ValueError(f"evaluation metric `{metrics}` must be one of {ValidEvaluationMetrics}")
            return metric # type: ignore

        match metrics:
            case None:
                self.metrics = None
            case str():
                self.metrics = validate_metric(metrics)
            case list():
                if len(metrics) == 1:
                    metric = metrics[0]
                    self.metrics = validate_metric(metric)
                else:
                    self.metrics = []
                    for metric in metrics:
                        metric = validate_metric(metric)
                        self.metrics.append(metric)

class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    plot: PlotConfig

    def __init__(self) -> None:
        T = TypeVar("T")
        def take_item(config: dict[str, Any], attribute_name: str, expected_type: type[T]) -> T:
            try:
                attribute = config[attribute_name]
            except:
                raise AttributeError(f"'{attribute_name}' config attribute not found")

            expected_type_orign: type | None = typing.get_origin(expected_type)
            if not expected_type_orign is None:
                expected_type = expected_type_orign

            attribute_type: type = type(attribute) # type: ignore
            if not attribute_type is expected_type:
                raise TypeError(f"actual type of '{attribute_name}' of '{attribute_type.__name__}' doesn't match expected type of '{expected_type.__name__}'")

            del config[attribute_name]

            return attribute

        def args() -> dict[str, Any]:
            """
            Questa funzione ci permette di sovrascrivere il tipo di ritorno della funzione 'add_arguments'
            """
            return add_arguments() # type: ignore

        config = args()

        data = take_item(config, "data", dict[str, Any])
        train_ratio = take_item(data, "train_ratio", float)
        test_val_ratio = take_item(data, "test_val_ratio", float)
        data_dir = take_item(data, "data_dir", str)

        # Controlliamo che non siano presenti attributi non riconosciuti
        for key, value in data.items():
            raise ValueError(f"Unrecognized configuration attribute {key}: {value}")

        model = take_item(config, "model", dict[str, Any])
        dropout = take_item(model, "dropout", float)

        # Controlliamo che non siano presenti attributi non riconosciuti
        for key, value in model.items():
            raise ValueError(f"Unrecognized configuration attribute {key}: {value}")

        training = take_item(config, "training", dict[str, Any])
        epochs = take_item(training, "epochs", int)
        batch_size = take_item(training, "batch_size", int)
        optimizer = take_item(training, "optimizer", str)
        base_lr = take_item(training, "base_lr", float)
        min_lr = take_item(training, "min_lr", float)
        warmup_ratio = take_item(training, "warmup_ratio", float)
        checkpoint_dir = take_item(training, "checkpoint_dir", str)
        model_name = take_item(training, "model_name", str)
        device = take_item(training, "device", str)
        evaluation_metric = take_item(training, "evaluation_metric", str)
        best_metric_lower_is_better = take_item(training, "best_metric_lower_is_better", bool)

        # Controlliamo che non siano presenti attributi non riconosciuti
        for key, value in training.items():
            raise ValueError(f"Unrecognized configuration attribute {key}: {value}")

        try:
            metrics = take_item(config, "plot", str)
        except AttributeError:
            metrics = None
        except TypeError:
            metrics = take_item(config, "plot", list[str])

        # Controlliamo che non siano presenti attributi non riconosciuti
        for key, value in config.items():
            raise ValueError(f"Unrecognized configuration attribute {key}: {value}")

        self.data = DataConfig(train_ratio, test_val_ratio, data_dir)
        self.model = ModelConfig(dropout)
        self.training = TrainingConfig(
            epochs,
            batch_size,
            optimizer,
            base_lr,
            min_lr,
            warmup_ratio,
            checkpoint_dir,
            model_name,
            device,
            evaluation_metric,
            best_metric_lower_is_better
        )
        self.plot = PlotConfig(metrics)
