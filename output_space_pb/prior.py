"""A-only deterministic prior training and frozen stochastic feature maps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import FeatureMapConfig, PriorConfig
from .data import (
    DatasetTensors,
    IndexedTensorDataset,
    InputTransform,
    imagenet_input_transform,
    split_fit_calibration,
)
from .encoders import FeatureEncoder, ResNet18Encoder, state_dict_sha256


class DeterministicPrior(nn.Module):
    """An encoder and deterministic class-score head, with no stochastic state."""

    def __init__(
        self,
        encoder: FeatureEncoder,
        number_classes: int,
        *,
        zero_head: bool,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.number_classes = int(number_classes)
        self.zero_head = bool(zero_head)
        self.head = None if zero_head else nn.Linear(encoder.feature_dim, number_classes)

    def raw_features(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def base_scores_from_features(self, features: Tensor) -> Tensor:
        if self.zero_head:
            return features.new_zeros((features.shape[0], self.number_classes))
        if self.head is None:  # pragma: no cover - construction prevents this
            raise RuntimeError("missing deterministic score head")
        return self.head(features)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.base_scores_from_features(self.raw_features(inputs))

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("prior freeze failed")


class FrozenFeatureMap(ABC):
    output_dim: int

    @abstractmethod
    def transform(self, raw_features: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def audit(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class StandardizedFeatureMap(FrozenFeatureMap):
    mean: Tensor
    std: Tensor
    kappa: float

    @property
    def output_dim(self) -> int:
        return int(self.mean.numel())

    def transform(self, raw_features: Tensor) -> Tensor:
        values = raw_features.detach().cpu().to(torch.float64)
        if values.ndim != 2 or values.shape[1] != self.output_dim:
            raise ValueError("raw features do not match standardized map")
        return (self.kappa * (values - self.mean) / self.std).contiguous()

    def audit(self) -> dict[str, Any]:
        return {
            "kind": "standardize",
            "output_dimension": self.output_dim,
            "kappa": self.kappa,
            "minimum_A_standard_deviation": float(self.std.min()),
        }


@dataclass(frozen=True)
class PcaWhitenBiasFeatureMap(FrozenFeatureMap):
    mean: Tensor
    directions: Tensor
    inverse_scales: Tensor
    eigenvalues: Tensor
    kappa: float

    @property
    def output_dim(self) -> int:
        return int(self.directions.shape[1] + 1)

    def transform(self, raw_features: Tensor) -> Tensor:
        values = raw_features.detach().cpu().to(torch.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.numel():
            raise ValueError("raw features do not match PCA map")
        whitened = ((values - self.mean) @ self.directions) * self.inverse_scales
        bias = torch.ones((values.shape[0], 1), dtype=torch.float64)
        return (
            self.kappa
            * torch.cat((bias, whitened), dim=1)
            / math.sqrt(self.output_dim)
        ).contiguous()

    def audit(self) -> dict[str, Any]:
        return {
            "kind": "pca_whiten_bias",
            "output_dimension": self.output_dim,
            "pca_components": int(self.directions.shape[1]),
            "kappa": self.kappa,
            "smallest_retained_A_eigenvalue": float(self.eigenvalues[-1]),
        }


@dataclass(frozen=True)
class UpstreamPcaFeatureMap(FrozenFeatureMap):
    mean: Tensor
    directions: Tensor
    inverse_scales: Tensor
    kappa: float
    artifact_sha256: str
    source_model_state_sha256: str

    @property
    def output_dim(self) -> int:
        return int(self.directions.shape[1] + 1)

    def transform(self, raw_features: Tensor) -> Tensor:
        values = raw_features.detach().cpu().to(torch.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.numel():
            raise ValueError("raw features do not match upstream PCA map")
        whitened = ((values - self.mean) @ self.directions) * self.inverse_scales
        bias = torch.ones((values.shape[0], 1), dtype=torch.float64)
        result = self.kappa * torch.cat((bias, whitened), dim=1) / math.sqrt(self.output_dim)
        if not bool(torch.all(torch.isfinite(result))):
            raise FloatingPointError("upstream feature map produced non-finite values")
        return result.contiguous()

    def audit(self) -> dict[str, Any]:
        return {
            "kind": "upstream_pca",
            "output_dimension": self.output_dim,
            "pca_components": int(self.directions.shape[1]),
            "kappa": self.kappa,
            "artifact_sha256": self.artifact_sha256,
            "source_model_state_sha256": self.source_model_state_sha256,
        }


@dataclass(frozen=True)
class PriorTrainingResult:
    selected_epoch: int
    selected_calibration_error: float | None
    final_training_loss: float | None
    state_sha256: str
    checkpoint_source: str | None


@dataclass(frozen=True)
class ExtractedComponents:
    raw_features: Tensor
    base_scores: Tensor
    labels: Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_or_load_prior(
    model: DeterministicPrior,
    data: DatasetTensors,
    prior_indices: Tensor,
    input_transform: InputTransform,
    config: PriorConfig,
    *,
    device: torch.device,
    workers: int,
    seed: int,
    checkpoint: Path | None,
) -> PriorTrainingResult:
    if config.source == "upstream":
        if checkpoint is not None:
            raise ValueError("use --encoder-weights, not --prior-checkpoint, upstream")
        model.freeze()
        return PriorTrainingResult(0, None, None, state_dict_sha256(model.state_dict()), None)
    if prior_indices.numel() == 0:
        raise ValueError("A-trained prior received an empty A block")
    model.to(device)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or "state_dict" not in payload:
            raise ValueError("prior checkpoint must contain state_dict and metadata")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("prior checkpoint metadata must be a mapping")
        expected = {
            "number_classes": model.number_classes,
            "A_index_sha256": tensor_sha256(prior_indices),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"prior checkpoint mismatch for {key}")
        state = payload["state_dict"]
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint state_dict must be a mapping")
        model.load_state_dict(state, strict=True)
        model.freeze()
        return PriorTrainingResult(
            int(metadata.get("selected_epoch", 0)),
            None,
            None,
            state_dict_sha256(model.state_dict()),
            str(checkpoint.resolve()),
        )

    fit_indices, calibration_indices = split_fit_calibration(
        prior_indices, config.calibration_fraction, seed + 1
    )
    train_dataset = IndexedTensorDataset(
        data,
        fit_indices,
        input_transform,
        augment=config.augmentation == "cifar",
        cutout_size=config.cutout_size,
    )
    generator = torch.Generator().manual_seed(seed + 2)
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs, eta_min=1e-5
            )
        )
    else:
        optimizer = torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
        scheduler = None
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_loss = math.inf
    best_error = math.inf
    final_loss: float | None = None
    for epoch in range(1, config.epochs + 1):
        if config.optimizer == "sgd":
            _set_sgd_learning_rate(optimizer, config, epoch)
        model.train()
        total_loss = 0.0
        total = 0
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            scores = model(inputs)
            loss = F.cross_entropy(
                scores,
                labels,
                label_smoothing=config.label_smoothing,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            total += labels.numel()
        final_loss = total_loss / total
        if calibration_indices.numel():
            if epoch % config.validation_every == 0 or epoch == config.epochs:
                calibration_loss, error = classifier_loss_and_error(
                    model,
                    data,
                    calibration_indices,
                    input_transform,
                    batch_size=max(config.batch_size, 256),
                    device=device,
                    workers=workers,
                )
                selection_key = (calibration_loss, error, epoch)
                current_key = (best_loss, best_error, best_epoch or epoch + 1)
                if selection_key < current_key:
                    best_loss = calibration_loss
                    best_error = error
                    best_epoch = epoch
                    best_state = _cpu_state(model.state_dict())
        else:
            best_epoch = epoch
            best_state = _cpu_state(model.state_dict())
        if scheduler is not None:
            scheduler.step()
    if best_state is None:
        raise RuntimeError("prior training selected no state")
    model.load_state_dict(best_state, strict=True)
    model.freeze()
    return PriorTrainingResult(
        best_epoch,
        None if not math.isfinite(best_error) else best_error,
        final_loss,
        state_dict_sha256(model.state_dict()),
        None,
    )


def _set_sgd_learning_rate(
    optimizer: torch.optim.Optimizer,
    config: PriorConfig,
    epoch: int,
) -> None:
    if config.warmup_epochs and epoch <= config.warmup_epochs:
        factor = epoch / config.warmup_epochs
    else:
        denominator = max(1, config.epochs - config.warmup_epochs)
        progress = (epoch - config.warmup_epochs) / denominator
        factor = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate * factor


@torch.inference_mode()
def extract_components(
    model: DeterministicPrior,
    data: DatasetTensors,
    indices: Tensor,
    input_transform: InputTransform,
    *,
    batch_size: int,
    device: torch.device,
    workers: int,
) -> ExtractedComponents:
    dataset = IndexedTensorDataset(data, indices, input_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    was_training = model.training
    model.eval()
    raw_parts: list[Tensor] = []
    score_parts: list[Tensor] = []
    label_parts: list[Tensor] = []
    try:
        for inputs, labels in loader:
            raw = model.raw_features(inputs.to(device, non_blocking=True))
            scores = model.base_scores_from_features(raw)
            raw_parts.append(raw.cpu().to(torch.float64))
            score_parts.append(scores.cpu().to(torch.float64))
            label_parts.append(labels.cpu().to(torch.long))
    finally:
        model.train(was_training)
    if not raw_parts:
        width = model.encoder.feature_dim
        return ExtractedComponents(
            torch.empty((0, width), dtype=torch.float64),
            torch.empty((0, model.number_classes), dtype=torch.float64),
            torch.empty(0, dtype=torch.long),
        )
    return ExtractedComponents(
        torch.cat(raw_parts),
        torch.cat(score_parts),
        torch.cat(label_parts),
    )


def fit_feature_map(
    raw_A: Tensor,
    config: FeatureMapConfig,
) -> FrozenFeatureMap:
    values = raw_A.detach().cpu().to(torch.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("feature-map fitting requires at least two A examples")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("A features contain non-finite values")
    if config.kind == "standardize":
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False)
        if bool(torch.any(std <= 0.0)):
            raise RuntimeError("A feature standardizer found a zero direction")
        return StandardizedFeatureMap(mean, std.clamp_min(1e-4), config.kappa)
    if config.kind == "pca_whiten_bias":
        mean = values.mean(dim=0)
        centered = values - mean
        covariance = centered.T @ centered / values.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)[: config.rank - 1]
        retained = eigenvalues.index_select(0, order)
        if not bool(torch.all(torch.isfinite(retained))) or float(retained[-1]) <= 0.0:
            raise RuntimeError("requested A-only PCA rank contains a zero direction")
        directions = _canonicalize_eigenvector_signs(eigenvectors.index_select(1, order))
        return PcaWhitenBiasFeatureMap(
            mean,
            directions,
            torch.rsqrt(retained),
            retained,
            config.kappa,
        )
    raise ValueError("upstream_pca maps must be loaded from an artifact")


def load_upstream_feature_map(
    path: Path,
    config: FeatureMapConfig,
) -> tuple[UpstreamPcaFeatureMap, InputTransform, dict[str, Any]]:
    artifact_hash = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("upstream artifact must be a mapping")
    required = {
        "artifact_type",
        "backbone",
        "feature_dimension",
        "labels_read",
        "model_state_sha256",
        "preprocessing",
        "tensors",
    }
    if not required.issubset(payload):
        raise ValueError("upstream artifact is missing required fields")
    if payload["artifact_type"] != "imagenet_penultimate_pca_whitening":
        raise ValueError("unexpected upstream artifact type")
    if payload["backbone"] != "resnet18" or int(payload["feature_dimension"]) != 512:
        raise ValueError("upstream artifact is not the expected ResNet-18 feature map")
    if bool(payload["labels_read"]):
        raise RuntimeError("upstream feature statistics must not read labels")
    tensors = payload["tensors"]
    if not isinstance(tensors, Mapping):
        raise ValueError("upstream tensor payload must be a mapping")
    tensor_hashes = payload.get("tensor_hashes", {})
    if not isinstance(tensor_hashes, Mapping):
        raise ValueError("upstream tensor hashes must be a mapping")
    for name in ("feature_mean", "top_eigenvectors", "top_inverse_sqrt_eigenvalues"):
        value = tensors.get(name)
        if not isinstance(value, Tensor):
            raise ValueError(f"upstream tensor {name} is missing")
        expected_hash = tensor_hashes.get(name)
        if expected_hash is not None and expected_hash != tensor_sha256(value):
            raise RuntimeError(f"upstream tensor hash mismatch: {name}")
    components = config.rank - 1
    mean = tensors["feature_mean"].to(torch.float64)
    directions = tensors["top_eigenvectors"][:, :components].to(torch.float64)
    inverse = tensors["top_inverse_sqrt_eigenvalues"][:components].to(torch.float64)
    if directions.shape != (512, components) or inverse.shape != (components,):
        raise ValueError("upstream artifact lacks the requested PCA width")
    artifact_kappa = float(payload.get("kappa", 1.0))
    if not math.isclose(artifact_kappa, config.kappa, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("configured kappa differs from the upstream artifact")
    feature_map = UpstreamPcaFeatureMap(
        mean,
        directions,
        inverse,
        config.kappa,
        artifact_hash,
        str(payload["model_state_sha256"]),
    )
    transform = imagenet_input_transform(payload["preprocessing"])
    audit = {
        "artifact_sha256": artifact_hash,
        "image_count": int(payload.get("image_count", 0)),
        "labels_read": bool(payload["labels_read"]),
        "source_model_state_sha256": str(payload["model_state_sha256"]),
    }
    return feature_map, transform, audit


def validate_upstream_encoder(
    encoder: FeatureEncoder,
    feature_map: UpstreamPcaFeatureMap,
) -> None:
    if not isinstance(encoder, ResNet18Encoder):
        raise ValueError("upstream PCA currently requires the registered ResNet-18")
    if encoder.source_state_sha256 != feature_map.source_model_state_sha256:
        raise RuntimeError("encoder weights differ from those defining upstream PCA")


@torch.inference_mode()
def deterministic_error(
    model: DeterministicPrior,
    data: DatasetTensors,
    indices: Tensor,
    input_transform: InputTransform,
    *,
    batch_size: int,
    device: torch.device,
    workers: int,
) -> float:
    _, error = classifier_loss_and_error(
        model,
        data,
        indices,
        input_transform,
        batch_size=batch_size,
        device=device,
        workers=workers,
    )
    return error


@torch.inference_mode()
def classifier_loss_and_error(
    model: DeterministicPrior,
    data: DatasetTensors,
    indices: Tensor,
    input_transform: InputTransform,
    *,
    batch_size: int,
    device: torch.device,
    workers: int,
) -> tuple[float, float]:
    extracted = extract_components(
        model,
        data,
        indices,
        input_transform,
        batch_size=batch_size,
        device=device,
        workers=workers,
    )
    if extracted.labels.numel() == 0:
        raise ValueError("classifier evaluation needs a nonempty block")
    loss = float(F.cross_entropy(extracted.base_scores, extracted.labels))
    error = float(
        (extracted.base_scores.argmax(1) != extracted.labels).to(torch.float64).mean()
    )
    return loss, error


def _cpu_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _canonicalize_eigenvector_signs(vectors: Tensor) -> Tensor:
    result = vectors.clone()
    pivot_rows = result.abs().argmax(dim=0)
    columns = torch.arange(result.shape[1])
    signs = torch.sign(result[pivot_rows, columns])
    signs[signs == 0] = 1
    return result * signs.unsqueeze(0)


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "DeterministicPrior",
    "ExtractedComponents",
    "FrozenFeatureMap",
    "PriorTrainingResult",
    "UpstreamPcaFeatureMap",
    "deterministic_error",
    "extract_components",
    "fit_feature_map",
    "load_upstream_feature_map",
    "seed_everything",
    "tensor_sha256",
    "train_or_load_prior",
    "validate_upstream_encoder",
]
