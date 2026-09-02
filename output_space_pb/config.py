"""Strict typed configuration for the single experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Mapping, TypeVar


T = TypeVar("T")


def _strict_values(cls: type[T], values: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")
    return dict(values)


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    prior_fraction: float
    split_seed: int
    number_classes: int
    synthetic_train_size: int = 128
    synthetic_test_size: int = 64
    synthetic_input_dimension: int = 12

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DatasetConfig":
        return cls(**_strict_values(cls, values))

    def validate(self) -> None:
        if self.name not in {"synthetic", "mnist", "cifar10", "cifar100"}:
            raise ValueError(f"unsupported dataset: {self.name}")
        if not 0.0 <= self.prior_fraction < 1.0:
            raise ValueError("prior_fraction must lie in [0, 1)")
        if self.split_seed < 0 or self.number_classes < 2:
            raise ValueError("split_seed must be nonnegative and classes at least two")
        if min(
            self.synthetic_train_size,
            self.synthetic_test_size,
            self.synthetic_input_dimension,
        ) <= 0:
            raise ValueError("synthetic dimensions must be positive")


@dataclass(frozen=True)
class EncoderConfig:
    name: str
    feature_dimension: int
    width: int = 32
    dropout: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EncoderConfig":
        return cls(**_strict_values(cls, values))

    def validate(self) -> None:
        if self.feature_dimension <= 0 or self.width <= 0:
            raise ValueError("encoder dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")


@dataclass(frozen=True)
class PriorConfig:
    source: str
    optimizer: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    momentum: float = 0.9
    warmup_epochs: int = 0
    calibration_fraction: float = 0.0
    label_smoothing: float = 0.0
    augmentation: str = "none"
    cutout_size: int = 0
    validation_every: int = 1

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PriorConfig":
        return cls(**_strict_values(cls, values))

    def validate(self) -> None:
        if self.source not in {"a_trained", "upstream"}:
            raise ValueError("prior source must be a_trained or upstream")
        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("prior optimizer must be adamw or sgd")
        if self.source == "a_trained" and min(
            self.epochs, self.batch_size, self.validation_every
        ) <= 0:
            raise ValueError("A-trained priors need positive epochs and batch size")
        if self.source == "upstream" and self.epochs != 0:
            raise ValueError("upstream priors must use epochs=0")
        _positive(self.learning_rate, "prior learning_rate")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("prior weight_decay must be finite and nonnegative")
        if not 0.0 <= self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must lie in [0, 0.5)")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must lie in [0, 1)")
        if self.augmentation not in {"none", "cifar"}:
            raise ValueError("augmentation must be none or cifar")
        if not 0 <= self.cutout_size <= 32:
            raise ValueError("cutout_size must lie in [0, 32]")


@dataclass(frozen=True)
class FeatureMapConfig:
    kind: str
    rank: int
    kappa: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FeatureMapConfig":
        return cls(**_strict_values(cls, values))

    def validate(self, encoder_dimension: int) -> None:
        if self.kind not in {"standardize", "pca_whiten_bias", "upstream_pca"}:
            raise ValueError(f"unsupported feature map: {self.kind}")
        if self.rank <= 0:
            raise ValueError("feature rank must be positive")
        _positive(self.kappa, "feature kappa")
        if self.kind == "standardize" and self.rank != encoder_dimension:
            raise ValueError("standardized feature rank must equal encoder dimension")
        if self.kind == "pca_whiten_bias" and not 2 <= self.rank <= encoder_dimension + 1:
            raise ValueError("PCA feature rank must include one bias and valid PCs")


@dataclass(frozen=True)
class PosteriorConfig:
    steps: int
    checkpoint_every: int
    batch_size: int
    learning_rates: tuple[float, ...]
    objectives: tuple[str, ...]
    warmup_steps: int
    training_quadrature_order: int
    selection_quadrature_order: int
    selection_chunk_size: int
    seed: int
    gradient_clip_norm: float = 10.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PosteriorConfig":
        normalized = _strict_values(cls, values)
        normalized["learning_rates"] = tuple(float(v) for v in normalized["learning_rates"])
        normalized["objectives"] = tuple(str(v) for v in normalized["objectives"])
        return cls(**normalized)

    def validate(self) -> None:
        integers = (
            self.steps,
            self.checkpoint_every,
            self.batch_size,
            self.training_quadrature_order,
            self.selection_quadrature_order,
            self.selection_chunk_size,
        )
        if min(integers) <= 0 or self.warmup_steps < 0:
            raise ValueError("posterior counts must be positive and warmup nonnegative")
        if not self.learning_rates:
            raise ValueError("at least one posterior learning rate is required")
        for value in self.learning_rates:
            _positive(value, "posterior learning rate")
        if not self.objectives or any(
            value not in {"classic", "quad", "exact"} for value in self.objectives
        ):
            raise ValueError("posterior objectives must be classic, quad, or exact")
        if len(set(self.objectives)) != len(self.objectives):
            raise ValueError("posterior objectives must be unique")
        if self.seed < 0:
            raise ValueError("posterior seed must be nonnegative")
        _positive(self.gradient_clip_norm, "gradient_clip_norm")


@dataclass(frozen=True)
class ConfidenceConfig:
    total_failure_probability: float
    pac_bayes_delta_each: float
    pac_bayes_family_count: int
    monte_carlo_delta: float
    direct_holdout_delta: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ConfidenceConfig":
        return cls(**_strict_values(cls, values))

    def validate(self) -> None:
        for value, name in (
            (self.total_failure_probability, "total failure probability"),
            (self.pac_bayes_delta_each, "PAC-Bayes delta"),
            (self.monte_carlo_delta, "Monte Carlo delta"),
            (self.direct_holdout_delta, "direct holdout delta"),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
        if self.pac_bayes_family_count <= 0:
            raise ValueError("pac_bayes_family_count must be positive")
        allocated = (
            self.pac_bayes_delta_each * self.pac_bayes_family_count
            + self.monte_carlo_delta
        )
        if allocated > self.total_failure_probability + 1e-15:
            raise ValueError("PAC-Bayes family plus MC allocation exceeds total failure")


@dataclass(frozen=True)
class CertificationConfig:
    monte_carlo_trials: int
    monte_carlo_chunk_size: int
    monte_carlo_seed: int
    diagnostic_quadrature_order: int
    evaluate_test: bool
    direct_holdout: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CertificationConfig":
        return cls(**_strict_values(cls, values))

    def validate(self) -> None:
        if min(
            self.monte_carlo_trials,
            self.monte_carlo_chunk_size,
            self.diagnostic_quadrature_order,
        ) <= 0:
            raise ValueError("certification counts must be positive")
        if self.monte_carlo_seed < 0:
            raise ValueError("Monte Carlo seed must be nonnegative")


@dataclass(frozen=True)
class NumericsConfig:
    minimum_posterior_std: float
    minimum_relative_singular_value: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "NumericsConfig":
        return cls(**_strict_values(cls, values))

    def validate(self) -> None:
        if not 0.0 < self.minimum_posterior_std < 1.0:
            raise ValueError("minimum_posterior_std must lie in (0, 1)")
        if not 0.0 <= self.minimum_relative_singular_value < 1.0:
            raise ValueError("minimum_relative_singular_value must lie in [0, 1)")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    dataset: DatasetConfig
    encoder: EncoderConfig
    prior: PriorConfig
    feature_map: FeatureMapConfig
    posterior: PosteriorConfig
    confidence: ConfidenceConfig
    certification: CertificationConfig
    numerics: NumericsConfig

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        normalized = _strict_values(cls, values)
        for name, nested in (
            ("dataset", DatasetConfig),
            ("encoder", EncoderConfig),
            ("prior", PriorConfig),
            ("feature_map", FeatureMapConfig),
            ("posterior", PosteriorConfig),
            ("confidence", ConfidenceConfig),
            ("certification", CertificationConfig),
            ("numerics", NumericsConfig),
        ):
            normalized[name] = nested.from_mapping(normalized[name])
        config = cls(**normalized)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.name or self.seed < 0:
            raise ValueError("experiment name is required and seed must be nonnegative")
        self.dataset.validate()
        self.encoder.validate()
        self.prior.validate()
        self.feature_map.validate(self.encoder.feature_dimension)
        self.posterior.validate()
        self.confidence.validate()
        self.certification.validate()
        self.numerics.validate()
        if self.prior.source == "upstream":
            if self.dataset.prior_fraction != 0.0:
                raise ValueError("upstream transfer requires an empty A block")
            if self.feature_map.kind != "upstream_pca":
                raise ValueError("upstream transfer requires upstream_pca features")
        elif self.dataset.prior_fraction <= 0.0:
            raise ValueError("A-trained prior requires a nonempty A block")
        if self.certification.monte_carlo_seed in {
            self.seed,
            self.dataset.split_seed,
            self.posterior.seed,
        }:
            raise ValueError("final Monte Carlo seed must be separate from training seeds")


def load_config(path: Path | str) -> ExperimentConfig:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("configuration root must be a JSON object")
    return ExperimentConfig.from_mapping(payload)


def smoke_config() -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "name": "synthetic-smoke",
            "seed": 41,
            "dataset": {
                "name": "synthetic",
                "prior_fraction": 0.75,
                "split_seed": 43,
                "number_classes": 4,
                "synthetic_train_size": 128,
                "synthetic_test_size": 64,
                "synthetic_input_dimension": 12,
            },
            "encoder": {
                "name": "mlp",
                "feature_dimension": 8,
                "width": 16,
                "dropout": 0.0,
            },
            "prior": {
                "source": "a_trained",
                "optimizer": "adamw",
                "epochs": 2,
                "batch_size": 32,
                "learning_rate": 0.01,
                "weight_decay": 0.0001,
            },
            "feature_map": {"kind": "standardize", "rank": 8, "kappa": 0.1},
            "posterior": {
                "steps": 3,
                "checkpoint_every": 1,
                "batch_size": 32,
                "learning_rates": [0.01],
                "objectives": ["exact"],
                "warmup_steps": 1,
                "training_quadrature_order": 8,
                "selection_quadrature_order": 12,
                "selection_chunk_size": 32,
                "seed": 47,
            },
            "confidence": {
                "total_failure_probability": 0.05,
                "pac_bayes_delta_each": 0.045,
                "pac_bayes_family_count": 1,
                "monte_carlo_delta": 0.005,
                "direct_holdout_delta": 0.05,
            },
            "certification": {
                "monte_carlo_trials": 4000,
                "monte_carlo_chunk_size": 1000,
                "monte_carlo_seed": 100047,
                "diagnostic_quadrature_order": 16,
                "evaluate_test": True,
                "direct_holdout": True,
            },
            "numerics": {
                "minimum_posterior_std": 0.0001,
                "minimum_relative_singular_value": 1e-12,
            },
        }
    )


__all__ = ["ExperimentConfig", "load_config", "smoke_config"]
