"""Registered deterministic feature encoders, independent of the stochastic model."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import EncoderConfig


class FeatureEncoder(nn.Module, ABC):
    """Backbone contract used by deterministic priors."""

    feature_dim: int

    @abstractmethod
    def forward_features(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, inputs: Tensor) -> Tensor:
        values = self.forward_features(inputs)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise RuntimeError("encoder returned an invalid feature shape")
        return values


EncoderFactory = Callable[[EncoderConfig, int | None, Path | None], FeatureEncoder]
_ENCODERS: dict[str, EncoderFactory] = {}


def register_encoder(name: str) -> Callable[[EncoderFactory], EncoderFactory]:
    if not name or name in _ENCODERS:
        raise ValueError(f"invalid or duplicate encoder name: {name}")

    def decorator(factory: EncoderFactory) -> EncoderFactory:
        _ENCODERS[name] = factory
        return factory

    return decorator


def available_encoders() -> tuple[str, ...]:
    return tuple(sorted(_ENCODERS))


def build_encoder(
    config: EncoderConfig,
    *,
    input_dimension: int | None = None,
    weights_path: Path | None = None,
) -> FeatureEncoder:
    try:
        encoder = _ENCODERS[config.name](config, input_dimension, weights_path)
    except KeyError as error:
        raise ValueError(
            f"unknown encoder {config.name!r}; available={available_encoders()}"
        ) from error
    if encoder.feature_dim != config.feature_dimension:
        raise RuntimeError("registered encoder feature dimension differs from config")
    return encoder


class MlpEncoder(FeatureEncoder):
    def __init__(self, input_dimension: int, feature_dimension: int, width: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dimension)
        self.network = nn.Sequential(
            nn.Linear(input_dimension, width),
            nn.ReLU(),
            nn.Linear(width, feature_dimension),
            nn.Tanh(),
        )

    def forward_features(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2:
            raise ValueError("MLP inputs must have shape [batch, dimension]")
        return self.network(inputs)


@register_encoder("mlp")
def _mlp_factory(
    config: EncoderConfig,
    input_dimension: int | None,
    weights_path: Path | None,
) -> FeatureEncoder:
    del weights_path
    if input_dimension is None:
        raise ValueError("MLP encoder requires input_dimension")
    return MlpEncoder(input_dimension, config.feature_dimension, config.width)


class MnistCnnEncoder(FeatureEncoder):
    def __init__(self, feature_dimension: int, width: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dimension)
        self.network = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(2 * width * 7 * 7, feature_dimension),
            nn.Tanh(),
        )

    def forward_features(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (1, 28, 28):
            raise ValueError("MNIST CNN inputs must have shape [batch,1,28,28]")
        return self.network(inputs)


@register_encoder("mnist_cnn")
def _mnist_cnn_factory(
    config: EncoderConfig,
    input_dimension: int | None,
    weights_path: Path | None,
) -> FeatureEncoder:
    del input_dimension, weights_path
    return MnistCnnEncoder(config.feature_dimension, config.width)


class LeNet5Encoder(FeatureEncoder):
    feature_dim = 84

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, self.feature_dim)

    def forward_features(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (1, 28, 28):
            raise ValueError("LeNet-5 inputs must have shape [batch,1,28,28]")
        values = F.max_pool2d(F.relu(self.conv1(inputs)), 2)
        values = F.max_pool2d(F.relu(self.conv2(values)), 2)
        values = F.relu(self.fc1(values.flatten(1)))
        return F.relu(self.fc2(values))


@register_encoder("lenet5")
def _lenet_factory(
    config: EncoderConfig,
    input_dimension: int | None,
    weights_path: Path | None,
) -> FeatureEncoder:
    del input_dimension, weights_path
    if config.feature_dimension != LeNet5Encoder.feature_dim:
        raise ValueError("LeNet-5 exposes exactly 84 features")
    return LeNet5Encoder()


class _WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.dropout = float(dropout)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        values = F.relu(self.bn1(inputs), inplace=False)
        shortcut_input = inputs if isinstance(self.shortcut, nn.Identity) else values
        shortcut = self.shortcut(shortcut_input)
        values = self.conv1(values)
        values = F.relu(self.bn2(values), inplace=False)
        if self.dropout:
            values = F.dropout(values, p=self.dropout, training=self.training)
        return shortcut + self.conv2(values)


class WideResNet28x4Encoder(FeatureEncoder):
    """WRN-28-4 with the learned bottleneck used by the paper presets."""

    def __init__(self, feature_dimension: int, dropout: float) -> None:
        super().__init__()
        self.feature_dim = int(feature_dimension)
        channels = (16, 64, 128, 256)
        self.stem = nn.Conv2d(3, channels[0], 3, padding=1, bias=False)
        self.group1 = self._group(4, channels[0], channels[1], 1, dropout)
        self.group2 = self._group(4, channels[1], channels[2], 2, dropout)
        self.group3 = self._group(4, channels[2], channels[3], 2, dropout)
        self.final_bn = nn.BatchNorm2d(channels[3])
        self.projection = nn.Linear(channels[3], feature_dimension)
        self.layer_norm = nn.LayerNorm(feature_dimension)
        self._initialize()

    @staticmethod
    def _group(
        blocks: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        for index in range(blocks):
            layers.append(
                _WideBasicBlock(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    stride if index == 0 else 1,
                    dropout,
                )
            )
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (3, 32, 32):
            raise ValueError("WRN inputs must have shape [batch,3,32,32]")
        values = self.group3(self.group2(self.group1(self.stem(inputs))))
        values = F.relu(self.final_bn(values), inplace=False)
        values = F.adaptive_avg_pool2d(values, 1).flatten(1)
        return torch.tanh(self.layer_norm(self.projection(values)))


@register_encoder("wrn28_4")
def _wrn_factory(
    config: EncoderConfig,
    input_dimension: int | None,
    weights_path: Path | None,
) -> FeatureEncoder:
    del input_dimension, weights_path
    return WideResNet28x4Encoder(config.feature_dimension, config.dropout)


class ResNet18Encoder(FeatureEncoder):
    feature_dim = 512

    def __init__(self, weights_path: Path | None) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        if weights_path is None:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)
        else:
            model = resnet18(weights=None)
            payload = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(payload, Mapping) and "state_dict" in payload:
                payload = payload["state_dict"]
            if not isinstance(payload, Mapping):
                raise ValueError("ResNet checkpoint must contain a state dictionary")
            state = {
                str(key).removeprefix("module."): value
                for key, value in payload.items()
                if isinstance(value, Tensor)
            }
            model.load_state_dict(state, strict=True)
        self.source_state_sha256 = state_dict_sha256(model.state_dict())
        model.fc = nn.Identity()
        self.model = model
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward_features(self, inputs: Tensor) -> Tensor:
        values = self.model(inputs)
        if values.shape[1] != self.feature_dim:
            raise RuntimeError("ResNet-18 returned an unexpected feature dimension")
        return values


@register_encoder("imagenet_resnet18")
def _resnet18_factory(
    config: EncoderConfig,
    input_dimension: int | None,
    weights_path: Path | None,
) -> FeatureEncoder:
    del input_dimension
    if config.feature_dimension != ResNet18Encoder.feature_dim:
        raise ValueError("ImageNet ResNet-18 exposes exactly 512 features")
    return ResNet18Encoder(weights_path)


def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        encoded_key = key.encode("utf-8")
        digest.update(len(encoded_key).to_bytes(8, "big"))
        digest.update(encoded_key)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "FeatureEncoder",
    "ResNet18Encoder",
    "available_encoders",
    "build_encoder",
    "state_dict_sha256",
]
