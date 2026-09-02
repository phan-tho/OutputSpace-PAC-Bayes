"""Dataset loading, observation-independent splits, and frozen input transforms."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from .config import DatasetConfig


@dataclass(frozen=True)
class DatasetTensors:
    images: Tensor
    labels: Tensor
    number_classes: int
    name: str

    def validate(self) -> None:
        if self.images.shape[0] != self.labels.numel():
            raise ValueError("image and label counts differ")
        if self.labels.dtype != torch.long or self.labels.ndim != 1:
            raise ValueError("labels must be one-dimensional torch.long")
        if self.labels.numel() == 0:
            raise ValueError("dataset is empty")
        if bool(torch.any((self.labels < 0) | (self.labels >= self.number_classes))):
            raise ValueError("labels lie outside the declared class range")


@dataclass(frozen=True)
class SplitIndices:
    prior: Tensor
    posterior: Tensor

    def validate(self, number_examples: int) -> None:
        if self.prior.dtype != torch.long or self.posterior.dtype != torch.long:
            raise ValueError("split indices must be torch.long")
        joined = torch.cat((self.prior, self.posterior))
        if joined.numel() != number_examples:
            raise RuntimeError("A/B split does not cover the training sample")
        if torch.unique(joined).numel() != number_examples:
            raise RuntimeError("A/B split overlaps or repeats an index")
        if joined.numel() and (
            int(joined.min()) != 0 or int(joined.max()) != number_examples - 1
        ):
            raise RuntimeError("A/B split contains an out-of-range index")


@dataclass(frozen=True)
class InputTransform:
    kind: str
    mean: tuple[float, ...] = ()
    std: tuple[float, ...] = ()
    resize_size: int = 256
    crop_size: int = 224

    def transform(self, image: Tensor) -> Tensor:
        if self.kind == "identity":
            return image.to(torch.float32)
        if image.ndim != 3 or image.dtype != torch.uint8:
            raise ValueError("image transforms expect one uint8 CHW image")
        value = image.to(torch.float32).div(255.0)
        if self.kind == "channel_standardize":
            mean = value.new_tensor(self.mean)[:, None, None]
            std = value.new_tensor(self.std)[:, None, None]
            return (value - mean) / std
        if self.kind == "imagenet_v1":
            from PIL import Image
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms import functional as TF

            array = image.permute(1, 2, 0).contiguous().numpy()
            pil_image = Image.fromarray(array, mode="RGB")
            resized = TF.resize(
                pil_image,
                [self.resize_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            cropped = TF.center_crop(resized, [self.crop_size])
            value = TF.pil_to_tensor(cropped).to(torch.float32).div(255.0)
            mean = value.new_tensor(self.mean)[:, None, None]
            std = value.new_tensor(self.std)[:, None, None]
            return (value - mean) / std
        raise ValueError(f"unknown input transform: {self.kind}")


class IndexedTensorDataset(Dataset[tuple[Tensor, Tensor]]):
    """An indexed tensor view with optional A-only CIFAR augmentation."""

    def __init__(
        self,
        data: DatasetTensors,
        indices: Tensor,
        transform: InputTransform,
        *,
        augment: bool = False,
        cutout_size: int = 0,
    ) -> None:
        self.data = data
        self.indices = indices.detach().cpu().to(torch.long)
        self.transform_spec = transform
        self.augment = bool(augment)
        self.cutout_size = int(cutout_size)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, position: int) -> tuple[Tensor, Tensor]:
        index = int(self.indices[position])
        image = self.data.images[index]
        if self.augment:
            if image.ndim != 3 or tuple(image.shape[-2:]) != (32, 32):
                raise ValueError("CIFAR augmentation requires 32x32 CHW images")
            value = image.to(torch.float32).div(255.0)
            padded = F.pad(value, (4, 4, 4, 4), mode="reflect")
            top = int(torch.randint(0, 9, ()).item())
            left = int(torch.randint(0, 9, ()).item())
            value = padded[:, top : top + 32, left : left + 32]
            if bool(torch.rand(()) < 0.5):
                value = torch.flip(value, dims=(2,))
            mean = value.new_tensor(self.transform_spec.mean)[:, None, None]
            std = value.new_tensor(self.transform_spec.std)[:, None, None]
            value = (value - mean) / std
            if self.cutout_size:
                center_y = int(torch.randint(0, 32, ()).item())
                center_x = int(torch.randint(0, 32, ()).item())
                half = self.cutout_size // 2
                y0, y1 = max(0, center_y - half), min(32, center_y + half)
                x0, x1 = max(0, center_x - half), min(32, center_x + half)
                value[:, y0:y1, x0:x1] = 0.0
            transformed = value
        else:
            transformed = self.transform_spec.transform(image)
        return transformed, self.data.labels[index]


def make_ab_split(
    number_examples: int,
    prior_fraction: float,
    seed: int,
) -> SplitIndices:
    """Split using only sample count and seed; observations are not accepted."""

    if number_examples <= 0 or not 0.0 <= prior_fraction < 1.0 or seed < 0:
        raise ValueError("invalid split request")
    if prior_fraction == 0.0:
        result = SplitIndices(
            prior=torch.empty(0, dtype=torch.long),
            posterior=torch.arange(number_examples, dtype=torch.long),
        )
    else:
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(number_examples, generator=generator)
        prior_count = int(round(prior_fraction * number_examples))
        if not 0 < prior_count < number_examples:
            raise ValueError("split produced an empty A or B block")
        result = SplitIndices(permutation[:prior_count], permutation[prior_count:])
    result.validate(number_examples)
    return result


def split_fit_calibration(
    prior_indices: Tensor,
    calibration_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Create an index-only internal A split for checkpoint selection."""

    if not 0.0 <= calibration_fraction < 0.5:
        raise ValueError("invalid calibration fraction")
    if calibration_fraction == 0.0:
        return prior_indices.clone(), torch.empty(0, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(prior_indices.numel(), generator=generator)
    calibration_count = max(1, int(round(calibration_fraction * prior_indices.numel())))
    calibration = prior_indices.index_select(0, order[:calibration_count])
    fit = prior_indices.index_select(0, order[calibration_count:])
    if fit.numel() == 0:
        raise ValueError("A calibration split left no fitting examples")
    return fit, calibration


def fit_input_transform(data: DatasetTensors, prior_indices: Tensor) -> InputTransform:
    if data.name == "synthetic":
        return InputTransform(kind="identity")
    if prior_indices.numel() == 0:
        raise ValueError("A-only normalization needs a nonempty A block")
    selected = data.images.index_select(0, prior_indices).to(torch.float64).div(255.0)
    if selected.ndim != 4:
        raise ValueError("image data must have shape [n,c,h,w]")
    mean = selected.mean(dim=(0, 2, 3))
    std = selected.std(dim=(0, 2, 3), unbiased=False)
    if bool(torch.any(std <= 0.0)):
        raise RuntimeError("A-only input normalization found a zero-variance channel")
    return InputTransform(
        kind="channel_standardize",
        mean=tuple(float(v) for v in mean),
        std=tuple(float(v) for v in std),
    )


def imagenet_input_transform(preprocessing: Mapping[str, Any]) -> InputTransform:
    required = {"mean", "std", "resize_size", "crop_size"}
    if not required.issubset(preprocessing):
        raise ValueError("upstream artifact lacks preprocessing fields")
    if "BILINEAR" not in str(preprocessing.get("interpolation", "")):
        raise ValueError("only the audited bilinear upstream transform is supported")
    if not bool(preprocessing.get("antialias", False)):
        raise ValueError("upstream preprocessing must enable antialiasing")
    def size(value: Any) -> int:
        if isinstance(value, (list, tuple)):
            if not value or any(int(item) != int(value[0]) for item in value):
                raise ValueError("only square upstream preprocessing is supported")
            return int(value[0])
        return int(value)

    return InputTransform(
        kind="imagenet_v1",
        mean=tuple(float(v) for v in preprocessing["mean"]),
        std=tuple(float(v) for v in preprocessing["std"]),
        resize_size=size(preprocessing["resize_size"]),
        crop_size=size(preprocessing["crop_size"]),
    )


def load_training_data(
    config: DatasetConfig,
    root: Path,
    *,
    download: bool,
    seed: int,
) -> DatasetTensors:
    return _load_data(config, root, train=True, download=download, seed=seed)


def load_test_data(
    config: DatasetConfig,
    root: Path,
    *,
    download: bool,
    seed: int,
) -> DatasetTensors:
    """Load test data separately so the pipeline can defer all test access."""

    return _load_data(config, root, train=False, download=download, seed=seed)


def _load_data(
    config: DatasetConfig,
    root: Path,
    *,
    train: bool,
    download: bool,
    seed: int,
) -> DatasetTensors:
    if config.name == "synthetic":
        result = _synthetic_data(config, train=train, seed=seed)
    elif config.name == "mnist":
        result = _torchvision_mnist(root, train=train, download=download)
    elif config.name in {"cifar10", "cifar100"}:
        result = _cifar_data(config.name, root, train=train, download=download)
    else:  # pragma: no cover - validated configuration prevents this
        raise ValueError(f"unsupported dataset: {config.name}")
    result.validate()
    if result.number_classes != config.number_classes:
        raise RuntimeError("loaded dataset class count differs from configuration")
    return result


def _synthetic_data(config: DatasetConfig, *, train: bool, seed: int) -> DatasetTensors:
    count = config.synthetic_train_size if train else config.synthetic_test_size
    data_seed = seed + (0 if train else 10_000)
    generator = torch.Generator().manual_seed(data_seed)
    images = torch.randn(
        count,
        config.synthetic_input_dimension,
        generator=generator,
        dtype=torch.float32,
    )
    weight_generator = torch.Generator().manual_seed(seed + 99)
    weights = torch.randn(
        config.synthetic_input_dimension,
        config.number_classes,
        generator=weight_generator,
    )
    logits = images @ weights + 0.15 * torch.randn(
        count, config.number_classes, generator=generator
    )
    labels = logits.argmax(dim=1).to(torch.long)
    return DatasetTensors(images, labels, config.number_classes, "synthetic")


def _torchvision_mnist(root: Path, *, train: bool, download: bool) -> DatasetTensors:
    from torchvision.datasets import MNIST

    dataset = MNIST(root=str(root), train=train, download=download)
    images = dataset.data.unsqueeze(1).contiguous()
    labels = dataset.targets.to(torch.long).contiguous()
    return DatasetTensors(images, labels, 10, "mnist")


def _cifar_data(
    name: str,
    root: Path,
    *,
    train: bool,
    download: bool,
) -> DatasetTensors:
    try:
        directory = _discover_cifar_directory(name, root)
    except FileNotFoundError:
        if not download:
            raise
        return _torchvision_cifar(name, root, train=train, download=True)
    return _load_cifar_python(name, directory, train=train)


def _torchvision_cifar(
    name: str,
    root: Path,
    *,
    train: bool,
    download: bool,
) -> DatasetTensors:
    from torchvision.datasets import CIFAR10, CIFAR100

    cls = CIFAR10 if name == "cifar10" else CIFAR100
    dataset = cls(root=str(root), train=train, download=download)
    images = torch.from_numpy(np.asarray(dataset.data)).permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(dataset.targets, dtype=torch.long)
    return DatasetTensors(images, labels, 10 if name == "cifar10" else 100, name)


def _discover_cifar_directory(name: str, root: Path) -> Path:
    filenames = (
        {"data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"}
        if name == "cifar10"
        else {"train", "test", "meta"}
    )
    candidates: list[Path] = []
    for directory in (root, *[path for path in root.rglob("*") if path.is_dir()]):
        if all((directory / filename).is_file() for filename in filenames):
            candidates.append(directory.resolve())
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise FileNotFoundError(
            f"expected one {name} Python-batch directory below {root}, found {len(unique)}"
        )
    return unique[0]


def _pickle_mapping(path: Path) -> Mapping[object, object]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    if not isinstance(payload, Mapping):
        raise ValueError(f"CIFAR file is not a mapping: {path}")
    return payload


def _mapping_value(mapping: Mapping[object, object], name: str) -> object:
    for key in (name, name.encode("ascii")):
        if key in mapping:
            return mapping[key]
    raise KeyError(name)


def _load_cifar_python(name: str, directory: Path, *, train: bool) -> DatasetTensors:
    if name == "cifar10":
        filenames = [f"data_batch_{index}" for index in range(1, 6)] if train else ["test_batch"]
        label_key = "labels"
        classes = 10
    else:
        filenames = ["train"] if train else ["test"]
        label_key = "fine_labels"
        classes = 100
    image_parts: list[Tensor] = []
    label_parts: list[Tensor] = []
    for filename in filenames:
        payload = _pickle_mapping(directory / filename)
        array = np.asarray(_mapping_value(payload, "data"), dtype=np.uint8)
        labels = np.asarray(_mapping_value(payload, label_key), dtype=np.int64)
        if array.ndim != 2 or array.shape[1] != 3072 or labels.shape != (array.shape[0],):
            raise ValueError(f"invalid CIFAR batch shape in {filename}")
        image_parts.append(torch.from_numpy(array.copy()).reshape(-1, 3, 32, 32))
        label_parts.append(torch.from_numpy(labels.copy()).to(torch.long))
    return DatasetTensors(
        torch.cat(image_parts).contiguous(),
        torch.cat(label_parts).contiguous(),
        classes,
        name,
    )


__all__ = [
    "DatasetTensors",
    "IndexedTensorDataset",
    "InputTransform",
    "SplitIndices",
    "fit_input_transform",
    "imagenet_input_transform",
    "load_test_data",
    "load_training_data",
    "make_ab_split",
    "split_fit_calibration",
]
