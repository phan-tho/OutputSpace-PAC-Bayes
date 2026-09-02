"""Symmetric independent-class-score canonical lift and verified core numerics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OutputCoordinates:
    """All observable directions of the frozen feature matrix on B."""

    right_basis: Tensor
    singular_values: Tensor
    number_examples: int
    latent_dimension: int

    @property
    def observed_rank(self) -> int:
        return int(self.singular_values.numel())

    @property
    def smallest_singular_value(self) -> float:
        return float(self.singular_values[-1])

    @property
    def condition_number(self) -> float:
        return float(self.singular_values[0] / self.singular_values[-1])

    @classmethod
    def from_features(
        cls,
        features: Tensor,
        *,
        minimum_relative_singular_value: float,
    ) -> "OutputCoordinates":
        values = features.detach().cpu().to(torch.float64)
        if values.ndim != 2 or min(values.shape) <= 0:
            raise ValueError("B features must have nonempty shape [n,r]")
        if not bool(torch.all(torch.isfinite(values))):
            raise ValueError("B features contain non-finite values")
        if values.shape[0] < values.shape[1]:
            raise RuntimeError("current protocol requires n_B >= feature dimension")
        if not 0.0 <= minimum_relative_singular_value < 1.0:
            raise ValueError("invalid SVD audit threshold")
        _, singular_values, right_transpose = torch.linalg.svd(values, full_matrices=False)
        if singular_values.numel() != values.shape[1]:
            raise RuntimeError("compact SVD did not return every feature direction")
        if not bool(torch.all(torch.isfinite(singular_values))):
            raise RuntimeError("SVD produced non-finite singular values")
        if float(singular_values[-1]) <= 0.0:
            raise RuntimeError(
                "B feature matrix is rank deficient; no numerical rank truncation is allowed"
            )
        relative = float(singular_values[-1] / singular_values[0])
        if relative < minimum_relative_singular_value:
            raise RuntimeError(
                "B feature matrix failed the conditioning audit; no direction was discarded"
            )
        basis = right_transpose.T.contiguous()
        gram = basis.T @ basis
        if not torch.allclose(
            gram,
            torch.eye(basis.shape[1], dtype=torch.float64),
            atol=2e-12,
            rtol=2e-12,
        ):
            raise RuntimeError("SVD right basis is not numerically orthonormal")
        return cls(
            basis.detach(),
            singular_values.detach(),
            values.shape[0],
            values.shape[1],
        )

    def audit(self) -> dict[str, object]:
        return {
            "latent_dimension": self.latent_dimension,
            "observed_rank": self.observed_rank,
            "directions_discarded": 0,
            "full_column_rank": self.observed_rank == self.latent_dimension,
            "singular_values": [float(value) for value in self.singular_values],
            "smallest_singular_value": self.smallest_singular_value,
            "condition_number": self.condition_number,
        }


@dataclass(frozen=True)
class KLPair:
    raw: Tensor
    output: Tensor


def raw_independent_score_kl(means: Tensor, stds: Tensor) -> Tensor:
    _validate_posterior_parameters(means, stds)
    log_variances = 2.0 * torch.log(stds)
    return 0.5 * torch.sum(
        torch.expm1(log_variances) - log_variances + means.square()
    )


def quotient_independent_score_kl(means: Tensor, stds: Tensor) -> Tensor:
    """Exact symmetric quotient KL in O(Kr), without a reference class."""

    _validate_posterior_parameters(means, stds)
    classes, rank = means.shape
    centered_means = means - means.mean(dim=0, keepdim=True)
    mean_term = centered_means.square().sum(dim=0)
    log_variances = 2.0 * torch.log(stds)
    log_classes = means.new_tensor(float(classes)).log()
    covariance_term = (
        ((classes - 1.0) / classes) * torch.expm1(log_variances).sum(dim=0)
        - log_variances.sum(dim=0)
        - (torch.logsumexp(-log_variances, dim=0) - log_classes)
    )
    result = 0.5 * (mean_term + covariance_term).sum()
    tolerance = 1000.0 * torch.finfo(result.dtype).eps * max(1, (classes - 1) * rank)
    if float(result.detach()) < -tolerance:
        raise FloatingPointError("quotient KL was materially negative")
    return result.clamp_min(0.0)


class SymmetricIndependentScoreLift(nn.Module):
    """One diagonal posterior for independent Gaussian scores of every class."""

    def __init__(
        self,
        right_basis: Tensor,
        *,
        number_classes: int,
        minimum_std: float,
    ) -> None:
        super().__init__()
        if right_basis.ndim != 2 or min(right_basis.shape) <= 0:
            raise ValueError("right_basis must have shape [r,s]")
        if number_classes < 2 or not 0.0 < minimum_std < 1.0:
            raise ValueError("invalid class count or minimum standard deviation")
        gram = right_basis.T @ right_basis
        identity = torch.eye(
            right_basis.shape[1], dtype=right_basis.dtype, device=right_basis.device
        )
        tolerance = 2e-6 if right_basis.dtype == torch.float32 else 2e-12
        if not torch.allclose(gram, identity, atol=tolerance, rtol=tolerance):
            raise ValueError("right_basis columns must be orthonormal")
        self.number_classes = int(number_classes)
        self.minimum_std = float(minimum_std)
        self.register_buffer("right_basis", right_basis.detach().clone())
        shape = (number_classes, right_basis.shape[1])
        self.mean = nn.Parameter(torch.zeros(shape, dtype=right_basis.dtype, device=right_basis.device))
        initial = torch.tensor(
            1.0 - minimum_std,
            dtype=right_basis.dtype,
            device=right_basis.device,
        )
        raw = torch.log(torch.expm1(initial))
        self.raw_std = nn.Parameter(raw.expand(shape).clone())

    @property
    def std(self) -> Tensor:
        return self.minimum_std + F.softplus(self.raw_std)

    @property
    def observed_rank(self) -> int:
        return int(self.right_basis.shape[1])

    def score_statistics(
        self,
        features: Tensor,
        base_scores: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self._validate_inputs(features, base_scores)
        projected = features @ self.right_basis
        means = base_scores + projected @ self.mean.T
        visible_variance = projected.square() @ self.std.square().T
        residual = features - projected @ self.right_basis.T
        invisible_variance = residual.square().sum(dim=1, keepdim=True)
        variances = visible_variance + invisible_variance
        if bool(torch.any(variances < 0.0)):
            raise FloatingPointError("score variance became negative")
        return means, variances

    def errors_gauss_hermite(
        self,
        features: Tensor,
        base_scores: Tensor,
        labels: Tensor,
        *,
        order: int,
    ) -> Tensor:
        means, variances = self.score_statistics(features, base_scores)
        stochastic = torch.all(variances > 0.0, dim=1)
        deterministic = torch.all(variances == 0.0, dim=1)
        if not bool(torch.all(stochastic | deterministic)):
            raise RuntimeError("each example must be wholly stochastic or deterministic")
        safe = torch.where(stochastic[:, None], variances, torch.ones_like(variances))
        stochastic_errors = independent_score_errors_gauss_hermite(
            means,
            torch.sqrt(safe),
            labels,
            order=order,
        )
        deterministic_errors = (means.argmax(dim=1) != labels).to(means.dtype)
        return torch.where(stochastic, stochastic_errors, deterministic_errors)

    def kl_pair(self) -> KLPair:
        raw = raw_independent_score_kl(self.mean, self.std)
        output = quotient_independent_score_kl(self.mean, self.std)
        tolerance = 1000.0 * torch.finfo(raw.dtype).eps * max(1, self.mean.numel())
        if float(output.detach() - raw.detach()) > tolerance:
            raise FloatingPointError("output KL exceeds raw KL for the same posterior")
        return KLPair(raw=raw, output=output)

    @torch.no_grad()
    def remove_common_mean_gauge(self) -> None:
        self.mean.sub_(self.mean.mean(dim=0, keepdim=True))

    def _validate_inputs(self, features: Tensor, base_scores: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.right_basis.shape[0]:
            raise ValueError("features have the wrong shape")
        if base_scores.shape != (features.shape[0], self.number_classes):
            raise ValueError("base scores have the wrong shape")
        if features.dtype != self.right_basis.dtype or base_scores.dtype != features.dtype:
            raise ValueError("posterior inputs must share dtype")
        if features.device != self.right_basis.device or base_scores.device != features.device:
            raise ValueError("posterior inputs must share device")


def independent_score_errors_gauss_hermite(
    means: Tensor,
    stds: Tensor,
    labels: Tensor,
    *,
    order: int,
    quadrature_chunk_size: int = 256,
    class_chunk_size: int = 128,
) -> Tensor:
    if means.ndim != 2 or means.shape[1] < 2 or means.shape[0] == 0:
        raise ValueError("means must have shape [n,K>=2]")
    if stds.shape != means.shape or bool(torch.any(stds <= 0.0)):
        raise ValueError("standard deviations must be positive and match means")
    if labels.shape != (means.shape[0],) or labels.dtype != torch.long:
        raise ValueError("labels must be one-dimensional torch.long")
    if bool(torch.any((labels < 0) | (labels >= means.shape[1]))):
        raise ValueError("labels lie outside the score classes")
    if order <= 0 or min(quadrature_chunk_size, class_chunk_size) <= 0:
        raise ValueError("quadrature settings must be positive")
    nodes_cpu, weights_cpu = _standard_normal_rule(order)
    nodes = nodes_cpu.to(device=means.device, dtype=means.dtype)
    log_weights = weights_cpu.to(device=means.device, dtype=means.dtype)
    true_means = means.gather(1, labels[:, None]).squeeze(1)
    true_stds = stds.gather(1, labels[:, None]).squeeze(1)
    log_total: Tensor | None = None
    for point_start in range(0, order, quadrature_chunk_size):
        point_end = min(point_start + quadrature_chunk_size, order)
        selected_nodes = nodes[point_start:point_end]
        selected_weights = log_weights[point_start:point_end]
        sampled_true = true_means[:, None] + true_stds[:, None] * selected_nodes[None, :]
        log_integrand = means.new_zeros((means.shape[0], selected_nodes.numel()))
        for class_start in range(0, means.shape[1], class_chunk_size):
            class_end = min(class_start + class_chunk_size, means.shape[1])
            arguments = (
                sampled_true[:, :, None] - means[:, None, class_start:class_end]
            ) / stds[:, None, class_start:class_end]
            factors = torch.special.log_ndtr(arguments)
            class_indices = torch.arange(class_start, class_end, device=means.device)
            is_true = labels[:, None] == class_indices[None, :]
            factors = torch.where(is_true[:, None, :], torch.zeros_like(factors), factors)
            log_integrand = log_integrand + factors.sum(dim=2)
        chunk = torch.logsumexp(log_integrand + selected_weights[None, :], dim=1)
        log_total = chunk if log_total is None else torch.logaddexp(log_total, chunk)
    if log_total is None:  # pragma: no cover
        raise RuntimeError("quadrature produced no points")
    return -torch.expm1(log_total.clamp_max(0.0))


@lru_cache(maxsize=32)
def _standard_normal_rule(order: int) -> tuple[Tensor, Tensor]:
    diagonal = torch.zeros(order, dtype=torch.float64)
    if order == 1:
        return diagonal, diagonal
    indices = torch.arange(1, order, dtype=torch.float64)
    off_diagonal = torch.sqrt(indices)
    jacobi = torch.diag(diagonal)
    jacobi += torch.diag(off_diagonal, diagonal=1)
    jacobi += torch.diag(off_diagonal, diagonal=-1)
    nodes, eigenvectors = torch.linalg.eigh(jacobi)
    return nodes, torch.log(eigenvectors[0].square())


def _validate_posterior_parameters(means: Tensor, stds: Tensor) -> None:
    if means.ndim != 2 or means.shape[0] < 2 or means.shape[1] < 1:
        raise ValueError("posterior means must have shape [K>=2,r>=1]")
    if stds.shape != means.shape or stds.dtype != means.dtype or stds.device != means.device:
        raise ValueError("posterior means and standard deviations must match")
    if not means.is_floating_point() or not bool(torch.all(torch.isfinite(means))):
        raise ValueError("posterior means must be finite floating point")
    if not bool(torch.all(torch.isfinite(stds))) or bool(torch.any(stds <= 0.0)):
        raise ValueError("posterior standard deviations must be finite and positive")


__all__ = [
    "KLPair",
    "OutputCoordinates",
    "SymmetricIndependentScoreLift",
    "independent_score_errors_gauss_hermite",
    "quotient_independent_score_kl",
    "raw_independent_score_kl",
]
