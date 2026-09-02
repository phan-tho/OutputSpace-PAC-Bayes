"""Generic B-only posterior optimization and full-B float64 selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .canonical import OutputCoordinates, SymmetricIndependentScoreLift
from .certification import (
    gauss_hermite_risk,
    pac_bayes_certificate,
    training_objective,
)
from .config import PosteriorConfig


@dataclass(frozen=True)
class SelectionSummary:
    candidate: str
    objective: str | None
    learning_rate: float | None
    step: int
    B_gauss_hermite_risk: float
    raw_kl: float
    output_kl: float
    selection_upper: float
    mean_l2: float
    std_min: float
    std_max: float


@dataclass(frozen=True)
class PosteriorSearchResult:
    posterior: SymmetricIndependentScoreLift
    selected: SelectionSummary
    q_equals_p: SelectionSummary
    candidate_states_evaluated: int
    state_sha256: str


def search_posterior(
    config: PosteriorConfig,
    coordinates: OutputCoordinates,
    features_B: Tensor,
    base_scores_B: Tensor,
    labels_B: Tensor,
    *,
    number_classes: int,
    minimum_std: float,
    delta: float,
    train_device: torch.device,
) -> PosteriorSearchResult:
    """Fit only on B and select every state using common CPU-float64 GH risk."""

    if features_B.dtype != torch.float64 or features_B.device.type != "cpu":
        raise ValueError("B components must enter posterior search as CPU float64")
    if labels_B.numel() != coordinates.number_examples:
        raise ValueError("B size differs from output coordinates")
    cpu_template = SymmetricIndependentScoreLift(
        coordinates.right_basis,
        number_classes=number_classes,
        minimum_std=minimum_std,
    )
    q_equals_p = _summary(
        "Q=P",
        None,
        None,
        0,
        cpu_template,
        features_B,
        base_scores_B,
        labels_B,
        config,
        delta,
    )
    selected_summary = q_equals_p
    selected_state = _cpu_state(cpu_template.state_dict())
    evaluated = 1
    training_dtype = torch.float64 if train_device.type == "cpu" else torch.float32
    basis_device = coordinates.right_basis.to(device=train_device, dtype=training_dtype)
    features_device = features_B.to(device=train_device, dtype=training_dtype)
    scores_device = base_scores_B.to(device=train_device, dtype=training_dtype)
    labels_device = labels_B.to(device=train_device)

    for objective_index, objective_name in enumerate(config.objectives):
        for rate_index, learning_rate in enumerate(config.learning_rates):
            candidate = SymmetricIndependentScoreLift(
                basis_device,
                number_classes=number_classes,
                minimum_std=minimum_std,
            )
            optimizer = torch.optim.Adam(candidate.parameters(), lr=learning_rate)
            generator = torch.Generator(device="cpu").manual_seed(
                config.seed + 10_000 * objective_index + 101 * rate_index
            )
            if config.warmup_steps:
                _mean_cross_entropy_warmup(
                    candidate,
                    optimizer,
                    features_device,
                    scores_device,
                    labels_device,
                    config,
                    generator,
                )
            best_candidate_summary: SelectionSummary | None = None
            best_candidate_state: dict[str, Tensor] | None = None
            for step in range(1, config.steps + 1):
                indices = _sample_indices(
                    labels_B.numel(), config.batch_size, generator, train_device
                )
                optimizer.zero_grad(set_to_none=True)
                errors = candidate.errors_gauss_hermite(
                    features_device.index_select(0, indices),
                    scores_device.index_select(0, indices),
                    labels_device.index_select(0, indices),
                    order=config.training_quadrature_order,
                )
                kl_value = candidate.kl_pair().output
                objective = training_objective(
                    objective_name,
                    errors.mean(),
                    kl_value,
                    labels_B.numel(),
                    delta,
                )
                if not bool(torch.isfinite(objective)):
                    raise FloatingPointError("posterior objective became non-finite")
                objective.backward()
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), config.gradient_clip_norm)
                optimizer.step()
                candidate.remove_common_mean_gauge()
                if step % config.checkpoint_every == 0 or step == config.steps:
                    portable = _portable_posterior(candidate, coordinates, number_classes, minimum_std)
                    summary = _summary(
                        f"{objective_name} lr={learning_rate:g}",
                        objective_name,
                        learning_rate,
                        step,
                        portable,
                        features_B,
                        base_scores_B,
                        labels_B,
                        config,
                        delta,
                    )
                    evaluated += 1
                    if best_candidate_summary is None or _selection_key(summary) < _selection_key(
                        best_candidate_summary
                    ):
                        best_candidate_summary = summary
                        best_candidate_state = _cpu_state(portable.state_dict())
            if best_candidate_summary is None or best_candidate_state is None:
                raise RuntimeError("posterior candidate produced no selectable checkpoint")
            if _selection_key(best_candidate_summary) < _selection_key(selected_summary):
                selected_summary = best_candidate_summary
                selected_state = best_candidate_state

    posterior = SymmetricIndependentScoreLift(
        coordinates.right_basis,
        number_classes=number_classes,
        minimum_std=minimum_std,
    )
    posterior.load_state_dict(selected_state, strict=True)
    posterior.eval()
    for parameter in posterior.parameters():
        parameter.requires_grad_(False)
    final_summary = _summary(
        selected_summary.candidate,
        selected_summary.objective,
        selected_summary.learning_rate,
        selected_summary.step,
        posterior,
        features_B,
        base_scores_B,
        labels_B,
        config,
        delta,
    )
    if final_summary != selected_summary:
        raise RuntimeError("selected posterior metrics did not replay exactly")
    return PosteriorSearchResult(
        posterior=posterior,
        selected=selected_summary,
        q_equals_p=q_equals_p,
        candidate_states_evaluated=evaluated,
        state_sha256=state_dict_sha256(posterior.state_dict()),
    )


def _mean_cross_entropy_warmup(
    posterior: SymmetricIndependentScoreLift,
    optimizer: torch.optim.Optimizer,
    features: Tensor,
    base_scores: Tensor,
    labels: Tensor,
    config: PosteriorConfig,
    generator: torch.Generator,
) -> None:
    for _ in range(config.warmup_steps):
        indices = _sample_indices(labels.numel(), config.batch_size, generator, labels.device)
        selected_features = features.index_select(0, indices)
        projected = selected_features @ posterior.right_basis
        mean_scores = base_scores.index_select(0, indices) + projected @ posterior.mean.T
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(mean_scores, labels.index_select(0, indices))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(posterior.parameters(), config.gradient_clip_norm)
        optimizer.step()
        posterior.remove_common_mean_gauge()


def _sample_indices(
    number_examples: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    count = min(number_examples, batch_size)
    values = torch.randint(
        0, number_examples, (count,), generator=generator, dtype=torch.long
    )
    return values.to(device)


def _portable_posterior(
    source: SymmetricIndependentScoreLift,
    coordinates: OutputCoordinates,
    number_classes: int,
    minimum_std: float,
) -> SymmetricIndependentScoreLift:
    result = SymmetricIndependentScoreLift(
        coordinates.right_basis,
        number_classes=number_classes,
        minimum_std=minimum_std,
    )
    state = {
        key: value.detach().cpu().to(torch.float64)
        for key, value in source.state_dict().items()
    }
    state["right_basis"] = coordinates.right_basis
    result.load_state_dict(state, strict=True)
    return result


def _summary(
    candidate: str,
    objective: str | None,
    learning_rate: float | None,
    step: int,
    posterior: SymmetricIndependentScoreLift,
    features: Tensor,
    base_scores: Tensor,
    labels: Tensor,
    config: PosteriorConfig,
    delta: float,
) -> SelectionSummary:
    risk = gauss_hermite_risk(
        posterior,
        features,
        base_scores,
        labels,
        order=config.selection_quadrature_order,
        chunk_size=config.selection_chunk_size,
    )
    pair = posterior.kl_pair()
    raw = float(pair.raw.detach())
    output = float(pair.output.detach())
    upper = pac_bayes_certificate(risk, output, labels.numel(), delta).upper
    return SelectionSummary(
        candidate=candidate,
        objective=objective,
        learning_rate=learning_rate,
        step=step,
        B_gauss_hermite_risk=risk,
        raw_kl=raw,
        output_kl=output,
        selection_upper=upper,
        mean_l2=float(torch.linalg.vector_norm(posterior.mean.detach())),
        std_min=float(posterior.std.detach().min()),
        std_max=float(posterior.std.detach().max()),
    )


def _selection_key(summary: SelectionSummary) -> tuple[float, float, str, int]:
    return (
        summary.selection_upper,
        summary.output_kl,
        summary.candidate,
        summary.step,
    )


def _cpu_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().cpu().to(torch.float64).clone() for key, value in state.items()}


def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        encoded = key.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


__all__ = ["PosteriorSearchResult", "SelectionSummary", "search_posterior"]
