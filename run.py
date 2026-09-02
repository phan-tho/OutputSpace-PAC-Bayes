#!/usr/bin/env python3
"""Run one symmetric independent-score output-space PAC-Bayes experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import scipy
import torch
from torch import Tensor

from output_space_pb.canonical import OutputCoordinates
from output_space_pb.certification import (
    clopper_pearson_upper,
    fresh_iid_monte_carlo,
    gauss_hermite_risk,
    pac_bayes_certificate,
)
from output_space_pb.config import ExperimentConfig, load_config, smoke_config
from output_space_pb.data import (
    fit_input_transform,
    load_test_data,
    load_training_data,
    make_ab_split,
)
from output_space_pb.encoders import available_encoders, build_encoder
from output_space_pb.optimize import search_posterior, state_dict_sha256
from output_space_pb.prior import (
    DeterministicPrior,
    extract_components,
    fit_feature_map,
    load_upstream_feature_map,
    seed_everything,
    tensor_sha256,
    train_or_load_prior,
    validate_upstream_encoder,
)
from output_space_pb.results import build_result, write_result


ROOT = Path(__file__).resolve().parent
PRESET_ROOT = ROOT / "configs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=available_presets())
    source.add_argument("--config", type=Path)
    source.add_argument(
        "--smoke",
        action="store_true",
        help="run the built-in small synthetic certificate check",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prior-checkpoint", type=Path)
    parser.add_argument("--upstream-stats", type=Path)
    parser.add_argument("--encoder-weights", type=Path)
    return parser


def available_presets() -> tuple[str, ...]:
    return tuple(path.stem for path in sorted(PRESET_ROOT.glob("*.json")))


def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.smoke:
        return smoke_config()
    if args.preset:
        return load_config(PRESET_ROOT / f"{args.preset}.json")
    return load_config(args.config)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def configure_runtime(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = resolve_config(args)
    config.validate()
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    device = resolve_device(args.device)
    configure_runtime(config.seed)
    data_root = args.data_root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else (ROOT / "results" / config.name / "result.json").resolve()
    )
    print(f"[setup] experiment={config.name} device={device} output={output}")

    # Training data is the only dataset loaded before all certified choices freeze.
    train = load_training_data(
        config.dataset,
        data_root,
        download=args.download,
        seed=config.seed,
    )
    split = make_ab_split(
        train.labels.numel(),
        config.dataset.prior_fraction,
        config.dataset.split_seed,
    )
    print(f"[data] A={split.prior.numel()} B={split.posterior.numel()} split=index-only")

    input_dimension = (
        config.dataset.synthetic_input_dimension
        if config.dataset.name == "synthetic"
        else None
    )
    encoder = build_encoder(
        config.encoder,
        input_dimension=input_dimension,
        weights_path=args.encoder_weights.resolve() if args.encoder_weights else None,
    )
    upstream_audit: dict[str, Any] | None = None
    if config.prior.source == "upstream":
        if args.upstream_stats is None:
            raise ValueError("the upstream preset requires --upstream-stats")
        feature_map, input_transform, upstream_audit = load_upstream_feature_map(
            args.upstream_stats.resolve(), config.feature_map
        )
        validate_upstream_encoder(encoder, feature_map)
        prior = DeterministicPrior(encoder, config.dataset.number_classes, zero_head=True)
    else:
        if args.upstream_stats is not None:
            raise ValueError("--upstream-stats is valid only for an upstream prior")
        input_transform = fit_input_transform(train, split.prior)
        prior = DeterministicPrior(encoder, config.dataset.number_classes, zero_head=False)

    prior_result = train_or_load_prior(
        prior,
        train,
        split.prior,
        input_transform,
        config.prior,
        device=device,
        workers=args.workers,
        seed=config.seed,
        checkpoint=args.prior_checkpoint.resolve() if args.prior_checkpoint else None,
    )
    prior.freeze()
    print(
        f"[prior] selected_epoch={prior_result.selected_epoch} "
        f"state={prior_result.state_sha256[:12]}"
    )

    if config.prior.source != "upstream":
        extracted_A = extract_components(
            prior,
            train,
            split.prior,
            input_transform,
            batch_size=max(config.prior.batch_size, 256),
            device=device,
            workers=args.workers,
        )
        feature_map = fit_feature_map(extracted_A.raw_features, config.feature_map)

    extracted_B = extract_components(
        prior,
        train,
        split.posterior,
        input_transform,
        batch_size=max(config.prior.batch_size, 256),
        device=device,
        workers=args.workers,
    )
    features_B = feature_map.transform(extracted_B.raw_features)
    base_B = extracted_B.base_scores.to(torch.float64)
    labels_B = extracted_B.labels.to(torch.long)
    coordinates = OutputCoordinates.from_features(
        features_B,
        minimum_relative_singular_value=config.numerics.minimum_relative_singular_value,
    )
    if coordinates.observed_rank != config.feature_map.rank:
        raise RuntimeError("observable B rank differs from the predeclared feature rank")
    print(
        f"[support] rank={coordinates.observed_rank}/{coordinates.latent_dimension} "
        f"condition={coordinates.condition_number:.6g}"
    )

    search = search_posterior(
        config.posterior,
        coordinates,
        features_B,
        base_B,
        labels_B,
        number_classes=config.dataset.number_classes,
        minimum_std=config.numerics.minimum_posterior_std,
        delta=config.confidence.pac_bayes_delta_each,
        train_device=device,
    )
    posterior = search.posterior
    frozen_hash = state_dict_sha256(posterior.state_dict())
    if frozen_hash != search.state_sha256:
        raise RuntimeError("selected posterior hash changed before certification")
    pair = posterior.kl_pair()
    raw_kl = float(pair.raw)
    output_kl = float(pair.output)
    print(
        f"[posterior] candidate={search.selected.candidate} step={search.selected.step} "
        f"raw_kl={raw_kl:.8g} output_kl={output_kl:.8g}"
    )

    direct_metrics: dict[str, Any] | None = None
    if config.certification.direct_holdout and config.prior.source == "a_trained":
        direct_errors = int(torch.count_nonzero(base_B.argmax(1) != labels_B))
        direct_metrics = {
            "errors": direct_errors,
            "sample_size": labels_B.numel(),
            "observed_error": direct_errors / labels_B.numel(),
            "delta": config.confidence.direct_holdout_delta,
            "upper": clopper_pearson_upper(
                direct_errors,
                labels_B.numel(),
                config.confidence.direct_holdout_delta,
            ),
            "confidence_statement_is_separate": True,
        }

    # A new generator is created only here, after selection and posterior hashing.
    monte_carlo = fresh_iid_monte_carlo(
        posterior,
        features_B,
        base_B,
        labels_B,
        trials=config.certification.monte_carlo_trials,
        chunk_size=config.certification.monte_carlo_chunk_size,
        seed=config.certification.monte_carlo_seed,
        delta=config.confidence.monte_carlo_delta,
    )
    certificate = pac_bayes_certificate(
        monte_carlo.clopper_pearson_upper,
        output_kl,
        labels_B.numel(),
        config.confidence.pac_bayes_delta_each,
    )
    if state_dict_sha256(posterior.state_dict()) != frozen_hash:
        raise RuntimeError("posterior changed during final Monte Carlo")
    print(
        f"[certificate] observed={100*monte_carlo.observed_risk:.6f}% "
        f"CP={100*monte_carlo.clopper_pearson_upper:.6f}% "
        f"bound={100*certificate.upper:.6f}%"
    )

    # Test data is deliberately inaccessible to every choice above this point.
    diagnostic_metrics: dict[str, Any] = {
        "test_role": "diagnostic_only",
        "test_loaded_after_certificate": False,
    }
    if config.certification.evaluate_test:
        test = load_test_data(
            config.dataset,
            data_root,
            download=args.download,
            seed=config.seed,
        )
        test_indices = torch.arange(test.labels.numel(), dtype=torch.long)
        extracted_test = extract_components(
            prior,
            test,
            test_indices,
            input_transform,
            batch_size=max(config.prior.batch_size, 256),
            device=device,
            workers=args.workers,
        )
        features_test = feature_map.transform(extracted_test.raw_features)
        test_base = extracted_test.base_scores.to(torch.float64)
        test_labels = extracted_test.labels.to(torch.long)
        test_risk = gauss_hermite_risk(
            posterior,
            features_test,
            test_base,
            test_labels,
            order=config.certification.diagnostic_quadrature_order,
            chunk_size=config.posterior.selection_chunk_size,
        )
        means, _ = posterior.score_statistics(features_test, test_base)
        diagnostic_metrics.update(
            {
                "test_loaded_after_certificate": True,
                "test_size": test_labels.numel(),
                "test_gibbs_risk_gauss_hermite": test_risk,
                "test_mean_argmax_error": float(
                    (means.argmax(1) != test_labels).to(torch.float64).mean()
                ),
            }
        )
    if state_dict_sha256(posterior.state_dict()) != frozen_hash:
        raise RuntimeError("posterior changed during diagnostic evaluation")

    metrics = {
        "status": "certified",
        "data": {
            "dataset": config.dataset.name,
            "training_size": train.labels.numel(),
            "A_size": split.prior.numel(),
            "B_size": split.posterior.numel(),
            "A_index_sha256": tensor_sha256(split.prior),
            "B_index_sha256": tensor_sha256(split.posterior),
            "observation_independent_unstratified_split": True,
        },
        "prior": {
            **asdict(prior_result),
            "input_transform": asdict(input_transform),
            "feature_map": feature_map.audit(),
            "direct_holdout": direct_metrics,
            "upstream": upstream_audit,
        },
        "support": coordinates.audit(),
        "posterior": {
            "candidate": search.selected.candidate,
            "objective": search.selected.objective,
            "learning_rate": search.selected.learning_rate,
            "step": search.selected.step,
            "B_gauss_hermite_risk": search.selected.B_gauss_hermite_risk,
            "selection_upper": search.selected.selection_upper,
            "mean_l2": search.selected.mean_l2,
            "std_min": search.selected.std_min,
            "std_max": search.selected.std_max,
            "q_equals_p_B_gauss_hermite_risk": search.q_equals_p.B_gauss_hermite_risk,
            "q_equals_p_selection_upper": search.q_equals_p.selection_upper,
            "candidate_states_evaluated": search.candidate_states_evaluated,
            "state_sha256": frozen_hash,
        },
        "kl": {
            "raw_nats": raw_kl,
            "output_nats": output_kl,
            "same_posterior_state_sha256": frozen_hash,
        },
        "fresh_mc": asdict(monte_carlo),
        "certificate": {
            "n_B": labels_B.numel(),
            "pac_bayes_delta_each": config.confidence.pac_bayes_delta_each,
            "pac_bayes_family_count": config.confidence.pac_bayes_family_count,
            "pac_bayes_union_total": (
                config.confidence.pac_bayes_delta_each
                * config.confidence.pac_bayes_family_count
            ),
            "monte_carlo_delta": config.confidence.monte_carlo_delta,
            "joint_failure_allocation": (
                config.confidence.pac_bayes_delta_each
                * config.confidence.pac_bayes_family_count
                + config.confidence.monte_carlo_delta
            ),
            "binary_kl_budget": certificate.budget,
            "population_gibbs_risk_upper": certificate.upper,
            "selection_preceded_final_mc": True,
        },
        "diagnostics": diagnostic_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result = build_result(
        config,
        git_commit=git_commit(ROOT),
        runtime=runtime_metadata(device),
        resolved_paths={
            "data_root": str(data_root),
            "output": str(output),
            "upstream_stats": str(args.upstream_stats.resolve()) if args.upstream_stats else None,
            "encoder_weights": str(args.encoder_weights.resolve()) if args.encoder_weights else None,
            "prior_checkpoint": str(args.prior_checkpoint.resolve()) if args.prior_checkpoint else None,
        },
        metrics=metrics,
    )
    write_result(output, result)
    print(f"[done] wrote {output}")
    return result


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "Apple MPS"
            if device.type == "mps"
            else platform.processor() or "CPU"
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except Exception as error:
        print(f"[failed] {type(error).__name__}: {error}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
