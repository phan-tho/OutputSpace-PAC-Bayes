#!/usr/bin/env python3
"""Train and certify the final ImageNet posterior from the prepared seed-7 artifact."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch

from data import load_test_set
from imagenet_stages import posterior_features
from models import (
    CanonicalPosterior,
    PriorModel,
    apply_feature_transform,
    extract_prior_outputs,
    feature_transform_report,
    make_backbone,
)
from pac_bayes import (
    fresh_monte_carlo,
    gauss_hermite_risk,
    kl_upper_inverse,
    optimize_posterior,
    pac_bayes_certificate,
)
from utils import (
    choose_device,
    git_commit,
    json_ready,
    load_config,
    runtime_info,
    save_json,
    set_deterministic,
    state_dict_sha256,
)


HERE = Path(__file__).resolve().parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs/imagenet-resnet18.json")
    parser.add_argument("--prepared", type=Path, default=Path("/workspace/imagenet_seed7/prepared.pt"))
    parser.add_argument(
        "--prior-checkpoint", type=Path, default=Path("/workspace/imagenet_seed7/prior.pt")
    )
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/imagenet_dataset"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/imagenet_seed7/result7.json"))
    parser.add_argument("--posterior-output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    started = time.perf_counter()
    config = load_config(args.config)
    if config["dataset"]["name"] != "imagenet" or config["prior"]["source"] != "a_trained":
        raise ValueError("imagenet_final.py requires the A-trained ImageNet configuration")
    if args.seed < 0 or args.workers < 0:
        raise ValueError("seed and workers must be nonnegative")
    config["seed"] = args.seed
    config["dataset"]["split_seed"] = args.seed
    config["posterior"]["seed"] = args.seed + 10_000
    config["certification"]["monte_carlo_seed"] = args.seed + 20_000
    device = choose_device(args.device)
    set_deterministic(args.seed)

    # 1. Load the A-only transform, frozen B scores, and audited B coordinates.
    artifact = torch.load(args.prepared.resolve(), map_location="cpu", weights_only=True)
    if artifact.get("artifact_type") != "imagenet_A_only_full_pca_and_B_scores":
        raise RuntimeError("unexpected prepared artifact type")
    if artifact["seed"] != args.seed:
        raise RuntimeError("prepared artifact seed does not match --seed")
    for section in ("dataset", "encoder", "prior", "numerics"):
        if artifact["config"][section] != config[section]:
            raise RuntimeError(f"prepared artifact uses a different {section} configuration")
    if artifact["config"]["feature_map"]["kind"] != config["feature_map"]["kind"]:
        raise RuntimeError("prepared artifact uses a different feature-map kind")

    prior_payload = torch.load(args.prior_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if not isinstance(prior_payload.get("state_dict"), dict):
        raise ValueError("prior checkpoint must contain a state_dict")
    prior_metadata = prior_payload.get("metadata", {})
    if prior_metadata.get("number_classes") != config["dataset"]["number_classes"]:
        raise RuntimeError("prior checkpoint class count does not match")
    if prior_metadata.get("A_index_sha256") != artifact["A_index_sha256"]:
        raise RuntimeError("prior checkpoint and prepared artifact use different A splits")
    prior_hash = state_dict_sha256(prior_payload["state_dict"])
    if prior_hash != artifact["prior_state_sha256"]:
        raise RuntimeError("prior checkpoint state differs from the prepared artifact")

    rank = config["feature_map"]["rank"]
    audited = artifact["observable_coordinates_by_rank"].get(str(rank))
    if audited is None:
        raise RuntimeError(f"prepared artifact has no observable-rank audit for rank {rank}")
    if audited["observed_rank"] != rank or audited["directions_discarded"] != 0:
        raise RuntimeError("prepared B support audit is invalid")
    features_B = posterior_features(
        artifact.pop("whitened_features_B"),
        rank,
        config["feature_map"]["kappa"],
    ).to(torch.float64)
    base_scores_B = artifact.pop("base_scores_B").to(torch.float64)
    labels_B = artifact.pop("labels_B").to(torch.long)
    if not (features_B.shape[0] == base_scores_B.shape[0] == labels_B.numel()):
        raise RuntimeError("prepared B tensors have inconsistent sizes")
    coordinates = {
        "right_basis": audited["right_basis"].to(torch.float64),
        "number_examples": labels_B.numel(),
    }
    full_transform = artifact["feature_transform"]
    components = rank - 1
    if full_transform["directions"].shape[1] < components:
        raise RuntimeError("prepared artifact contains too few PCA directions")
    sliced_transform = {
        "kind": "pca_whiten_bias",
        "mean": full_transform["mean"],
        "directions": full_transform["directions"][:, :components],
        "inverse_scales": full_transform["inverse_scales"][:components],
        "eigenvalues": full_transform["eigenvalues"][:components],
        "kappa": config["feature_map"]["kappa"],
    }
    print(
        f"[prepared] seed={args.seed} B={labels_B.numel()} rank={rank} "
        f"condition={audited['condition_number']:.6g} prior={prior_hash[:12]}",
        flush=True,
    )

    # 2. Optimize on B and select among Q=P and the configured checkpoints.
    posterior, selected, q_equals_p, states_evaluated = optimize_posterior(
        config["posterior"],
        coordinates,
        features_B,
        base_scores_B,
        labels_B,
        config["dataset"]["number_classes"],
        config["numerics"]["minimum_posterior_std"],
        config["confidence"]["pac_bayes_delta_each"],
        device,
    )
    posterior.freeze()
    posterior_hash = state_dict_sha256(posterior.state_dict())
    raw_kl, output_kl = (float(value) for value in posterior.kl_values())
    print(
        f"[posterior] step={selected['step']} raw_KL={raw_kl:.9g} "
        f"output_KL={output_kl:.9g}",
        flush=True,
    )

    # Save the selected law before the slower MC and diagnostic stages.
    posterior_output = (
        args.posterior_output.resolve()
        if args.posterior_output is not None
        else args.output.resolve().with_name("posterior.pt")
    )
    posterior_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "seed": args.seed,
            "rank": rank,
            "kappa": config["feature_map"]["kappa"],
            "selected": selected,
            "q_equals_p": q_equals_p,
            "posterior_state": posterior.state_dict(),
            "posterior_state_sha256": posterior_hash,
            "A_index_sha256": artifact["A_index_sha256"],
            "B_index_sha256": artifact["B_index_sha256"],
            "prior_state_sha256": prior_hash,
        },
        posterior_output,
    )

    # 3. Direct holdout for the stochastic prior, with its separate MC stream.
    direct_total_delta = config["confidence"]["direct_holdout_delta"]
    direct_mc_delta = config["confidence"]["monte_carlo_delta"]
    direct_data_delta = direct_total_delta - direct_mc_delta
    if direct_data_delta <= 0.0:
        raise ValueError("direct holdout delta must exceed its MC allocation")
    stochastic_prior = CanonicalPosterior(
        coordinates["right_basis"],
        config["dataset"]["number_classes"],
        config["numerics"]["minimum_posterior_std"],
    )
    stochastic_prior.freeze()
    prior_mc = fresh_monte_carlo(
        stochastic_prior,
        features_B,
        base_scores_B,
        labels_B,
        config["certification"]["monte_carlo_trials"],
        config["certification"]["monte_carlo_chunk_size"],
        config["certification"]["monte_carlo_seed"] + 1,
        direct_mc_delta,
    )
    direct_budget = math.log(1.0 / direct_data_delta) / labels_B.numel()
    direct_upper = kl_upper_inverse(prior_mc["clopper_pearson_upper"], direct_budget)
    direct_holdout = {
        "method": "stochastic_prior_Q_equals_P",
        "sample_size": labels_B.numel(),
        "monte_carlo": prior_mc,
        "monte_carlo_delta": direct_mc_delta,
        "concentration_delta": direct_data_delta,
        "total_delta": direct_total_delta,
        "concentration_binary_kl_budget": direct_budget,
        "population_gibbs_risk_upper": direct_upper,
        "confidence_statement_is_separate": True,
    }

    # 4. Fresh post-selection MC, then both KL certificates for the same Q.
    fresh_mc = fresh_monte_carlo(
        posterior,
        features_B,
        base_scores_B,
        labels_B,
        config["certification"]["monte_carlo_trials"],
        config["certification"]["monte_carlo_chunk_size"],
        config["certification"]["monte_carlo_seed"],
        config["confidence"]["monte_carlo_delta"],
    )
    if state_dict_sha256(posterior.state_dict()) != posterior_hash:
        raise RuntimeError("posterior changed during certification")
    output_certificate = pac_bayes_certificate(
        fresh_mc["clopper_pearson_upper"],
        output_kl,
        labels_B.numel(),
        config["confidence"]["pac_bayes_delta_each"],
    )
    raw_certificate = pac_bayes_certificate(
        fresh_mc["clopper_pearson_upper"],
        raw_kl,
        labels_B.numel(),
        config["confidence"]["pac_bayes_delta_each"],
    )
    confidence_report = {
        "n_B": labels_B.numel(),
        "pac_bayes_delta_each": config["confidence"]["pac_bayes_delta_each"],
        "pac_bayes_family_count": config["confidence"]["pac_bayes_family_count"],
        "monte_carlo_delta": config["confidence"]["monte_carlo_delta"],
        "joint_failure_allocation": (
            config["confidence"]["pac_bayes_delta_each"]
            * config["confidence"]["pac_bayes_family_count"]
            + config["confidence"]["monte_carlo_delta"]
        ),
        "selection_preceded_final_mc": True,
    }
    output_certificate.update(confidence_report)
    output_certificate["kl_type"] = "output_space_quotient"
    raw_certificate.update(confidence_report)
    raw_certificate["kl_type"] = "raw_gaussian_parameter"
    print(
        f"[certificate] Rhat={100*fresh_mc['observed_risk']:.5f}% "
        f"output={100*output_certificate['population_gibbs_risk_upper']:.5f}% "
        f"raw={100*raw_certificate['population_gibbs_risk_upper']:.5f}% "
        f"direct={100*direct_upper:.5f}%",
        flush=True,
    )

    # 5. Only now read ImageNet validation images and evaluate diagnostically.
    test_risk = None
    test_mean_error = None
    test_size = 0
    if config["certification"]["evaluate_test"]:
        backbone = make_backbone(config["encoder"], None, None)
        prior = PriorModel(backbone, config["dataset"]["number_classes"], zero_head=False)
        prior.load_state_dict(prior_payload["state_dict"], strict=True)
        prior.to(device).freeze()
        test_images, test_labels = load_test_set(
            config["dataset"], args.data_root.resolve(), False, args.seed
        )
        test_indices = torch.arange(test_labels.numel(), dtype=torch.long)
        raw_test, base_scores_test, labels_test = extract_prior_outputs(
            prior,
            test_images,
            test_labels,
            test_indices,
            artifact["input_transform"],
            max(config["prior"]["batch_size"], 256),
            device,
            args.workers,
        )
        features_test = apply_feature_transform(raw_test, sliced_transform)
        test_risk = gauss_hermite_risk(
            posterior,
            features_test,
            base_scores_test,
            labels_test,
            config["certification"]["diagnostic_quadrature_order"],
            config["posterior"]["selection_chunk_size"],
        )
        mean_test_scores, _ = posterior.score_statistics(features_test, base_scores_test)
        test_mean_error = float(
            (mean_test_scores.argmax(1) != labels_test).to(torch.float64).mean()
        )
        test_size = labels_test.numel()
    if state_dict_sha256(posterior.state_dict()) != posterior_hash:
        raise RuntimeError("posterior changed during diagnostic test evaluation")

    # 6. Write one compact result JSON with a directly usable paper table.
    paper_table = {
        "ours_output_space_kl": {
            "R_hat_B_percent": 100.0 * fresh_mc["observed_risk"],
            "KL_over_n_B": output_kl / labels_B.numel(),
            "certificate_percent": 100.0 * output_certificate["population_gibbs_risk_upper"],
            "test_risk_percent": None if test_risk is None else 100.0 * test_risk,
        },
        "raw_gaussian_parameter_kl": {
            "R_hat_B_percent": 100.0 * fresh_mc["observed_risk"],
            "KL_over_n_B": raw_kl / labels_B.numel(),
            "certificate_percent": 100.0 * raw_certificate["population_gibbs_risk_upper"],
            "test_risk_percent": None if test_risk is None else 100.0 * test_risk,
        },
        "direct_holdout_stochastic_prior": {
            "R_hat_B_percent": 100.0 * prior_mc["observed_risk"],
            "KL_over_n_B": 0.0,
            "certificate_percent": 100.0 * direct_upper,
            "test_risk_percent": None,
        },
    }
    test_report = "not evaluated" if test_risk is None else f"{100.0 * test_risk:.5f}%"
    support = {
        key: value for key, value in audited.items()
        if key not in {"right_basis", "singular_values"}
    }
    result = {
        "config": {
            "scientific": config,
            "runtime": runtime_info(device),
            "git_commit": git_commit(HERE),
            "paths": {
                "prepared": str(args.prepared.resolve()),
                "prior_checkpoint": str(args.prior_checkpoint.resolve()),
                "posterior_checkpoint": str(posterior_output),
                "data_root": str(args.data_root.resolve()),
                "output": str(args.output.resolve()),
            },
        },
        "metrics": {
            "status": "certified",
            "data": {
                "dataset": "imagenet",
                "A_size": artifact["A_indices"].numel(),
                "B_size": labels_B.numel(),
                "A_index_sha256": artifact["A_index_sha256"],
                "B_index_sha256": artifact["B_index_sha256"],
                "observation_independent_unstratified_split": True,
            },
            "prior": {
                "state_sha256": prior_hash,
                "checkpoint_metadata": prior_metadata,
                "input_transform": artifact["input_transform"],
                "feature_transform": feature_transform_report(sliced_transform),
                "direct_holdout": direct_holdout,
            },
            "support": support,
            "posterior": {
                **selected,
                "q_equals_p_B_gauss_hermite_risk": q_equals_p["B_gauss_hermite_risk"],
                "q_equals_p_selection_upper": q_equals_p["selection_upper"],
                "candidate_states_evaluated": states_evaluated,
                "state_sha256": posterior_hash,
            },
            "kl": {
                "raw_nats": raw_kl,
                "quotient_nats": output_kl,
                "raw_over_n_B": raw_kl / labels_B.numel(),
                "quotient_over_n_B": output_kl / labels_B.numel(),
                "same_posterior_state_sha256": posterior_hash,
            },
            "fresh_mc": fresh_mc,
            "certificate": output_certificate,
            "raw_gaussian_certificate": raw_certificate,
            "diagnostics": {
                "test_role": "diagnostic_only",
                "test_loaded_after_certificate": test_risk is not None,
                "test_size": test_size,
                "test_gibbs_risk_gauss_hermite": test_risk,
                "test_mean_argmax_error": test_mean_error,
            },
            "paper_table": paper_table,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "report": (
            f"ImageNet seed {args.seed}: Rhat_B={paper_table['ours_output_space_kl']['R_hat_B_percent']:.5f}%, "
            f"output KL/n={paper_table['ours_output_space_kl']['KL_over_n_B']:.9g}, "
            f"output certificate={paper_table['ours_output_space_kl']['certificate_percent']:.5f}%, "
            f"raw certificate={paper_table['raw_gaussian_parameter_kl']['certificate_percent']:.5f}%, "
            f"direct stochastic-prior certificate={paper_table['direct_holdout_stochastic_prior']['certificate_percent']:.5f}%, "
            f"test Gibbs risk={test_report}."
        ),
    }
    save_json(args.output, json_ready(result))
    print(f"[done] posterior={posterior_output}", flush=True)
    print(f"[done] result={args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
