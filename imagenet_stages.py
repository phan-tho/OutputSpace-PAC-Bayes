#!/usr/bin/env python3
"""Temporary staged ImageNet runner for expensive preparation and Kaggle tuning."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from data import fit_A_only_input_transform, load_training_set, observation_independent_split
from models import (
    PriorModel,
    extract_prior_outputs,
    feature_transform_report,
    fit_feature_transform,
    make_backbone,
    train_or_load_prior,
)
from pac_bayes import observable_coordinates, optimize_posterior
from utils import (
    choose_device,
    json_ready,
    load_config,
    save_json,
    set_deterministic,
    state_dict_sha256,
    tensor_sha256,
)


HERE = Path(__file__).resolve().parent
AUDITED_POSTERIOR_RANKS = (129, 257)


def posterior_features(whitened_features: torch.Tensor, rank: int, kappa: float) -> torch.Tensor:
    """Build kappa/sqrt(r) [1; first r-1 whitened PCA coordinates]."""

    if rank < 2 or rank - 1 > whitened_features.shape[1]:
        raise ValueError("posterior rank does not fit the saved PCA whitening")
    bias = whitened_features.new_ones((whitened_features.shape[0], 1))
    return kappa * torch.cat((bias, whitened_features[:, : rank - 1]), 1) / rank**0.5


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=HERE / "configs/imagenet-resnet18.json")
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--prior-checkpoint", type=Path)
    prepare.add_argument("--seed", type=int, default=7)
    prepare.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    prepare.add_argument("--workers", type=int, default=8)
    prepare.add_argument("--prior-only", action="store_true")

    tune = commands.add_parser("tune")
    tune.add_argument("--config", type=Path, default=HERE / "configs/imagenet-resnet18.json")
    tune.add_argument("--prepared", type=Path, required=True)
    tune.add_argument("--output", type=Path, required=True)
    tune.add_argument("--learning-rate", type=float, required=True)
    tune.add_argument("--steps", type=int)
    tune.add_argument("--seed", type=int, default=7)
    tune.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return result


def prepare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if config["dataset"]["name"] != "imagenet" or config["prior"]["source"] != "a_trained":
        raise ValueError("prepare requires the A-trained ImageNet configuration")
    config["seed"] = args.seed
    config["dataset"]["split_seed"] = args.seed

    device = choose_device(args.device)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    set_deterministic(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_images, train_labels = load_training_set(
        config["dataset"], args.data_root.resolve(), False, args.seed
    )
    A_indices, B_indices = observation_independent_split(
        train_labels.numel(), config["dataset"]["prior_fraction"], args.seed
    )
    input_transform = fit_A_only_input_transform(train_images, A_indices, "imagenet")
    if rank == 0:
        print(
            f"[prepare] device={device} workers_per_rank={args.workers} "
            f"A={A_indices.numel()} B={B_indices.numel()}",
            flush=True,
        )
    backbone = make_backbone(config["encoder"], None, None)
    prior = PriorModel(backbone, config["dataset"]["number_classes"], zero_head=False)
    prior_metrics = train_or_load_prior(
        prior, train_images, train_labels, A_indices, input_transform, config["prior"],
        device, args.workers, args.seed,
        args.prior_checkpoint.resolve() if args.prior_checkpoint else None,
    )

    checkpoint_path = output_dir / "prior.pt"
    if rank == 0:
        torch.save({
            "state_dict": {
                name: value.detach().cpu() for name, value in prior.state_dict().items()
            },
            "metadata": {
                "number_classes": config["dataset"]["number_classes"],
                "A_index_sha256": tensor_sha256(A_indices),
                "selected_epoch": prior_metrics["selected_epoch"],
                "seed": args.seed,
            },
        }, checkpoint_path)
    if distributed:
        dist.barrier()

    if args.prior_only:
        if distributed:
            dist.destroy_process_group()
        if rank == 0:
            print(f"[prepared] prior={checkpoint_path}")
        return

    # Reload rank 0's BatchNorm buffers so all extraction shards use one exact prior.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    prior.load_state_dict(checkpoint["state_dict"], strict=True)
    prior.to(device).freeze()
    local_A = torch.tensor_split(A_indices, world_size)[rank]
    local_B = torch.tensor_split(B_indices, world_size)[rank]
    extraction_batch = max(config["prior"]["batch_size"], 256)
    raw_A, _, _ = extract_prior_outputs(
        prior, train_images, train_labels, local_A, input_transform,
        extraction_batch, device, args.workers,
        include_scores=False, output_dtype=torch.float32,
    )
    raw_B, base_B, labels_B = extract_prior_outputs(
        prior, train_images, train_labels, local_B, input_transform,
        extraction_batch, device, args.workers, output_dtype=torch.float32,
    )
    shard_path = output_dir / f"features-rank{rank}.pt"
    torch.save({"raw_A": raw_A, "raw_B": raw_B, "base_B": base_B, "labels_B": labels_B}, shard_path)
    del raw_A, raw_B, base_B, labels_B

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return

    shards = [
        torch.load(output_dir / f"features-rank{index}.pt", map_location="cpu", weights_only=True)
        for index in range(world_size)
    ]
    raw_A = torch.cat([shard["raw_A"] for shard in shards])
    full_pca_config = dict(config["feature_map"])
    full_pca_config["rank"] = config["encoder"]["feature_dimension"] + 1
    feature_transform = fit_feature_transform(raw_A, full_pca_config)
    del raw_A
    raw_B = torch.cat([shard["raw_B"] for shard in shards])
    base_scores_B = torch.cat([shard["base_B"] for shard in shards])
    labels_B = torch.cat([shard["labels_B"] for shard in shards])
    del shards
    values_B = raw_B.to(torch.float64)
    whitened_features_B = (
        (values_B - feature_transform["mean"]) @ feature_transform["directions"]
    ) * feature_transform["inverse_scales"]
    del raw_B, values_B
    coordinates_by_rank = {}
    for rank_width in AUDITED_POSTERIOR_RANKS:
        # Kappa is deliberately omitted here. Positive scalar scaling changes
        # singular values, but not rank, conditioning, or the right basis.
        unscaled = posterior_features(whitened_features_B, rank_width, 1.0)
        coordinates = observable_coordinates(
            unscaled, config["numerics"]["minimum_relative_singular_value"]
        )
        if coordinates["observed_rank"] != rank_width:
            raise RuntimeError(f"observable B rank differs from requested rank {rank_width}")
        coordinates_by_rank[str(rank_width)] = coordinates

    prepared_path = output_dir / "prepared.pt"
    torch.save({
        "schema_version": 1,
        "artifact_type": "imagenet_A_only_full_pca_and_B_scores",
        "seed": args.seed,
        "config": config,
        "A_indices": A_indices,
        "B_indices": B_indices,
        "A_index_sha256": tensor_sha256(A_indices),
        "B_index_sha256": tensor_sha256(B_indices),
        "prior_state_sha256": state_dict_sha256(prior.state_dict()),
        "input_transform": input_transform,
        "feature_transform": feature_transform,
        "whitened_features_B": whitened_features_B.to(torch.float32),
        "base_scores_B": base_scores_B.to(torch.float32),
        "labels_B": labels_B,
        "observable_coordinates_by_rank": coordinates_by_rank,
    }, prepared_path)
    support_by_rank = {
        rank_width: {
            key: value for key, value in coordinates.items()
            if key not in {"right_basis", "singular_values"}
        }
        for rank_width, coordinates in coordinates_by_rank.items()
    }
    save_json(output_dir / "prepared.json", json_ready({
        "seed": args.seed,
        "prior": prior_metrics,
        "A_size": A_indices.numel(),
        "B_size": B_indices.numel(),
        "A_index_sha256": tensor_sha256(A_indices),
        "B_index_sha256": tensor_sha256(B_indices),
        "feature_transform": feature_transform_report(feature_transform),
        "stored_whitened_dimensions": whitened_features_B.shape[1],
        "stored_whitened_features_include_bias": False,
        "stored_whitened_features_include_kappa": False,
        "available_posterior_ranks": list(AUDITED_POSTERIOR_RANKS),
        "observable_svd_kappa": 1.0,
        "support_by_rank": support_by_rank,
        "prepared_artifact": str(prepared_path),
    }))
    for index in range(world_size):
        (output_dir / f"features-rank{index}.pt").unlink()
    print(f"[prepared] prior={checkpoint_path}")
    print(f"[prepared] features/PCA/support={prepared_path}")


def tune(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    config["seed"] = args.seed
    config["dataset"]["split_seed"] = args.seed
    config["posterior"]["learning_rates"] = [args.learning_rate]
    if args.steps is not None:
        config["posterior"]["steps"] = args.steps
    config["posterior"]["seed"] = args.seed + 10_000
    artifact = torch.load(args.prepared, map_location="cpu", weights_only=True)
    if artifact.get("artifact_type") != "imagenet_A_only_full_pca_and_B_scores":
        raise RuntimeError("unexpected prepared artifact type")
    if artifact["seed"] != args.seed:
        raise RuntimeError("prepared artifact seed does not match")
    for section in ("dataset", "encoder", "prior", "numerics"):
        if artifact["config"][section] != config[section]:
            raise RuntimeError(f"prepared artifact uses a different {section} configuration")
    if artifact["config"]["feature_map"]["kind"] != config["feature_map"]["kind"]:
        raise RuntimeError("prepared artifact uses a different feature-map kind")
    rank = config["feature_map"]["rank"]
    coordinates_for_rank = artifact["observable_coordinates_by_rank"].get(str(rank))
    if coordinates_for_rank is None:
        raise RuntimeError(f"prepared artifact has no observable-rank audit for rank {rank}")
    whitened_features_B = artifact["whitened_features_B"].to(torch.float64)
    features_B = posterior_features(
        whitened_features_B, rank, config["feature_map"]["kappa"]
    )
    base_scores_B = artifact["base_scores_B"].to(torch.float64)
    labels_B = artifact["labels_B"].to(torch.long)
    coordinates = {
        "right_basis": coordinates_for_rank["right_basis"].to(torch.float64),
        "number_examples": labels_B.numel(),
    }
    device = choose_device(args.device)
    set_deterministic(args.seed)
    posterior, selected, q_equals_p, states_evaluated = optimize_posterior(
        config["posterior"], coordinates, features_B, base_scores_B, labels_B,
        config["dataset"]["number_classes"], config["numerics"]["minimum_posterior_std"],
        config["confidence"]["pac_bayes_delta_each"], device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "seed": args.seed,
        "rank": rank,
        "kappa": config["feature_map"]["kappa"],
        "learning_rate": args.learning_rate,
        "steps": config["posterior"]["steps"],
        "selected": selected,
        "q_equals_p": q_equals_p,
        "candidate_states_evaluated": states_evaluated,
        "posterior_state": posterior.state_dict(),
    }, args.output)
    print(f"[tune] {selected}")
    print(f"[tune] wrote {args.output}")


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        tune(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
