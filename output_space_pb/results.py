"""Single compact result schema and atomic writer."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .config import ExperimentConfig


SCHEMA_VERSION = 1


def build_result(
    config: ExperimentConfig,
    *,
    git_commit: str,
    runtime: Mapping[str, Any],
    resolved_paths: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    config_payload = asdict(config)
    config_payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "git_commit": git_commit,
            "method": "symmetric_independent_class_scores",
            "runtime": dict(runtime),
            "paths": dict(resolved_paths),
        }
    )
    metrics_payload = dict(metrics)
    report = _short_report(config.name, metrics_payload)
    result = {
        "config": _jsonable(config_payload),
        "metrics": _jsonable(metrics_payload),
        "report": report,
    }
    validate_result(result)
    return result


def validate_result(result: Mapping[str, Any]) -> None:
    if set(result) != {"config", "metrics", "report"}:
        raise ValueError("result must contain exactly config, metrics, and report")
    if not isinstance(result["config"], Mapping) or not isinstance(result["metrics"], Mapping):
        raise ValueError("result config and metrics must be objects")
    if not isinstance(result["report"], str) or not result["report"].strip():
        raise ValueError("result report must be a nonempty string")
    status = result["metrics"].get("status")
    if status != "certified":
        raise ValueError("only completed certified runs may be written")


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    validate_result(result)
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _short_report(name: str, metrics: Mapping[str, Any]) -> str:
    certificate = metrics["certificate"]
    mc = metrics["fresh_mc"]
    kl = metrics["kl"]
    support = metrics["support"]
    lines = [
        f"# {name}",
        "",
        "Certified the symmetric independent-class-score Gibbs top-one rule.",
        f"The population-risk upper bound is {100.0 * certificate['population_gibbs_risk_upper']:.6f}%.",
        (
            f"Fresh MC observed {mc['errors']} errors in {mc['trials']} trials; "
            f"the one-sided endpoint is {100.0 * mc['clopper_pearson_upper']:.6f}%."
        ),
        (
            f"Raw/output KL are {kl['raw_nats']:.8g}/{kl['output_nats']:.8g} nats; "
            f"the B feature rank is {support['observed_rank']}/{support['latent_dimension']}."
        ),
        "Test values, when present, are diagnostic only.",
    ]
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


__all__ = ["build_result", "validate_result", "write_result"]
