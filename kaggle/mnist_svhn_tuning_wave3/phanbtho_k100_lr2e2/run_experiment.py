#!/usr/bin/env python3
"""Thin Kaggle wrapper for one SVHN-to-MNIST tuning run."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch


KAPPA = 1.00
LEARNING_RATE = 0.020
SEED = 7
EXPECTED_COMMIT = "8be97fed238bf75913710750693e6ca07117a951"


def replace_once(path, old, new):
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"expected one runtime patch target in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def find_one(name):
    matches = list(Path("/kaggle/input").glob(f"**/{name}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {matches}")
    return matches[0]


working = Path("/kaggle/working")
repo = Path("/tmp/OutputSpace-PAC-Bayes")
data_root = Path("/tmp/torchvision-data")
if repo.exists():
    shutil.rmtree(repo)
if data_root.exists():
    shutil.rmtree(data_root)

subprocess.run([
    "git", "clone", "--quiet", "--depth", "1", "--branch", "main",
    "https://github.com/phan-tho/OutputSpace-PAC-Bayes.git", str(repo),
], check=True)
subprocess.run([
    "git", "-C", str(repo), "fetch", "--quiet", "--depth", "1", "origin", EXPECTED_COMMIT
], check=True)
subprocess.run([
    "git", "-C", str(repo), "checkout", "--quiet", EXPECTED_COMMIT
], check=True)
commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
if commit != EXPECTED_COMMIT:
    raise RuntimeError(f"unexpected source commit {commit}")

# The public commit fixes kappa=1. These three tiny, checked replacements make
# this kernel an explicit sweep of phi=[1, kappa*raw_32] without changing Git.
replace_once(
    repo / "models.py",
    'return torch.cat((bias, values), 1).contiguous()',
    'return torch.cat((bias, transform["kappa"] * values), 1).contiguous()',
)
replace_once(
    repo / "run.py",
    '"kind": "raw_feature_bias", "raw_dimension": backbone.feature_dim',
    '"kind": "raw_feature_bias", "raw_dimension": backbone.feature_dim, "kappa": config["feature_map"]["kappa"]',
)
replace_once(
    repo / "utils.py",
    'feature_map["rank"] != encoder["feature_dimension"] + 1 or feature_map["kappa"] != 1.0',
    'feature_map["rank"] != encoder["feature_dimension"] + 1 or feature_map["kappa"] <= 0.0',
)

# Mount native files in torchvision's expected layout; no dataset is downloaded.
data_root.mkdir(parents=True)
os.symlink(find_one("train_32x32.mat"), data_root / "train_32x32.mat")
mnist_raw = data_root / "MNIST" / "raw"
mnist_raw.mkdir(parents=True)
for filename in (
    "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
):
    os.symlink(find_one(filename), mnist_raw / filename)

config = json.loads((repo / "configs" / "mnist-svhn-cnn32.json").read_text())
config["feature_map"]["kappa"] = KAPPA
config["posterior"]["learning_rates"] = [LEARNING_RATE]
config["confidence"]["pac_bayes_family_count"] = 6
config["confidence"]["pac_bayes_delta_each"] = 0.0075
config_path = Path("/tmp/tuning-config.json")
config_path.write_text(json.dumps(config), encoding="utf-8")

result_path = working / "metrics.json"
log_path = working / "run.log"
command = [
    sys.executable, str(repo / "run.py"), "--config", str(config_path),
    "--data-root", str(data_root), "--seed", str(SEED), "--device", "cuda",
    "--workers", "4", "--output", str(result_path),
]
print({
    "kappa": KAPPA, "learning_rate": LEARNING_RATE, "seed": SEED,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
})
started = time.perf_counter()
with log_path.open("w", encoding="utf-8") as log:
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
elapsed = time.perf_counter() - started
print(log_path.read_text(encoding="utf-8"))

result = json.loads(result_path.read_text())
result["config"]["kaggle_sweep"] = {
    "source_commit": commit,
    "kappa": KAPPA,
    "learning_rate": LEARNING_RATE,
    "prior_family_count": 6,
    "runtime_feature_map": "phi=[1,kappa*raw_32]",
    "elapsed_seconds": elapsed,
}
result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
metrics = result["metrics"]
report = (
    f"kappa={KAPPA:g}, lr={LEARNING_RATE:g}, "
    f"selected_step={metrics['posterior']['step']}, "
    f"B_risk={metrics['posterior']['B_gauss_hermite_risk']:.8f}, "
    f"output_KL={metrics['kl']['quotient_nats']:.8f}, "
    f"certificate={metrics['certificate']['population_gibbs_risk_upper']:.8f}, "
    f"test_risk={metrics['diagnostics']['test_gibbs_risk_gauss_hermite']:.8f}"
)
(working / "report.md").write_text(report + "\n", encoding="utf-8")
shutil.rmtree(repo)
shutil.rmtree(data_root)
