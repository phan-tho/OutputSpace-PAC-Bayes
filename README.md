# Output-space PAC-Bayes experiment

This folder contains one experiment: the multiclass symmetric independent-class-score canonical lift. The implementation is deliberately organized like research code rather than a reusable framework.

## Files

- `run.py` is the complete experiment, in statistical order from data loading to the one result file.
- `data.py` loads MNIST/CIFAR tensors, creates the index-only A/B split, and applies input preprocessing.
- `models.py` contains the backbones, deterministic prior training, A-only feature transforms, stochastic canonical posterior, and exact raw/quotient KL.
- `pac_bayes.py` contains observable-coordinate SVD, posterior optimization, Gauss--Hermite risk, fresh Monte Carlo, and the certificate.
- `utils.py` loads the plain JSON configuration and provides reproducibility/result helpers.
- `configs/` contains the paper presets. `kaggle/` contains thin Git-and-CLI notebooks only.

There is no score rescaling or temperature argument. A backbone is selected by a small dictionary in `models.py` and exposes `feature_dim`; the same canonical posterior is used for every dataset. Launching with `torchrun` uses one process per GPU and DDP for prior training. Posterior fitting and certification operate on the frozen extracted tensors.

## Install and verify

```bash
cd experiments/output_space_PB
python -m pip install -e .
python run.py --smoke --seed 7 --device cpu --output /tmp/output-space-pb-smoke.json
```

The smoke run is the built-in certificate-critical check. It exercises the observation-independent split, A-only prior and transforms, full-rank SVD without truncation, B-only selection, same-state raw/quotient KL, fresh post-selection Monte Carlo, delayed test diagnostics, and the single JSON writer. There is intentionally no `tests/` directory.

## Paper presets

```bash
python run.py --preset mnist-cnn32 --seed 7 --data-root ./data --download --device auto --output result.json
python run.py --preset mnist-lenet5 --seed 17 --data-root ./data --download --device auto --output result.json
python run.py --preset cifar10-wrn28-4 --seed 17 --data-root /datasets/cifar10 --device cuda --workers 4 --output result.json
python run.py --preset cifar100-wrn28-4 --seed 17 --data-root /datasets/cifar100 --device cuda --workers 4 --output result.json
```

CIFAR data can be a torchvision download or an extracted Python-batch directory anywhere under `--data-root`.

The transfer preset reuses the separately audited ImageNet statistic artifact:

```bash
python run.py --preset cifar10-imagenet-r18 \
  --seed 17 \
  --data-root /datasets/cifar10 \
  --upstream-stats /artifacts/imagenet_resnet18_pca_whitening.pt \
  --device cuda --workers 4 --output result.json
```

If torchvision cannot supply the matching official ResNet-18 weights, pass the exact local state dictionary with `--encoder-weights`. Its state hash must match the upstream artifact.

For the data-dependent ImageNet run, `configs/imagenet-resnet18.json` fixes the A/B split at one half and the standard 90-epoch ResNet-18 SGD recipe. `imagenet_stages.py prepare` is the temporary staged-workflow entry point: under `torchrun` it trains or loads the prior, extracts A and B on both GPUs, fits and saves all 512 A-only PCA directions, stores the 512 unscaled whitened B coordinates, and audits B support at ranks 129 and 257. The configured rank is sliced and the configured kappa is applied only by `imagenet_stages.py tune`, so neither is baked into the prepared features. Add `--prior-only` on the rented machine when extraction and tuning will be moved to Kaggle.

Use `--config my-run.json` to modify a preset. Dataset paths, output path, device, worker count, and checkpoint/artifact paths remain CLI arguments. Unknown JSON fields are rejected centrally in `utils.py`.

The required `--seed` controls the whole run: the A/B split uses that seed, posterior optimization uses `seed + 10000`, and final Monte Carlo uses `seed + 20000` (the direct-prior MC uses the next seed).

Every run writes one JSON document with exactly three top-level fields: `config`, `metrics`, and a one-line `report`. Candidate logs, copied source, manifests, and separate reports are not produced.
