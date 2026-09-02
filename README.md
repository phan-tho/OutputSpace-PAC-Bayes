# Output-Space PAC-Bayes Experiments

This folder is the implementation of the multiclass symmetric
independent-class-score canonical lift. It supports local, server, and Kaggle
runs through the same `run.py` command.
 <!-- Base scores enter the model unchanged; -->
<!-- there is no score-scaling option or search. -->

## Install and smoke check

```bash
cd experiments/output_space_PB
python -m pip install -e .
python run.py --smoke --device cpu --output /tmp/output-space-pb-smoke.json
```

The smoke command exercises the complete A/B split, A-only prior, feature-map
freeze, full-rank SVD, B-only posterior selection, raw and quotient KL,
fresh-IID Monte Carlo, Clopper--Pearson endpoint, PAC-Bayes inverse, delayed
test diagnostic, and compact result writer.

## Paper configurations

```bash
python run.py --preset mnist-cnn32 \
  --data-root ./data --download --device auto \
  --output ./results/mnist-cnn32/result.json

python run.py --preset mnist-lenet5 \
  --data-root ./data --download --device auto \
  --output ./results/mnist-lenet5/result.json

python run.py --preset cifar10-wrn28-4 \
  --data-root /datasets/cifar-10-batches-py --device cuda \
  --output ./results/cifar10-wrn28-4/result.json

python run.py --preset cifar100-wrn28-4 \
  --data-root /datasets/cifar-100-python --device cuda \
  --output ./results/cifar100-wrn28-4/result.json
```

CIFAR data may be a torchvision download or an offline Python-batch directory
located anywhere below `--data-root`.

The transfer configuration consumes the already audited ImageNet statistic
artifact. 
<!-- It does not import or execute the legacy experiment tree: -->

```bash
python run.py --preset cifar10-imagenet-r18 \
  --data-root /datasets/cifar-10-batches-py \
  --upstream-stats /artifacts/imagenet_resnet18_pca_whitening.pt \
  --device cuda \
  --output ./results/cifar10-imagenet-r18/result.json
```

<!-- If the official ResNet checkpoint is not in the torchvision cache, it is
downloaded by torchvision. `--encoder-weights` may instead name an exact local
state dictionary; its logical state hash must match the upstream artifact.
The existing legacy ImageNet statistics script may continue to produce the
upstream artifact until that preprocessing is independently replaced. -->

Use `--config path.json` for a small explicit configuration instead of a
committed preset. Unknown fields are rejected. CLI flags are restricted to
data paths, output paths, device/runtime choices, and trusted checkpoint
inputs.

## Architecture

The registered encoder contract is only `feature_dim` plus
`forward_features(inputs)`. The deterministic score head and frozen feature
map are composed around any registered encoder. The stochastic canonical
model consumes only frozen base scores and stochastic features, so it has no
dataset or backbone branches.

The authoritative output is one JSON document with exactly three top-level
fields:

```text
config   fully resolved scientific configuration and runtime provenance
metrics  support audit, selected posterior, both KLs, fresh MC, certificate,
         and explicitly diagnostic test values
report   a short embedded Markdown summary
```

Candidate tables, checkpoint reports, manifests, and separate report files
are not written. Training progress is emitted only to stdout.

<!-- ## Certificate-critical behavior

- A/B indices are generated from sample count and seed; labels and inputs are
  not accepted by the split function.
- Encoder weights, BatchNorm state, input normalization, feature
  standardization/PCA, and prior checkpoint selection are A-only or
  upstream-only.
- Posterior fitting and numerical selection use B only, and `n_B` is the
  complexity denominator.
- `Q=P` is always an eligible posterior candidate.
- The compact B-feature SVD uses CPU float64 and retains every returned
  direction. Rank loss or a failed conditioning audit aborts the run.
- Quadrature is used for optimization and selection only. The reported bound
  uses a new post-selection IID stream and a separately allocated one-sided
  Clopper--Pearson endpoint.
- Raw and quotient KL are computed from the same frozen posterior state and
  recorded with the same state hash.
- Test data is loaded only after posterior selection, fresh MC, and the
  certificate are complete. Test metrics never enter the bound.
- Every result describes one seed and one randomized/Gibbs top-one classifier.

The CLI performs these checks during every run. There is intentionally no
separate test suite or `tests/` directory in this experiment folder.

## Kaggle

The notebooks in `kaggle/` contain only environment setup, Git clone/update,
editable installation, paths, and CLI invocation. They contain no experiment
implementation, encoded source archive, or duplicated model code. Replace the
repository URL and dataset paths before running them. -->

