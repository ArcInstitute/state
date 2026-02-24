# Predicting cellular responses to perturbation across diverse contexts with State

> Train State transition models or pretrain State embedding models. See the State [paper](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2).
> 
> See the [Google Colab](https://colab.research.google.com/drive/1QKOtYP7bMpdgDJEipDxaJqOchv7oQ-_l) to train STATE for the [Virtual Cell Challenge](https://virtualcellchallenge.org/).

## Associated repositories

- Model evaluation framework: [cell-eval](https://github.com/ArcInstitute/cell-eval)
- Dataloaders and preprocessing: [cell-load](https://github.com/ArcInstitute/cell-load)

## Getting started

- Train an ST model for genetic perturbation prediction using the Replogle-Nadig dataset: [Colab](https://colab.research.google.com/drive/1Ih-KtTEsPqDQnjTh6etVv_f-gRAA86ZN)
- Perform inference using an ST model trained on Tahoe-100M: [Colab](https://colab.research.google.com/drive/1bq5v7hixnM-tZHwNdgPiuuDo6kuiwLKJ)
- Embed and annotate a new dataset using SE: [Colab](https://colab.research.google.com/drive/1uJinTJLSesJeot0mP254fQpSxGuDEsZt)
- Train STATE for the Virtual Cell Challenge: [Colab](https://colab.research.google.com/drive/1QKOtYP7bMpdgDJEipDxaJqOchv7oQ-_l)

## Installation

### Installation from PyPI

This package is distributed via [`uv`](https://docs.astral.sh/uv).

```bash
uv tool install arc-state
```

### Installation from Source

```bash
git clone git@github.com:ArcInstitute/state.git
cd state
uv run state
```

When making fundamental changes to State, install an editable version with the `-e` flag.

```bash
git clone git@github.com:ArcInstitute/state.git
cd state
uv tool install -e .
```

## CLI Usage

If installed via `uv tool install`, run `state ...`. From source, run `uv run state ...`.
Use `state --help` (or `state tx --help`, `state emb --help`) to see available subcommands.

## State Transition Model (ST)

Use `state tx` to train and run perturbation prediction models.

### preprocess_train

Prepares training h5ad files by normalizing counts, applying log1p, and selecting highly variable genes (HVGs).
The HVG matrix is stored in `.obsm["X_hvg"]`, and the output .h5ad is written to `--output`.

```bash
state tx preprocess_train \
  --adata /path/to/raw_data.h5ad \
  --output /path/to/preprocessed_training_data.h5ad \
  --num_hvgs 2000
```

### train

Trains an ST model using Hydra overrides. Point `data.kwargs.toml_config_path` at a TOML file that defines datasets and splits.
`output_dir` and `name` define the run directory (`output_dir/name`).

```bash
state tx train \
  data.kwargs.toml_config_path=examples/fewshot.toml \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  training.max_steps=40000 \
  training.batch_size=8 \
  model=state \
  output_dir="$HOME/state" \
  name="test"
```

Optional downsampling overrides:
- `data.kwargs.downsample` downsample counts during loading (`<=1` keeps that fraction by binomial sampling, `>1` targets that read depth per cell for `output_space=all`).
- `data.kwargs.downsample_cells` caps the number of cells loaded per `(cell_type, perturbation[, batch])` group while leaving smaller groups unchanged.

### ST training contract: `embed_key`, `output_space`, decoder, and metrics

For ST (`state tx train`), two settings jointly define target tensors, decoder usage, and logged losses:
- `data.kwargs.embed_key`
- `data.kwargs.output_space`

The current contract is:

| `embed_key` | `output_space` | Main target/loss space | Decoder target (if present) | Logged metrics |
|---|---|---|---|---|
| `X_hvg` or `null` | `gene` | Expression (HVG/count space) | None expected | `train/expression_loss`, `val/expression_loss` |
| non-`X_hvg` embedding key (for example `X_state`, `X_uce`) | `gene` | Embedding space | Expression (HVG/count space) | `train/embedding_loss`, `val/embedding_loss`, plus `train/expression_loss`, `val/expression_loss` |
| non-`X_hvg` embedding key | `all` | Embedding space | Expression (full transcriptome) | `train/embedding_loss`, `val/embedding_loss`, plus `train/expression_loss`, `val/expression_loss` |
| any | `embedding` | Embedding space | Disabled | `train/embedding_loss`, `val/embedding_loss` |

Notes:
- Main loss is always computed from the model output against `batch["pert_cell_emb"]`.
- Decoder loss (expression loss) is only logged when decoder targets are available in `batch["pert_cell_counts"]`.
- With `output_space=embedding`, decoder is disabled.
- Best-checkpoint monitoring is:
  - `val/expression_loss` when `output_space` is `gene` or `all`
  - `val/embedding_loss` when `output_space` is `embedding`

### predict

Evaluates a trained run with `cell-eval` metrics (or just runs prediction with `--predict-only`).
`--output-dir` should point to the run directory that contains `config.yaml` and `checkpoints/`.

```bash
state tx predict --output-dir $HOME/state/test --checkpoint final.ckpt
```

Use `--toml` to evaluate on a different dataset/split config than the one saved in the run.

### infer

Runs inference on new data (not necessarily in the training TOML). Provide the run directory via `--model-dir`
and an input AnnData via `--adata`. Use `--embed-key X_hvg` if you trained on HVG features.

```bash
state tx infer \
  --model-dir /path/to/run \
  --checkpoint /path/to/run/checkpoints/final.ckpt \
  --adata /path/to/preprocessed_data.h5ad \
  --pert-col gene \
  --embed-key X_hvg \
  --output /path/to/output.h5ad
```

If `--output` ends with `.npy`, only the predictions matrix is written (no .h5ad).

### ST TOML configuration

The TOML file referenced by `data.kwargs.toml_config_path` defines dataset paths and splits.

Required sections:
- `[datasets]`: map dataset names to directories of h5ad files
- `[training]`: select which datasets participate in training

Optional sections:
- `[zeroshot]`: hold out entire cell types for val/test
- `[fewshot]`: hold out perturbations within specific cell types

Minimal example:

```toml
[datasets]
replogle = "/data/replogle/"

[training]
replogle = "train"

[zeroshot]
"replogle.jurkat" = "test"

[fewshot]
[fewshot."replogle.k562"]
val = ["AARS"]
test = ["NUP107"]
```

Notes:
- Use the `"dataset.cell_type"` format in `[zeroshot]` and `[fewshot]`.
- Anything not listed in `[zeroshot]`/`[fewshot]` defaults to training.
- Dataset paths should point to directories containing h5ad files.

See `examples/zeroshot.toml` and `examples/fewshot.toml` for more.

## State Embedding Model (SE)

Use `state emb` to pretrain cell embeddings or embed new datasets.

### preprocess (build a profile for your data)

This scans your train/val datasets and creates a new embedding profile. It:
- expects CSVs with `species,path,names` columns (each `names` value becomes a dataset id in the config)
- auto-detects the gene-name field in `var` (checks `_index`, `gene_name`, `gene_symbols`, `feature_name`, `gene_id`, `symbol`)
- uses `--all-embeddings` if provided (a torch dict of `{gene_name: embedding}`); otherwise builds one-hot embeddings
- writes `all_embeddings_{profile}.pt`, `ds_emb_mapping_{profile}.torch`, `valid_genes_masks_{profile}.torch`,
  plus `train_{profile}.csv` and `val_{profile}.csv` in `--output-dir`
- updates the embeddings/dataset entries in the config file; it does not modify your h5ad files

By default it updates `src/state/configs/state-defaults.yaml` unless you pass `--config-file`.

Training on your own data (recommended workflow):

```bash
cp src/state/configs/state-defaults.yaml my_state.yaml

uv run state emb preprocess \
  --profile-name my_data \
  --train-csv /path/train.csv \
  --val-csv /path/val.csv \
  --output-dir /path/state_emb_profile \
  --config-file my_state.yaml \
  --all-embeddings /path/gene_embeddings.pt \
  --num-threads 8
```

Example CSV format:

```csv
species,path,names
human,/path/ds1.h5ad,ds1
human,/path/ds2.h5ad,ds2
```

Tip: `--config-file` should point to a full config (copy `src/state/configs/state-defaults.yaml`), since preprocess only updates
the embeddings/dataset sections.

### fit

Trains the embedding model. You must set `embeddings.current` and `dataset.current` to the profile created above.

```bash
uv run state emb fit --conf my_state.yaml \
  embeddings.current=my_data \
  dataset.current=my_data
```

You can override any training setting via Hydra (e.g., `experiment.num_gpus_per_node=4`).

### transform

Computes embeddings for a new dataset using a checkpoint. Provide either `--checkpoint` or `--model-folder`.
`--output` is required unless you write directly to LanceDB.

```bash
state emb transform \
  --model-folder /path/to/SE-600M \
  --checkpoint /path/to/SE-600M/se600m_epoch15.ckpt \
  --input /path/to/input.h5ad \
  --output /path/to/output.h5ad
```

If `--output` ends with `.npy`, only the embeddings matrix is written (no .h5ad is saved).
Gene names are auto-detected from `var` (or `var.index`) based on overlap with the model's embeddings.

### Vector Database (optional)

Install optional dependencies:

```bash
uv tool install ".[vectordb]"
```

If working off a previous installation, you may need:

```bash
uv sync --extra vectordb
```

Build a vector database:

```bash
state emb transform \
  --model-folder /path/to/SE-600M \
  --input /path/to/dataset.h5ad \
  --lancedb tmp/state_embeddings.lancedb
```

Query the database:

```bash
state emb query \
  --lancedb tmp/state_embeddings.lancedb \
  --input /path/to/query.h5ad \
  --output tmp/similar_cells.csv \
  --k 3
```

## Singularity

Containerization for STATE is available via the `singularity.def` file.

Build the container:

```bash
singularity build state.sif singularity.def
```

Run the container:

```bash
singularity run state.sif --help
```

Example run of `state emb transform`:

```bash
singularity run --nv -B /large_storage:/large_storage \
  state.sif emb transform \
    --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
    --checkpoint /large_storage/ctc/userspace/aadduri/SE-600M/se600m_epoch15.ckpt \
    --input /large_storage/ctc/datasets/replogle/rpe1_raw_singlecell_01.h5ad \
    --output test_output.h5ad
```



## Licenses
State code is [licensed](LICENSE) under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).

The model weights and output are licensed under the [Arc Research Institute State Model Non-Commercial License](MODEL_LICENSE.md) and subject to the [Arc Research Institute State Model Acceptable Use Policy](MODEL_ACCEPTABLE_USE_POLICY.md).

Any publication that uses this source code or model parameters should cite the State [paper](https://arcinstitute.org/manuscripts/State).
