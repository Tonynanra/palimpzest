## Note About Datasets Used in Evaluation
Enron is run using the `enron-eval` dataset

Real Estate is run using the `real-estate-eval` dataset

For the easy and hard code generation evaluations, we needed to create a range of dataset sizes based on the `real-estate-eval` dataset. Thus, I created `real-estate-eval-5`, `real-estate-eval-10`, ..., `real-estate-eval-30`. Note that `real-estate-eval-15` should be equivalent to `real-estate-eval`.

Groundtruth labels are stored in the `groundtruth` folder. The Enron and Real Estate groundtruth files match the evaluation directories of the same name. The codegen groundtruths are slightly different than the criteria used for Real Estate, so they have their own set of labels in e.g. `codegen-easy-eval-[5,30].csv` which map(s) to `real-estate-eval-[5,30]`.  

## CUAD

All reusable CUAD data is stored under `testdata/cuad/`:

```text
cuad/
├── raw/                         # Downloaded archive, reference script, and raw JSON
├── chunks/                      # Chunked train/test JSON
└── sft/
    ├── train-dev/
    │   ├── train.jsonl
    │   ├── dev.jsonl
    │   └── manifest.json
    └── test/
        ├── test.jsonl
        └── manifest.json
```

These directories are Git-ignored. Dataset manifests stay with their bundles.
Training reads `sft/train-dev`; standalone evaluation reads the isolated `sft/test`
holdout. Models, checkpoints, logs, metrics, and plots are stored under
`finetuning/artifacts/`.

To create missing datasets, run these commands from the repository root:

```bash
./.venv/bin/python finetuning/setup_cuad_data.py
./.venv/bin/python finetuning/generate_cuad_chunk_dataset.py
./.venv/bin/python finetuning/cuad_sft_data.py --split train
./.venv/bin/python finetuning/cuad_sft_data.py --split test
```

The SFT output defaults depend on `--split`: `train` writes to `sft/train-dev`,
and `test` writes to `sft/test`. Explicit directory options override the defaults.
The Slurm workflow reuses existing non-empty SFT JSONL files and builds only
missing bundles.

Finetuning loads and generates data through `finetuning/cuad_data.py`; category
definitions live in `finetuning/cuad_schema.py`, and SFT views and manifests in
`finetuning/cuad_sft_data.py`. Labels are aggregated directly from each title's
ordered annotations. The JSON and JSONL stages above, sampling, and dataset
contents are unchanged. Historical research scripts remain separate and are not
needed for this workflow.

### Relocation of existing datasets

The existing datasets were moved without regenerating their contents:

| Previous directory | Current directory |
|---|---|
| `testdata/cuad-data/` | `testdata/cuad/raw/` |
| `testdata/cuad-chunk/` | `testdata/cuad/chunks/` |
| `finetuning/artifacts/cuad-sft/` | `testdata/cuad/sft/train-dev/` |
| `finetuning/artifacts/cuad-sft-test/` | `testdata/cuad/sft/test/` |

Only the raw/chunk directory references in dataset manifests were updated.
Historical run manifests and logs retain the paths used at run time; use this
mapping when locating their input datasets. The relocation preserves dataset
rows, source hashes, seeds, and split assignments.
