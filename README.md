# Assessment of Pre-Trained Models Across Languages and Grammars

[![CI](https://github.com/amunozo/multilingual-assessment/actions/workflows/ci.yml/badge.svg)](https://github.com/amunozo/multilingual-assessment/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/ACL-Anthology-red.svg)](https://aclanthology.org/2023.ijcnlp-main.23/)

Code and analysis for **[Assessment of Pre-Trained Models Across Languages and
Grammars](https://aclanthology.org/2023.ijcnlp-main.23/)**, by Alberto
Muñoz-Ortiz, David Vilares, and Carlos Gómez-Rodríguez (IJCNLP-AACL 2023).

The experiments probe mBERT, XLM-R, CANINE-C, and CANINE-S through
sequence-labeling formulations of dependency and constituency parsing. The
repository contains the experiment code, recorded scores, analysis notebooks,
and pinned versions of the external tree encoders.

## Reproducibility scope

This is a research artifact, not a packaged parser. A full reproduction needs
the licensed PTB, CTB, and SPMRL corpora, Universal Dependencies 2.9, model
downloads, and substantial compute. Those corpora and trained checkpoints are
not redistributed here.

The control flow, data conversion, metrics, and command-line interfaces are
covered by lightweight tests. The complete experiment matrix has not been
rerun after the portability cleanup.

## Setup

Clone the external encoders and install the research dependencies:

```bash
git clone --recurse-submodules https://github.com/amunozo/multilingual-assessment.git
cd multilingual-assessment
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For an existing clone, restore the pinned dependencies with:

```bash
git submodule update --init --recursive
```

`external/tree2labels` is the exact historical Python 2 source. The three
encoding modules used by this project have a mechanically converted and
attributed Python 3 compatibility copy in `vendor/tree2labels_py3/`.
`external/dep2label` is called through its Python API, which also avoids a typo
in its historical command-line encoding choices.

## Data layout

No corpus path is hard-coded. Pass it for each run:

- `--ud-root` is the directory containing folders such as
  `UD_Basque-BDT/` and `UD_Welsh-CCG/`.
- For English, `--corpus-root` is the PTB directory containing
  `train.trees`, `dev.trees`, and `test.trees`.
- For Chinese, it is the CTB directory containing `train_ch.trees`,
  `dev_ch.trees`, and `test_ch.trees`.
- For another constituency language, it is the parent of folders such as
  `GERMAN_SPMRL/`.

Encoded inputs are written under `data/`. Prepared datasets, checkpoints,
predictions, scores, and plots go under ignored `artifacts/` by default. Set
`MULTILINGUAL_ASSESSMENT_OUTPUT_DIR` or pass `--artifact-root` to use another
location.

## Run one experiment

Dependency example:

```bash
python scripts/train.py dependency \
  --treebank UD_Basque-BDT \
  --ud-root /path/to/ud-treebanks-v2.9 \
  --encoding 2-planar-brackets-greedy \
  --model xlm-roberta-base \
  --finetuned not_finetuned \
  --pretrained pretrained \
  --device cuda:0

python scripts/eval.py dependency \
  --treebank UD_Basque-BDT \
  --ud-root /path/to/ud-treebanks-v2.9 \
  --encoding 2-planar-brackets-greedy \
  --model xlm-roberta-base \
  --finetuned not_finetuned \
  --pretrained pretrained \
  --device cuda:0
```

Constituency example:

```bash
python scripts/train.py constituency \
  --language english \
  --corpus-root /path/to/PTB \
  --model google/canine-c \
  --finetuned not_finetuned \
  --pretrained pretrained \
  --device cuda:0

python scripts/eval.py constituency \
  --language english \
  --corpus-root /path/to/PTB \
  --model google/canine-c \
  --finetuned not_finetuned \
  --pretrained pretrained \
  --device cuda:0
```

Use `--device cpu` for CPU execution or `--device auto` to let the Trainer
choose. Every command runs one explicit configuration; experiment matrices can
be assembled in a shell scheduler without editing Python source.

## Randomly initialized controls

Select the same architecture without loading its pretrained weights:

```bash
python scripts/train.py dependency \
  --treebank UD_Basque-BDT \
  --ud-root /path/to/ud-treebanks-v2.9 \
  --encoding 2-planar-brackets-greedy \
  --model xlm-roberta-base \
  --finetuned not_finetuned \
  --pretrained not_pretrained \
  --seed 13
```

## Analysis and tests

Recorded paper-era scores remain in `data/`, and the original analysis is in
`notebooks/`. Generate an error-reduction plot with:

```bash
python scripts/plot.py dependency \
  --scores data/dep_scores.csv \
  --encoding 2-planar-brackets-greedy \
  --output artifacts/plots/dependency.png
```

Run the lightweight checks with:

```bash
python -m pip install -r requirements-dev.txt
ruff check src scripts tests
pytest
```

## Citation

```bibtex
@inproceedings{munoz-ortiz-etal-2023-assessment,
    title = "Assessment of Pre-Trained Models Across Languages and Grammars",
    author = "Mu{\\~n}oz-Ortiz, Alberto and Vilares, David and G{\\'o}mez-Rodr{\\'i}guez, Carlos",
    booktitle = "Proceedings of IJCNLP-AACL 2023 (Volume 1: Long Papers)",
    year = "2023",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.ijcnlp-main.23/",
    pages = "343--358"
}
```

## License

The project code is available under the [MIT License](LICENSE). Third-party
code retains its original license; see the files under `external/` and
`vendor/tree2labels_py3/`.
