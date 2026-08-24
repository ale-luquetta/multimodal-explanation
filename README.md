# Multimodal

Unimodal (vision-only) and multimodal (vision + text) training and explainability research on the Rico dataset.

## Setup

```bash
# create and activate env (conda)
conda create -n multimodal python=3.11
conda activate multimodal

# install dependencies
pip install -r requirements.txt

# install this project as an editable package
pip install -e .
```

## Project layout

```
multimodal/
├── configs/                       # YAML experiment configs
├── data/                          # data artifacts (gitignored)
│   ├── datasets/rico/
│   ├── reviews/
│   ├── reviews_processed/
│   └── embeddings_cache/
├── models/                        # saved checkpoints (gitignored)
│   ├── unimodal/
│   └── multimodal/
├── results/
│   ├── training/{unimodal,multimodal}/
│   └── explanations/{unimodal,multimodal}/
└── src/multimodal/                # installable package
    ├── common/                    # shared utilities
    ├── data/                      # datasets and preprocessing
    ├── training/                  # training entrypoints
    └── explainability/            # XAI entrypoints
```

## Usage

Unified menu (to be implemented):

```bash
python main.py
```
