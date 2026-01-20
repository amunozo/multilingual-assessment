# Assessment of Pre-Trained Models Across Languages and Grammars

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Paper](https://img.shields.io/badge/ACL-Anthology-red.svg)](https://aclanthology.org/2023.ijcnlp-main.23/)

This repository contains the official implementation for the paper:
**[Assessment of Pre-Trained Models Across Languages and Grammars](https://aclanthology.org/2023.ijcnlp-main.23/)**
*Alberto Muñoz-Ortiz, David Vilares, and Carlos Gómez-Rodríguez*
Presented at **IJCNLP-AACL 2023** in Nusa Dua, Bali, Indonesia.

---

## Overview

This project evaluates the performance of various pre-trained language models (like BERT, XLM-R, and CANINE) across different languages and grammatical structures (Constituency and Dependency parsing).

## Project Structure

The repository is organized as follows:

- `src/`: Core logic and model implementations.
- `scripts/`: Entry-point scripts for training, evaluation, and plotting.
- `notebooks/`: Jupyter notebooks for data analysis and visualization.
- `data/`: Directory for storing datasets and intermediate scores.
- `results/`: Output logs and evaluation results.
- `config/`: Model and training configurations.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amunozo/multilingual-assessment.git
   cd multilingual-assessment
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To train the models for dependency parsing:
```bash
python scripts/train.py
```

### Evaluation
To evaluate the trained models:
```bash
python scripts/eval.py
```

### Plotting Results
To generate plots from the evaluation scores:
```bash
python scripts/plot.py
```

## Results

The main findings of the paper show how different subword tokenization strategies and model architectures impact the cross-lingual transferability of grammatical knowledge. For detailed results, please refer to our [paper](https://aclanthology.org/2023.ijcnlp-main.23/).

## Contact

For any questions or issues, please contact the main author:
**Alberto Muñoz-Ortiz** - [alberto.munoz.ortiz@udc.es](mailto:alberto.munoz.ortiz@udc.es)

## Citation

If you use this code or our findings in your research, please cite:

```bibtex
@inproceedings{munoz-ortiz-etal-2023-assessment,
    title = "Assessment of Pre-Trained Models Across Languages and Grammars",
    author = "Mu{\~n}oz-Ortiz, Alberto  and
      Vilares, David  and
      G{\'o}mez-Rodr{\'i}guez, Carlos",
    booktitle = "Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = nov,
    year = "2023",
    address = "Nusa Dua, Bali",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.ijcnlp-main.23",
    pages = "343--358",
}
```

## Acknowledgments

We gratefully acknowledge the support of the following organizations:
- **European Research Council (ERC)** (SALSA, grant No 101100615)
- **ERDF/MICINN-AEI** (Grant SCANNER-UDC)
- **Xunta de Galicia** (Grant ED431C 2020/11 and CITIC)
- **MCIN/AEI** (FPI 2021 grant)
