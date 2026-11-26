# Network Intrusion Detection with Machine Learning

An academic project for exploring network-flow traffic and classifying benign versus potentially malicious connections using the CICIDS2017 dataset.

**Dataset:** CICIDS2017, Canadian Institute for Cybersecurity (UNB)  
**Official dataset page:** https://www.unb.ca/cic/datasets/ids-2017.html  
**Related reference project:** https://github.com/Keshav-Kathuria/ai-cybersecurity-unit1-mini-project-keshav

> This repository contains an independently organised implementation of the same project theme. The related project is credited as background/reference material. Before reusing any code or assets from it, review its licence and retain all notices required by that licence.

## Objectives

- Explore network-flow records and class distribution.
- Clean missing, infinite, duplicate, and non-numeric values.
- Build preprocessing pipelines fitted only on training data.
- Compare Logistic Regression, Decision Tree, and a Multi-Layer Perceptron (MLP).
- Evaluate precision, recall, F1, ROC-AUC, and confusion matrices.
- Discuss class imbalance, rare attack types, leakage, and limitations.

## Project structure

```text
.
├── data/                         # Put CICIDS2017 CSV files here
├── notebooks/
│   └── cyber_threat_classification.ipynb
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── eda.py
│   └── train.py
├── results/                      # Generated reports and plots
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset setup

Download CICIDS2017 from the [official UNB dataset page](https://www.unb.ca/cic/datasets/ids-2017.html). Extract its CSV flow files into `data/`. The scripts expect a target column named `Label`. Dataset files are not included in this ZIP because they are large and subject to the dataset provider's distribution terms.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Exploratory analysis

From the project root:

```bash
python -m src.eda --data-dir data --output-dir results
```

This reports dataset dimensions, label counts, missing values, infinite values, duplicate rows, and constant columns. It also saves a label-distribution plot.

## Train and compare models

```bash
python -m src.train --data-dir data --output-dir results --max-rows 200000
```

Use `--max-rows 0` to use every available row (this may require substantial memory). The training pipeline:

1. Maps `BENIGN` to class 0 and other labels to class 1.
2. Removes invalid target rows and non-finite feature values.
3. Uses a stratified train/test split.
4. Fits imputation and scaling on training data only.
5. Trains Logistic Regression, Decision Tree, and MLP models.
6. Reports accuracy, precision, recall, F1, ROC-AUC (when defined), and training time.
7. Saves a metrics CSV and plots in `results/`.

Metrics are produced when you run the code on your downloaded dataset; no performance values are claimed in advance.

## Notebook

Open `notebooks/cyber_threat_classification.ipynb` in Jupyter and run cells from top to bottom.

```bash
jupyter notebook
```

## Interpretation and limitations

- CICIDS2017 is highly imbalanced. Accuracy alone can hide weak detection of rare attacks; inspect precision, recall, F1, and per-class results.
- A random flow-level split can place very similar flows in training and test sets, potentially overestimating generalisation.
- Dataset artifacts and highly predictive fields can act as shortcuts. Consider feature ablations and time/group-based splits before making deployment claims.
- This is an educational offline classifier, not a production intrusion-prevention system. Do not use predictions as the sole basis for security decisions.

## References

1. I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” *ICISSP*, 2018.
2. Canadian Institute for Cybersecurity, University of New Brunswick, [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html).
3. Related project for background and comparison: [Keshav-Kathuria/ai-cybersecurity-unit1-mini-project-keshav](https://github.com/Keshav-Kathuria/ai-cybersecurity-unit1-mini-project-keshav).
