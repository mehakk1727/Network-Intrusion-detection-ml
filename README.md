Network Intrusion Detection with Machine Learning

Project Overview

This project uses Machine Learning techniques to detect network intrusions using the CICIDS2017 dataset. It classifies network traffic as Benign or Malicious.

Dataset

CICIDS2017 – Canadian Institute for Cybersecurity (UNB).

Official Source: https://www.unb.ca/cic/datasets/ids-2017.html

Dataset CSV files must be downloaded separately and placed inside the data/ folder.

Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

Machine Learning Models

1. Logistic Regression
2. Decision Tree
3. MLP Neural Network

Project Structure

├── data/
├── notebooks/
├── results/
├── src/
│   ├── _init_.py
│   ├── data_utils.py
│   ├── eda.py
│   └── train.py
├── .gitignore
├── requirements.txt
└── README.md

Installation

pip install -r requirements.txt

Run the Project

Exploratory Data Analysis:

python -m src.eda --data-dir data --output-dir results

Model Training:

python -m src.train --data-dir data --output-dir results

Results

The project generates:

* Label distribution graph
* Model comparison graph
* Confusion matrices
* Model performance metrics (CSV)

Conclusion

This project demonstrates the application of Machine Learning in network intrusion detection by analysing network traffic and comparing different classification models.

Reference

Canadian Institute for Cybersecurity – CICIDS2017 Dataset.