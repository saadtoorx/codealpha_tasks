# Credit Scoring Model

A machine learning model to predict customer creditworthiness.

## Project Structure

```
credit-scoring/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── main.ipynb
├── artifacts/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── src/
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python src/train.py
```

### Prediction

```bash
python src/predict.py
```

## Model

- **Algorithm**: Random Forest Classifier
- **Target**: creditworthy (1 = Good/Standard, 0 = Poor)

## Author

CodeAlpha Internship - Task 1
