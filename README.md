# Credit Scoring Project

A machine learning pipeline for credit scoring using **Logistic Regression** with **Weight of Evidence (WoE)** transformation via `scorecardpy`. The model predicts whether a customer has a **Poor** credit score (binary classification).

---

## Project Structure

```
credit-scoring-Project/
├── config/             # Configuration files
├── data/               # Raw dataset
├── src/mlProject/
│   ├── components/     # Core logic (ingestion, validation, preprocessing, training, evaluation)
│   ├── config/         # Configuration manager
│   ├── constants/      # Project-wide constants
│   ├── entity/         # Dataclasses for configs and artifacts
│   ├── pipeline/       # Stage pipelines
│   └── utils/          # Utility functions
├── artifacts/          # Generated outputs (auto-created)
├── main.py             # Entry point
├── params.yaml         # Model hyperparameters
├── schema.yaml         # Dataset schema
└── config/config.yaml  # Path configurations
```

---

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| 01 | **Data Ingestion** — Copy raw CSV to artifacts |
| 02 | **Data Validation** — Validate columns against schema |
| 03 | **Data Preprocessing** — Clean, WoE transform, train/test split |
| 04 | **Model Trainer** — Train Logistic Regression + generate scorecard |
| 05 | **Model Evaluation** — AUC, KS, Gini, PSI, IV summary |

---

## Requirements

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

Key libraries: `scorecardpy`, `scikit-learn`, `pandas`, `numpy`, `joblib`

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/credit-scoring-Project.git
cd credit-scoring-Project
```

**2. Create and activate virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Place your dataset**

Put your CSV file at:
```
data/credit-score.csv
```

**5. Run the full pipeline**
```bash
python main.py
```

All outputs will be saved under the `artifacts/` directory.

---

## Output Artifacts

```
artifacts/
├── data_ingestion/         # Copied raw data
├── data_validation/        # Validation status report
├── data_preprocessing/     # WoE-transformed train/test + raw splits + bins
├── model_trainer/          # Trained model, scorecard, feature names
└── model_evaluation/       # Metrics JSON + evaluation report CSV
```

---

## Evaluation Results (Example)

| Metric | Train | Test |
|--------|-------|------|
| AUC | 0.8253 | 0.8245 |
| KS | 0.5537 | 0.5454 |
| Gini | 0.6506 | 0.6490 |
| Overfit Gap | — | 0.0008 |

---

## Notes

- Delete the `artifacts/` folder before re-running to ensure a clean pipeline execution.
- PSI is calculated using the scorecard score distribution between train and test sets.