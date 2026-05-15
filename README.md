# Student Performance Prediction (ML + Streamlit)

This project trains a regression model to predict a student’s **Grade** from tabular features (age, study hours, attendance, etc.) and exposes predictions through a **Streamlit** web app.

## What is included

### 1) Streamlit app (prediction UI)
- File: `app.py`
- Lets the user enter feature values.
- Converts inputs to a 1-row DataFrame.
- Loads persisted artifacts:
  - `artifacts/preprocessor.pkl`
  - `artifacts/model.pkl`
- Outputs the predicted numeric grade.

### 2) Training pipeline (offline training)
Training is split into 3 steps:

1. **Data ingestion**
   - File: `src/components/data_ingestion.py`
   - Reads dataset: `data/Students Performance .csv`
   - Splits into train/test and saves:
     - `artifacts/raw.csv`
     - `artifacts/train.csv`
     - `artifacts/test.csv`

2. **Data transformation (preprocessing)**
   - File: `src/components/data_transformation.py`
   - Builds a preprocessing pipeline using:
     - Numerical features: imputation + `StandardScaler`
     - Categorical features: imputation + `OneHotEncoder` + `StandardScaler(with_mean=False)`
   - Converts target `Grade` from letter grades to numeric values using `grade_map`.
   - Fits preprocessing on train and transforms both train/test.
   - Saves preprocessor to:
     - `artifacts/preprocessor.pkl`

3. **Model training (regression)**
   - File: `src/components/model_trainer.py`
   - Trains multiple regressors and evaluates regression metrics:
     - MAE, MSE, RMSE, R2
     - Cross-validated MAE and RMSE (CV)
   - Selects the best model by lowest CV_RMSE.
   - Saves the best model to:
     - `artifacts/model.pkl`

### 3) Prediction pipeline (used by Streamlit)
- File: `src/pipeline/prediction_pipeline.py`
- Loads persisted artifacts and performs:
  - Feature column validation + correct column ordering
  - Preprocessor transform
  - Model prediction
- Also includes a safeguard to avoid crashes when the user provides categorical values not seen during training (OneHotEncoder “unknown” categories).

### 4) Training entrypoint script
- File: `src/pipeline/training_pipeline.py`
- Re-exports `CustomData` and `PredictPipeline` used by `app.py`.
- Also contains a runnable `__main__` section to generate artifacts from scratch:
  - ingest → transform → train

## Step-by-step: how to run the project

### Step 0: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 1: Train the model (generate artifacts)
Run:
```bash
python src/pipeline/training_pipeline.py
```
This will create/update:
- `artifacts/preprocessor.pkl`
- `artifacts/model.pkl`
- `artifacts/train.csv`, `artifacts/test.csv`, `artifacts/raw.csv`

### Step 2: Start the Streamlit app
Run:
```bash
streamlit run app.py
```
Then open the displayed local URL (usually `http://localhost:8501`).

### Step 3: Make a prediction
- Enter values in the form.
- Click **Predict**.
- The app prints the predicted numeric **Grade**.

## Project structure
- `app.py` — Streamlit UI
- `src/components/` — ingestion, transformation, training
- `src/pipeline/` — prediction pipeline and training entrypoint
- `artifacts/` — saved preprocessing + trained model
- `data/` — original dataset
- `logs/` — logs created by `src/logger.py`

## Notes / Known considerations
- The `Grade` target is treated as **regression** by mapping letter grades to numeric values.
- Predictions depend on the exact feature schema expected by the fitted preprocessor.
- Persisted artifacts should be used with compatible `scikit-learn` versions to avoid transformation issues.

