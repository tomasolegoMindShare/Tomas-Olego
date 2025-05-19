# %% Cell 1: Imports and Environment Setup
import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: enables the hist gradient boosting API
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split  # ensure split import
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Define paths from environment variables (cannot change existing env vars)
DATA_PATH = Path("/mnt/data") / os.environ["DOMINO_PROJECT_NAME"] / "WineQualityData.csv"
ARTIFACTS_DIR = Path("/mnt/artifacts")
MODEL_DIR = Path("/mnt/code/models")

logger.info(f"Data path set to: {DATA_PATH}")
assert DATA_PATH.exists(), f"Data file not found at {DATA_PATH}"


# %% Cell 2: Data Loading & Initial Inspection
from IPython.display import display

# Load dataset
logger.info("Loading data...")
df = pd.read_csv(DATA_PATH)
logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Display first few rows and summary statistics
display(df.head())
df.info()
display(df.describe())


# %% Cell 3: Preprocessing & Feature Selection
# 1. Clean column names
logger.info("Cleaning column names...")
df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]

# 2. Create binary feature for red wine
df['is_red'] = (df['type'] == 'red').astype(int)

# 3. Drop rows with missing values
logger.info("Dropping missing values...")
df = df.dropna()

# 4. Compute Pearson correlations with target on numeric columns only
target = 'quality'
corrs = df.corr(numeric_only=True)[target].drop(target)
important_feats = corrs[corrs.abs() > 0.08].index.tolist()
logger.info(f"Selected features based on correlation threshold: {important_feats}")

# 5. Define feature matrix X and target vector y
X = df[important_feats]
y = df[target].astype(float)

display(X.head())
display(y.describe())


# %% Cell 4: Build Preprocessing & Modeling Pipeline
# Define numeric features and preprocessing transformer
numeric_features = important_feats
numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features)
], remainder='drop')

# Create the modeling pipeline
model = HistGradientBoostingRegressor(random_state=42)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', model)
])

logger.info("Pipeline created successfully:")
logger.info(pipeline)


# %% Cell 5: Train/Test Split, Training, and Evaluation with MLflow
# Ensure train_test_split is available
try:
    _ = train_test_split
except NameError:
    from sklearn.model_selection import train_test_split

# Ensure metrics are imported
try:
    _ = r2_score, mean_squared_error
except NameError:
    from sklearn.metrics import r2_score, mean_squared_error

# Split the data
test_size = float(os.environ.get('TEST_SIZE', 0.3))  # allow override via env var
random_state = int(os.environ.get('RANDOM_STATE', 42))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# Start MLflow run
mlflow.set_experiment(
    experiment_name=f"{os.environ.get('DOMINO_PROJECT_NAME')} {os.environ.get('DOMINO_STARTING_USERNAME')} {os.environ.get('MLFLOW_NAME')}"
)
with mlflow.start_run():
    mlflow.set_tag("Model_Type", "sklearn_histgb")
    mlflow.sklearn.autolog(silent=True, log_models=False)

    # Train the pipeline
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    logger.info("Evaluating model...")
    preds = pipeline.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    logger.info(f"R2 Score: {r2:.3f}")
    logger.info(f"Mean Squared Error: {mse:.3f}")

    # Log metrics explicitly
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mean_squared_error", mse)

    # Save metrics to dominostats.json
    dominostats = {'r2_score': round(r2, 3), 'mean_squared_error': round(mse, 3)}
    with open(ARTIFACTS_DIR / 'dominostats.json', 'w') as f:
        import json
        json.dump(dominostats, f)

    # Optional: Log and register model with signature and input example
    from mlflow.models.signature import infer_signature
    # Infer model signature from training data
    signature = infer_signature(X_train, pipeline.predict(X_train))
    input_example = X_train.head(5)
    mlflow.sklearn.log_model(
        pipeline,
        "histgb_pipeline_model",
        signature=signature,
        input_example=input_example
    )

logger.info("Training and evaluation complete.")

# %% Cell 6: Serialize Final Model
import joblib

model_file = MODEL_DIR / 'histgb_pipeline_model.joblib'
joblib.dump(pipeline, model_file)
logger.info(f"Model serialized to {model_file}")