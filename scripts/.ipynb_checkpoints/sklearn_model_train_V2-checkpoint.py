# %% Cell 1: Imports, Environment Setup & MLflow Autolog
import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

def cast_to_float64(X):
    """Cast any array or DataFrame of numerics to float64."""
    return X.astype(np.float64)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Define paths from environment variables (cannot change existing env vars)
DATA_PATH     = Path("/mnt/data") / os.environ.get("DOMINO_PROJECT_NAME", "") / "WineQualityData.csv"
ARTIFACTS_DIR = Path("/mnt/artifacts")
MODEL_DIR     = Path("/mnt/code/models")

logger.info(f"Data path set to: {DATA_PATH}")
assert DATA_PATH.exists(), f"Data file not found at {DATA_PATH}"

# Configure MLflow experiment & autolog (no model logging to avoid duplicates)
mlflow.set_experiment(
    experiment_name=(
        f"{os.environ.get('DOMINO_PROJECT_NAME','')} "
        f"{os.environ.get('DOMINO_STARTING_USERNAME','')} "
        f"{os.environ.get('MLFLOW_NAME','')}"
    )
)
mlflow.sklearn.autolog(log_models=False)

# %% Cell 2: Load Data & Initial Inspection
logger.info(f"Reading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

# %% Cell 3: Preprocessing & Feature Selection
# 1️⃣ Clean & standardize column names
logger.info("Cleaning column names...")
df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]

# 2️⃣ Create binary “is_red” feature
logger.info("Creating 'is_red' feature...")
df['is_red'] = (df['type'] == 'red').astype(int)

# 3️⃣ Drop any rows with missing values
logger.info("Dropping missing values...")
df.dropna(inplace=True)

# 4️⃣ Compute Pearson correlations on numeric columns
target = 'quality'
corrs = df.corr(numeric_only=True)[target].drop(target)

# 5️⃣ Select features with |corr| > 0.08, sorted by absolute correlation
important_feats = (
    corrs
    .abs()
    .sort_values(ascending=False)
    .loc[lambda s: s > 0.08]
    .index
    .tolist()
)
logger.info(f"Selected features based on |corr|>0.08: {important_feats}")

# 6️⃣ Define feature matrix X and target vector y
X = df[important_feats]
y = df[target].astype(float)

# 7️⃣ Quick inspection
logger.info("Feature selection complete. X.shape=%s, y.shape=%s", X.shape, y.shape)

# %% Cell 4: Build Preprocessing & Modeling Pipeline
# 1️⃣ No-op transformer for casting inputs to float64
cast_to_float = FunctionTransformer(cast_to_float64, validate=False)

# 2️⃣ Numeric features selected in Cell 3
numeric_features = important_feats

# 3️⃣ Preprocessing: scale numeric features
numeric_transformer = Pipeline([('scaler', StandardScaler())])
preprocessor = ColumnTransformer(
    [('num', numeric_transformer, numeric_features)],
    remainder='drop'
)

# 4️⃣ Model with early stopping to speed up training and guard against overfitting
model = HistGradientBoostingRegressor(
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)

# 5️⃣ Assemble the full pipeline
pipeline = Pipeline([
    ('cast_to_float', cast_to_float),
    ('preprocessor',   preprocessor),
    ('regressor',      model)
])

logger.info("Pipeline created successfully")
logger.info(pipeline)

# %% Cell 5: Train/Test Split, Training, MLflow Logging & PyFunc Wrapping
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec
import joblib

# 1️⃣ Split the data
test_size    = float(os.environ.get('TEST_SIZE',    0.3))
random_state = int(os.environ.get('RANDOM_STATE',  42))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# 2️⃣ Define the CastAndPredict wrapper (picklable top‐level class)
class CastAndPredictModel(PythonModel):
    def load_context(self, context):
        # load the sklearn pipeline you just logged earlier in this run
        self.pipeline = mlflow.sklearn.load_model(context.artifacts["sk_model"])
    def predict(self, context, model_input):
        # cast all incoming columns to float64 so MLflow schema check passes
        return self.pipeline.predict(model_input.astype("float64"))

# 3️⃣ Start the one MLflow run (autolog already set up in Cell 1)
with mlflow.start_run() as run:
    mlflow.set_tag("Model_Type", "sklearn_histgb")

    # — train & eval as before —
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    r2   = r2_score(y_test, preds)
    mse  = mean_squared_error(y_test, preds)
    mlflow.log_metrics({"r2_score": r2, "mean_squared_error": mse})

    with open(ARTIFACTS_DIR / 'dominostats.json', 'w') as f:
        json.dump({"r2_score": round(r2,3), "mean_squared_error": round(mse,3)}, f)

    # — 4️⃣ Log the sklearn pipeline as before —
    signature     = infer_signature(X_train, pipeline.predict(X_train))
    input_example = X_train.head(5).astype("float64")
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="histgb_pipeline_model",
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "scikit-learn>=1.0.0",
            "pandas>=1.0.0",
            "cloudpickle>=2.0.0",
            "mlflow>=2.0.0"
        ]
    )

    # — 5️⃣ **Now** log the PyFunc wrapper in the _same_ run —
    sklearn_artifact_uri = f"runs:/{run.info.run_id}/histgb_pipeline_model"
    mlflow.pyfunc.log_model(
        artifact_path="wine_quality_pyfunc",
        python_model=CastAndPredictModel(),
        artifacts={"sk_model": sklearn_artifact_uri},
        signature=ModelSignature(
            inputs=Schema([ColSpec("double", name=f) for f in important_feats]),
            outputs=Schema([ColSpec("double")])
        ),
        # you can omit input_example here
    )

    # — 6️⃣ Register the PyFunc flavor in the registry (outside the run, if you like) —
    pyfunc_uri = f"runs:/{run.info.run_id}/wine_quality_pyfunc"
    model_name = os.environ.get("MLFLOW_MODEL_NAME", "WineQualityModel")
    mlflow.register_model(pyfunc_uri, model_name)

# when this `with` block exits, MLflow will automatically end the run.
logger.info("All artifacts logged and model registered in a single MLflow run.")

# %% Cell 6: Serialize Final Model
import joblib
model_file = MODEL_DIR / 'histgb_pipeline_model.joblib'
joblib.dump(pipeline, model_file)
logger.info(f"Model serialized to {model_file}")

