#!/usr/bin/env python3
import os
import json
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import h2o
from h2o.automl import H2OAutoML
import mlflow
import mlflow.h2o

# Optional: ensure multi-thread conversion dependencies are available
def setup_logging():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO
    )
    try:
        import pyarrow  # noqa: F401
        import polars  # noqa: F401
        logging.info("PyArrow and Polars available for multi-threaded conversion")
    except ImportError:
        logging.warning(
            "Install pyarrow and polars for faster H2OFrame to DataFrame conversion"
        )


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logging.info(f"Loaded {len(df)} rows from {path}")
    return df.dropna()


def select_features(df: pd.DataFrame, threshold: float = 0.08) -> pd.DataFrame:
    corr = df.corr(numeric_only=True)["quality"].abs().drop("quality")
    selected = corr[corr > threshold].index.tolist()
    logging.info(f"Selected {len(selected)} features with |corr| > {threshold}")
    return df[selected + ["quality"]]


def split_data(df: pd.DataFrame, train_frac: float = 0.7, seed: int = 42):
    train = df.sample(frac=train_frac, random_state=seed)
    test = df.drop(train.index)
    logging.info(f"Train size: {len(train)}, Test size: {len(test)}")
    return train, test


def init_h2o(max_mem: str = '4G', nthreads: int = -1):
    h2o.init(max_mem_size=max_mem, nthreads=nthreads)
    logging.info(f"H2O version: {h2o.__version__}")


def configure_mlflow():
    proj = os.environ.get('DOMINO_PROJECT_NAME')
    user = os.environ.get('DOMINO_STARTING_USERNAME')
    mlflow_name = os.environ.get('MLFLOW_NAME')
    exp_name = f"{proj} {user} {mlflow_name}"
    mlflow.set_experiment(experiment_name=exp_name)
    logging.info(f"MLflow experiment: {exp_name}")


def train_automl(h2o_train, h2o_test, features, target, max_models=10, max_secs=30, seed=42):
    aml = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_secs,
        sort_metric="r2",
        seed=seed
    )
    aml.train(x=features, y=target, training_frame=h2o_train)
    leader = aml.leader
    logging.info(f"AutoML leader: {leader.model_id}")
    # Use multi-threaded conversion for faster performance
    preds_df = leader.predict(h2o_test).as_data_frame(
        use_pandas=True, use_multi_thread=True
    )
    preds = preds_df['predict']
    return leader, preds


def log_metrics_and_artifacts(leader, h2o_train, output_dir: Path, test_df, preds):
    r2 = round(leader.r2(xval=True), 3)
    mse = round(leader.mse(xval=True), 3)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("MSE", mse)
    logging.info(f"Metrics -> R2: {r2}, MSE: {mse}")

    # Save Domino stats under artifacts directory
    stats_path = output_dir / 'dominostats.json'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump({"R2": r2, "MSE": mse}, f)
    logging.info(f"Dominostats saved to {stats_path}")

    # Ensure artifact directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scatter plot: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Actual vs Predictions')
    ax.scatter(test_df['quality'], preds, alpha=0.6)
    ax.plot(
        [test_df['quality'].min(), test_df['quality'].max()],
        [test_df['quality'].min(), test_df['quality'].max()],
        'r--'
    )
    scatter_path = output_dir / 'scatter.png'
    fig.savefig(scatter_path)
    mlflow.log_figure(fig, 'scatter.png')
    plt.close(fig)

    # Histogram: Actual vs Predicted distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Quality Distribution: Actual vs Predicted')
    ax.hist(test_df['quality'], bins=6, alpha=0.5, label='Actual')
    ax.hist(preds, bins=6, alpha=0.5, label='Predicted')
    ax.legend()
    hist_path = output_dir / 'histogram.png'
    fig.savefig(hist_path)
    mlflow.log_figure(fig, 'histogram.png')
    plt.close(fig)

    # Save the trained model
    model_dir = Path('/mnt/code/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    h2o.save_model(leader, str(model_dir))
    logging.info(f"Model saved to {model_dir}")


if __name__ == '__main__':
    setup_logging()

    data_path = Path(f"/mnt/data/Winequality-Workshop/WineQualityData.csv")
    df = load_data(data_path)
    df = select_features(df)
    train_df, test_df = split_data(df)

    init_h2o()
    configure_mlflow()

    h2o_train = h2o.H2OFrame(train_df)
    h2o_test = h2o.H2OFrame(test_df)
    features = [c for c in h2o_train.columns if c != 'quality']
    target = 'quality'

    with mlflow.start_run():
        mlflow.set_tag("Framework", "H2OAutoML")
        leader, preds = train_automl(h2o_train, h2o_test, features, target)
        log_metrics_and_artifacts(leader, h2o_train, Path('/mnt/artifacts'), test_df, preds)
    logging.info("Script complete!")
