#!/usr/bin/env Rscript

## ─── Load Libraries ───────────────────────────────────────────────────────────
library(tidyverse)    # readr, dplyr, tidyr, ggplot2, etc.
library(vctrs)        # for vec_cast()
library(rsample)      # data splitting
library(recipes)      # preprocessing
library(parsnip)      # model specification
library(workflows)    # glue recipe + model
library(yardstick)    # performance metrics
library(mlflow)       # experiment tracking
library(glue)         # string interpolation
library(fs)           # file‐path construction
library(jsonlite)     # JSON output

## ─── Environment Variables ────────────────────────────────────────────────────
project_name <- Sys.getenv("DOMINO_PROJECT_NAME")
user_name    <- Sys.getenv("DOMINO_STARTING_USERNAME")
mlflow_name  <- Sys.getenv("MLFLOW_NAME")

## ─── MLflow Experiment Setup ─────────────────────────────────────────────────
mlflow_set_experiment(
  experiment_name = glue("{project_name} {user_name} {mlflow_name}")
)

## ─── Data Ingestion ───────────────────────────────────────────────────────────
data_path  <- fs::path("/mnt/data", project_name, "WineQualityData.csv")
message("Reading data from ", data_path)
wine_data  <- read_csv(data_path, show_col_types = FALSE)

## ─── Feature Engineering ──────────────────────────────────────────────────────
wine_data <- wine_data %>%
  mutate(
    # explicit vctrs cast avoids rlang::as_integer() deprecation
    is_red = vec_cast(type != "white", integer())
  ) %>%
  select(-type) %>%
  drop_na()

## ─── Train/Test Split ─────────────────────────────────────────────────────────
set.seed(123)
split      <- initial_split(wine_data, prop = 0.75)
train_data <- training(split)
test_data  <- testing(split)

## ─── Preprocessing Recipe ────────────────────────────────────────────────────
rec <- recipe(quality ~ ., data = train_data) %>%
  step_rm(id) %>%             # drop id column if present
  step_normalize(all_predictors())

## ─── Model Specification & Workflow ──────────────────────────────────────────
lm_spec <- linear_reg() %>%
  set_engine("lm")

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(lm_spec)

## ─── Training, Evaluation & MLflow Logging ────────────────────────────────────
with(mlflow_start_run(), {
  mlflow_set_tag("Model_Type", "R")
  message("Training linear regression model…")
  fit_wf <- fit(wf, data = train_data)
  
  ## Predictions on test set
  preds <- predict(fit_wf, test_data) %>%
    bind_cols(test_data %>% select(quality))
  
  ## Compute performance metrics (RMSE, R²) and derive MSE
  perf <- preds %>%
    metrics(truth = quality, estimate = .pred) %>%
    select(.metric, .estimate) %>%
    pivot_wider(names_from = .metric, values_from = .estimate) %>%
    mutate(mse = rmse^2)
  
  ## Log metrics
  mlflow_log_metric("R2",  perf$rsq)
  mlflow_log_metric("RMSE", perf$rmse)
  mlflow_log_metric("MSE",  perf$mse)
  
  ## Save & log the trained workflow
  model_out <- "/mnt/code/models/r_linear_model.rds"
  saveRDS(fit_wf, file = model_out)
  mlflow_log_artifact(model_out, artifact_path = "models")
  
  ## Write out diagnostics for downstream use
  write_json(
    perf %>% select(rsq, mse) %>% deframe() %>% as.list(),
    path         = "/mnt/artifacts/dominostats.json",
    auto_unbox   = TRUE,
    pretty       = TRUE
  )
})