"""Train NFL MOV + totals LightGBM regressors and convert predictions to
probabilities via the shared services/ml/probability.py helpers."""
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

logger = structlog.get_logger()

_PARAMS = {
    "objective": "regression", "metric": "rmse", "num_leaves": 31,
    "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 5, "min_data_in_leaf": 30, "verbose": -1,
}


def train_regressor(frame: pd.DataFrame, feature_cols: list[str],
                    target_col: str, seed: int = 42):
    X = frame[feature_cols].fillna(0.0)
    y = frame[target_col].astype(float)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(_PARAMS, dtrain, num_boost_round=2000,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    resid = y_val.values - model.predict(X_val, num_iteration=model.best_iteration)
    return model, float(np.std(resid))


def save_bundle(path: str, model, feature_cols: list[str], resid_std: float,
                trained_seasons: list[int]) -> None:
    joblib.dump({"model": model, "feature_cols": feature_cols,
                 "resid_std": resid_std, "trained_seasons": list(trained_seasons)}, path)
    logger.info("nfl_bundle_saved", path=path, resid_std=round(resid_std, 3))


def load_bundle(path: str) -> dict:
    return joblib.load(path)


def predict_mov(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    X = frame[bundle["feature_cols"]].fillna(0.0)
    model = bundle["model"]
    it = getattr(model, "best_iteration", None)
    return model.predict(X, num_iteration=it)
