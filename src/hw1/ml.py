"""Machine learning helper wrappers (placeholders) for Problem 2.

Provides unified interface to fit penalized linear models and a simple neural net
using scikit-learn. Real implementation would handle train/test splits, CV, and
hyperparameter selection.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

try:  # Optional dependency handling so imports don't break minimal environments
    from sklearn.linear_model import Lasso, Ridge, ElasticNet
    from sklearn.neural_network import MLPRegressor
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - environment without sklearn
    Lasso = Ridge = ElasticNet = MLPRegressor = object  # type: ignore
    SKLEARN_AVAILABLE = False

@dataclass
class ModelResult:
    name: str
    params: Dict[str, Any]
    in_sample_r2: float
    n_obs: int

__all__ = ["fit_lasso", "fit_ridge", "fit_elastic_net", "fit_mlp", "ModelResult", "SKLEARN_AVAILABLE"]


def _prepare_xy(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not in DataFrame")
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    return X, y


def _check_sklearn():  # pragma: no cover - trivial
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not installed. Install dependencies in requirements.txt to enable ML features.")


def fit_lasso(df: pd.DataFrame, target_col: str = "y", alpha: float = 0.001) -> ModelResult:
    _check_sklearn()
    X, y = _prepare_xy(df, target_col)
    model = Lasso(alpha=alpha, max_iter=10_000, random_state=0)
    model.fit(X, y)
    r2 = model.score(X, y)
    return ModelResult("lasso", {"alpha": alpha}, r2, len(y))


def fit_ridge(df: pd.DataFrame, target_col: str = "y", alpha: float = 1.0) -> ModelResult:
    _check_sklearn()
    X, y = _prepare_xy(df, target_col)
    model = Ridge(alpha=alpha, random_state=0)
    model.fit(X, y)
    r2 = model.score(X, y)
    return ModelResult("ridge", {"alpha": alpha}, r2, len(y))


def fit_elastic_net(df: pd.DataFrame, target_col: str = "y", alpha: float = 0.001, l1_ratio: float = 0.5) -> ModelResult:
    _check_sklearn()
    X, y = _prepare_xy(df, target_col)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10_000, random_state=0)
    model.fit(X, y)
    r2 = model.score(X, y)
    return ModelResult("elastic_net", {"alpha": alpha, "l1_ratio": l1_ratio}, r2, len(y))


def fit_mlp(df: pd.DataFrame, target_col: str = "y", hidden_layer_sizes=(32,), max_iter: int = 200) -> ModelResult:
    _check_sklearn()
    X, y = _prepare_xy(df, target_col)
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation="relu", solver="adam",
                         random_state=0, max_iter=max_iter)
    model.fit(X, y)
    r2 = model.score(X, y)
    return ModelResult("mlp", {"hidden_layer_sizes": hidden_layer_sizes, "max_iter": max_iter}, r2, len(y))
