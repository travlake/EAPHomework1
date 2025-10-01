"""
Solution for problem 2.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import warnings

# Suppress convergence warnings for this script
warnings.filterwarnings('ignore', category=UserWarning)


# Ensure 'src' directory is on path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hw1.data import load_market_returns
from hw1.paths import PROCESSED_DATA

def create_features(returns: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Create features and target variable for prediction.

    For LASSO, Ridge, Elastic Net:
    - Past 5 daily returns individually.
    - Sum of past returns over windows: 10, 21, 42, ..., 252 days.

    For Neural Network:
    - All 252 past daily returns.
    """
    # Target variable is the next day's return
    y = returns.shift(-1)

    # --- Features for Linear Models ---
    features_linear = pd.DataFrame(index=returns.index)
    # Past 5 days of returns
    for i in range(1, 6):
        features_linear[f'ret_lag_{i}'] = returns.shift(i)

    # Sum of past returns over various windows
    windows = [10, 21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 231, 252]
    for w in windows:
        features_linear[f'ret_sum_{w}'] = returns.rolling(window=w).sum().shift(1)

    # --- Features for Neural Network ---
    lags = range(1, 253)
    features_nn = pd.concat([returns.shift(l) for l in lags], axis=1)
    features_nn.columns = [f'ret_lag_{l}' for l in lags]


    # Combine and drop NaNs created by lagging/rolling
    # Also drop NaNs from y from the shift
    combined = pd.concat([y, features_linear, features_nn], axis=1).dropna()

    y_clean = combined.iloc[:, 0]
    X_linear = combined.iloc[:, 1:features_linear.shape[1]+1]
    X_nn = combined.iloc[:, features_linear.shape[1]+1:]


    return X_linear, X_nn, y_clean


def solve_problem2():
    """
    Generates the table for problem 2.
    """
    # --- 1. Load and Prepare Data ---
    print("Loading data...")
    daily_returns = load_market_returns(kind="value")

    # Create features for the different models
    X_linear, X_nn, y = create_features(daily_returns)

    # --- 2. Sample Split ---
    # First half for CV, second half for testing
    split_idx = len(y) // 2

    # Linear model features
    X_linear_cv = X_linear.iloc[:split_idx]
    y_cv = y.iloc[:split_idx]
    X_linear_test = X_linear.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # NN features
    X_nn_cv = X_nn.iloc[:split_idx]
    X_nn_test = X_nn.iloc[split_idx:]

    # --- 3. Normalization ---
    # Normalize returns using mu and sigma from the 1st half of the sample
    mu = y_cv.mean()
    sigma = y_cv.std()

    y_cv_norm = (y_cv - mu) / sigma
    y_test_norm = (y_test - mu) / sigma

    # Scale features based on the training set (CV part)
    scaler_linear = StandardScaler().fit(X_linear_cv)
    X_linear_cv_norm = scaler_linear.transform(X_linear_cv)
    X_linear_test_norm = scaler_linear.transform(X_linear_test)

    scaler_nn = StandardScaler().fit(X_nn_cv)
    X_nn_cv_norm = scaler_nn.transform(X_nn_cv)
    X_nn_test_norm = scaler_nn.transform(X_nn_test)

    # --- 4. Model Evaluation ---
    results = []

    # Define chronological k-fold cross-validation with a gap
    # n_splits determines the number of folds
    # A gap of 1 day is left between train and test sets in each fold
    tscv = TimeSeriesSplit(n_splits=5, gap=1)

    # --- Random Walk ---
    print("Evaluating Random Walk...")
    rw_forecasts = y_test.expanding().mean().shift(1).dropna()
    rw_actuals = y_test.loc[rw_forecasts.index]
    rw_mse = mean_squared_error(rw_actuals, rw_forecasts)
    rw_mae = mean_absolute_error(rw_actuals, rw_forecasts)
    results.append({'Model': 'Random Walk', 'Hyperparams': '', 'MSE': rw_mse, 'MAE': rw_mae, '$R^2$': ''})

    # --- LASSO ---
    print("Evaluating LASSO...")
    lasso_cv = LassoCV(cv=tscv, n_jobs=-1, random_state=42).fit(X_linear_cv_norm, y_cv_norm)
    lasso_alpha = lasso_cv.alpha_

    # Re-train sequentially on the test set
    lasso_preds_norm = []
    for i in tqdm(range(len(X_linear_test_norm))):
        # Train on all data up to the current point in the test set
        train_X = np.vstack([X_linear_cv_norm, X_linear_test_norm[:i]])
        train_y = np.hstack([y_cv_norm, y_test_norm[:i]])
        model = Lasso(alpha=lasso_alpha, random_state=42).fit(train_X, train_y)
        pred = model.predict(X_linear_test_norm[i].reshape(1, -1))
        lasso_preds_norm.append(pred[0])

    lasso_preds = (np.array(lasso_preds_norm) * sigma) + mu # De-normalize
    lasso_mse = mean_squared_error(y_test, lasso_preds)
    lasso_mae = mean_absolute_error(y_test, lasso_preds)
    lasso_r2 = 1 - (lasso_mse / rw_mse)
    results.append({'Model': 'LASSO', 'Hyperparams': f'alpha={lasso_alpha:.4f}', 'MSE': lasso_mse, 'MAE': lasso_mae, '$R^2$': lasso_r2})

    # --- Ridge ---
    print("Evaluating Ridge...")
    ridge_cv = RidgeCV(cv=tscv, alphas=np.logspace(-6, 6, 100)).fit(X_linear_cv_norm, y_cv_norm)
    ridge_alpha = ridge_cv.alpha_

    ridge_preds_norm = []
    for i in tqdm(range(len(X_linear_test_norm))):
        train_X = np.vstack([X_linear_cv_norm, X_linear_test_norm[:i]])
        train_y = np.hstack([y_cv_norm, y_test_norm[:i]])
        model = Ridge(alpha=ridge_alpha, random_state=42).fit(train_X, train_y)
        pred = model.predict(X_linear_test_norm[i].reshape(1, -1))
        ridge_preds_norm.append(pred[0])

    ridge_preds = (np.array(ridge_preds_norm) * sigma) + mu
    ridge_mse = mean_squared_error(y_test, ridge_preds)
    ridge_mae = mean_absolute_error(y_test, ridge_preds)
    ridge_r2 = 1 - (ridge_mse / rw_mse)
    results.append({'Model': 'Ridge', 'Hyperparams': f'alpha={ridge_alpha:.4f}', 'MSE': ridge_mse, 'MAE': ridge_mae, '$R^2$': ridge_r2})

    # --- Elastic Net ---
    print("Evaluating Elastic Net...")
    enet_cv = ElasticNetCV(cv=tscv, n_jobs=-1, random_state=42).fit(X_linear_cv_norm, y_cv_norm)
    enet_alpha = enet_cv.alpha_
    enet_l1_ratio = enet_cv.l1_ratio_

    enet_preds_norm = []
    for i in tqdm(range(len(X_linear_test_norm))):
        train_X = np.vstack([X_linear_cv_norm, X_linear_test_norm[:i]])
        train_y = np.hstack([y_cv_norm, y_test_norm[:i]])
        model = ElasticNet(alpha=enet_alpha, l1_ratio=enet_l1_ratio, random_state=42).fit(train_X, train_y)
        pred = model.predict(X_linear_test_norm[i].reshape(1, -1))
        enet_preds_norm.append(pred[0])

    enet_preds = (np.array(enet_preds_norm) * sigma) + mu
    enet_mse = mean_squared_error(y_test, enet_preds)
    enet_mae = mean_absolute_error(y_test, enet_preds)
    enet_r2 = 1 - (enet_mse / rw_mse)
    results.append({'Model': 'Elastic Net', 'Hyperparams': f'alpha={enet_alpha:.4f}, l1_ratio={enet_l1_ratio:.2f}', 'MSE': enet_mse, 'MAE': enet_mae, '$R^2$': enet_r2})

    # --- Neural Network ---
    print("Evaluating Neural Network...")
    hidden_layer_sizes = [16, 32, 64, 128]
    best_nn_size = None
    best_nn_mse = float('inf')

    for size in hidden_layer_sizes:
        fold_mses = []
        for train_idx, test_idx in tscv.split(X_nn_cv_norm):
            X_train, X_test = X_nn_cv_norm[train_idx], X_nn_cv_norm[test_idx]
            y_train, y_test_fold = y_cv_norm.iloc[train_idx], y_cv_norm.iloc[test_idx]

            nn = MLPRegressor(hidden_layer_sizes=(size,), activation='relu', max_iter=500, random_state=42)
            nn.fit(X_train, y_train)
            preds = nn.predict(X_test)
            fold_mses.append(mean_squared_error(y_test_fold, preds))

        avg_mse = np.mean(fold_mses)
        if avg_mse < best_nn_mse:
            best_nn_mse = avg_mse
            best_nn_size = size

    nn_preds_norm = []
    for i in tqdm(range(len(X_nn_test_norm))):
        train_X = np.vstack([X_nn_cv_norm, X_nn_test_norm[:i]])
        train_y = np.hstack([y_cv_norm, y_test_norm[:i]])
        model = MLPRegressor(hidden_layer_sizes=(best_nn_size,), activation='relu', max_iter=500, random_state=42).fit(train_X, train_y)
        pred = model.predict(X_nn_test_norm[i].reshape(1, -1))
        nn_preds_norm.append(pred[0])

    nn_preds = (np.array(nn_preds_norm) * sigma) + mu
    nn_mse = mean_squared_error(y_test, nn_preds)
    nn_mae = mean_absolute_error(y_test, nn_preds)
    nn_r2 = 1 - (nn_mse / rw_mse)
    results.append({'Model': 'Neural Net', 'Hyperparams': f'hidden_layer={best_nn_size}', 'MSE': nn_mse, 'MAE': nn_mae, '$R^2$': nn_r2})

    # --- 5. Save Results ---
    result_table = pd.DataFrame(results).set_index('Model')
    output_path = PROCESSED_DATA / "problem2_solution.csv"
    result_table.to_csv(output_path)

    print("\nProblem 2 solution saved to:", output_path)
    print(result_table)
    return result_table

if __name__ == '__main__':
    solve_problem2()
