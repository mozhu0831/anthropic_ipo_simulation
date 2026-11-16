"""
Empirical Pricing & Underpricing Regression Engine.
Estimates the opening pop and close return using multi-factor linear/ridge models
and a calibrated non-linear Size Drag penalty.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

class EmpiricalPricingModel:
    def __init__(self, comps_df: pd.DataFrame):
        self.df = comps_df.copy()
        self.fitted = False
        self.weights_open = None
        self.weights_close = None
        self.feature_names = [
            "const",
            "log_oversub",
            "log_deal_size",
            "retail_pct",
            "float_pct",
            "vix"
        ]
        
    def _extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(df)
        X = np.zeros((n, len(self.feature_names)))
        X[:, 0] = 1.0 # intercept
        X[:, 1] = np.log(df["Oversubscription_x"].values)
        X[:, 2] = np.log(df["Deal_Size_B"].values)
        X[:, 3] = df["Retail_Pct"].values / 100.0
        X[:, 4] = df["Float_Pct"].values / 100.0
        X[:, 5] = df["VIX_Pricing"].values
        
        y_open = df["Pop_Open_Pct"].values
        y_close = df["Pop_Close_Pct"].values
        return X, y_open, y_close

    def fit(self, ridge_alpha: float = 1.0) -> Dict[str, Any]:
        X, y_open, y_close = self._extract_features(self.df)
        
        # Ridge Regression: w = (X^T X + alpha * I)^(-1) X^T y
        p = X.shape[1]
        I = np.eye(p)
        I[0, 0] = 0.0 # Do not regularize intercept
        
        xtx_reg = X.T @ X + ridge_alpha * I
        self.weights_open = np.linalg.solve(xtx_reg, X.T @ y_open)
        self.weights_close = np.linalg.solve(xtx_reg, X.T @ y_close)
        self.fitted = True
        
        # In-sample metrics
        y_pred_open = X @ self.weights_open
        y_pred_close = X @ self.weights_close
        
        r2_open = 1.0 - np.sum((y_open - y_pred_open)**2) / np.sum((y_open - np.mean(y_open))**2)
        mae_open = np.mean(np.abs(y_open - y_pred_open))
        rmse_open = np.sqrt(np.mean((y_open - y_pred_open)**2))
        
        r2_close = 1.0 - np.sum((y_close - y_pred_close)**2) / np.sum((y_close - np.mean(y_close))**2)
        mae_close = np.mean(np.abs(y_close - y_pred_close))
        rmse_close = np.sqrt(np.mean((y_close - y_pred_close)**2))
        
        return {
            "r2_open": float(r2_open),
            "mae_open": float(mae_open),
            "rmse_open": float(rmse_open),
            "r2_close": float(r2_close),
            "mae_close": float(mae_close),
            "rmse_close": float(rmse_close),
            "weights_open": dict(zip(self.feature_names, self.weights_open.tolist())),
            "weights_close": dict(zip(self.feature_names, self.weights_close.tolist()))
        }

    def predict_target(self, oversub: float, deal_size_b: float, retail_pct: float, float_pct: float, vix: float) -> Dict[str, float]:
        if not self.fitted:
            self.fit()
            
        x_vec = np.array([
            1.0,
            np.log(oversub),
            np.log(deal_size_b),
            retail_pct / 100.0,
            float_pct / 100.0,
            vix
        ])
        
        pred_open_raw = float(x_vec @ self.weights_open)
        pred_close_raw = float(x_vec @ self.weights_close)
        
        # Apply Size-Drag Non-Linear Dampening for mega-offerings > $20B
        # Market liquidity constraint: each additional $10B float exponentially increases friction
        size_threshold = 20.0
        if deal_size_b > size_threshold:
            size_excess = deal_size_b - size_threshold
            size_drag_factor = 1.0 / (1.0 + 0.012 * size_excess)
            pred_open_dampened = pred_open_raw * size_drag_factor
            pred_close_dampened = pred_close_raw * size_drag_factor
        else:
            pred_open_dampened = pred_open_raw
            pred_close_dampened = pred_close_raw
            
        return {
            "predicted_pop_open_pct": pred_open_dampened,
            "predicted_pop_close_pct": pred_close_dampened,
            "raw_pop_open_pct": pred_open_raw,
            "raw_pop_close_pct": pred_close_raw,
            "size_drag_penalty_pct": pred_open_raw - pred_open_dampened
        }

if __name__ == "__main__":
    db_path = r"D:\Professional\Personal Projects\Anthropic_IPO\data\historical_tech_ipos.csv"
    df = pd.read_csv(db_path)
    model = EmpiricalPricingModel(df)
    metrics = model.fit(ridge_alpha=1.5)
    print("Model Fit Metrics:")
    print(f"Opening Pop R^2: {metrics['r2_open']:.3f}, MAE: {metrics['mae_open']:.2f}%")
    print(f"Closing Pop R^2: {metrics['r2_close']:.3f}, MAE: {metrics['mae_close']:.2f}%")
    
    # Anthropic Deal Parameters:
    # 7x oversubscribed, $64.4B raise, 17.5% retail, 4.6% float, 18.5 VXN
    pred = model.predict_target(oversub=7.0, deal_size_b=64.35, retail_pct=17.5, float_pct=4.6, vix=18.5)
    print("\nAnthropic Empirical Prediction:")
    print(f"Predicted Open Pop: +{pred['predicted_pop_open_pct']:.2f}% (Raw: +{pred['raw_pop_open_pct']:.2f}%)")
    print(f"Predicted Close Pop: +{pred['predicted_pop_close_pct']:.2f}%")
