"""
Vectorized Correlated Monte Carlo Simulation Engine.
Executes 10,000+ stochastic iterations modeling correlated macro sentiment,
institutional book oversubscription, retail flow skew, and intraday path dynamics.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.config import AnthropicDealConfig, SimulationConfig

class MonteCarloEngine:
    def __init__(self, deal_config: AnthropicDealConfig, sim_config: SimulationConfig):
        self.deal = deal_config
        self.sim = sim_config
        
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        n = self.sim.n_simulations
        np.random.seed(self.sim.random_seed)
        
        # 1. Correlated Multivariate Shocks using Cholesky Decomposition
        # Variables: [Macro Sentiment Shock, Oversubscription Shock, Retail Surge Shock]
        # Correlation matrix:
        # Macro & Oversub: +0.45
        # Macro & Retail: +0.60
        # Oversub & Retail: +0.35
        corr_matrix = np.array([
            [1.00, 0.45, 0.60],
            [0.45, 1.00, 0.35],
            [0.60, 0.35, 1.00]
        ])
        L = np.linalg.cholesky(corr_matrix)
        uncorrelated = np.random.normal(0, 1, (3, n))
        correlated_shocks = L @ uncorrelated
        
        z_macro = correlated_shocks[0]
        z_oversub = correlated_shocks[1]
        z_retail = correlated_shocks[2]
        
        # 2. Offer Price Simulation (P_offer)
        # Syndicate pricing discretion for 7x book:
        # 65% chance priced at cap ($78.00)
        # 20% chance bumped above ($80 - $84)
        # 15% chance priced inside ($74 - $77.50)
        offer_regimes = np.random.choice(["top", "bumped", "inside"], size=n, p=[0.65, 0.20, 0.15])
        p_offer = np.zeros(n)
        for i, regime in enumerate(offer_regimes):
            if regime == "top":
                p_offer[i] = self.deal.filing_price_high
            elif regime == "bumped":
                p_offer[i] = np.random.choice([80.00, 81.00, 82.00, 84.00], p=[0.4, 0.3, 0.2, 0.1])
            else:
                p_offer[i] = np.random.uniform(self.deal.filing_price_low + 4.0, self.deal.filing_price_high - 0.5)
                
        # 3. Oversubscription realization
        oversub_realized = self.deal.reported_oversubscription * np.exp(0.10 * z_oversub)
        
        # 4. Opening Pop (%):
        # Size Drag model: Anthropic $64.4B deal has natural friction
        base_pop = 21.5 # %
        pop_open_pct = (
            base_pop 
            + 2.4 * (oversub_realized - self.deal.reported_oversubscription)
            + 4.2 * z_macro
            + 1.8 * z_retail
            + np.random.normal(0, 2.5, n)
        )
        pop_open_pct = np.clip(pop_open_pct, -5.0, 75.0)
        p_open = p_offer * (1.0 + pop_open_pct / 100.0)
        
        # 5. Intraday High (Morning Momentum Overshoot)
        # First 45 mins retail/momentum rush pushes above open
        high_overshoot_pct = np.random.gamma(shape=3.2, scale=2.4, size=n)
        p_high = p_open * (1.0 + high_overshoot_pct / 100.0)
        
        # 6. Intraday Low (Midday Retail/Fast-Money Flipping)
        # Profit taking causes retracement, backed by greenshoe defense at 0.975 of offer
        pullback_pct = np.random.beta(a=3.0, b=12.0, size=n) * 35.0
        p_low_raw = p_high * (1.0 - pullback_pct / 100.0)
        greenshoe_support = p_offer * 0.975
        p_low = np.maximum(np.minimum(p_low_raw, p_open * 0.99), greenshoe_support)
        
        # 7. Day 1 Close (Institutional VWAP Absorption)
        reversion_weight = np.random.uniform(0.42, 0.72, size=n)
        p_close_raw = p_open + reversion_weight * (p_low + p_high - 2.0 * p_open) + np.random.normal(0, 1.2, n)
        p_close = np.clip(p_close_raw, p_low, p_high)
        close_return_pct = (p_close - p_offer) / p_offer * 100.0
        
        # 8. User Position Metrics (5 Shares)
        user_shares = self.deal.user_shares_entered
        user_cost = user_shares * p_offer
        user_val_open = user_shares * p_open
        user_val_high = user_shares * p_high
        user_val_close = user_shares * p_close
        
        profit_open = user_val_open - user_cost
        profit_high = user_val_high - user_cost
        profit_close = user_val_close - user_cost
        
        # Build Raw Runs DataFrame
        df_runs = pd.DataFrame({
            "Offer_Price": p_offer,
            "Open_Price": p_open,
            "Open_Pop_Pct": pop_open_pct,
            "Day1_High": p_high,
            "Day1_Low": p_low,
            "Day1_Close": p_close,
            "Close_Return_Pct": close_return_pct,
            "User_Cost": user_cost,
            "User_Profit_Open": profit_open,
            "User_Profit_High": profit_high,
            "User_Profit_Close": profit_close
        })
        
        # Compute Statistics
        def calc_stats(arr):
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            skew = float(np.mean(((arr - mean) / (std + 1e-9)) ** 3))
            kurt = float(np.mean(((arr - mean) / (std + 1e-9)) ** 4) - 3.0)
            return {
                "Mean": mean,
                "Std": std,
                "Median": float(np.median(arr)),
                "IQR_25": float(np.percentile(arr, 25)),
                "IQR_75": float(np.percentile(arr, 75)),
                "Pct_05": float(np.percentile(arr, 5)),
                "Pct_95": float(np.percentile(arr, 95)),
                "Min": float(np.min(arr)),
                "Max": float(np.max(arr)),
                "Skewness": skew,
                "Kurtosis": kurt
            }
            
        stats_data = {
            "Offer_Price": calc_stats(p_offer),
            "Open_Price": calc_stats(p_open),
            "Open_Pop_Pct": calc_stats(pop_open_pct),
            "Day1_High": calc_stats(p_high),
            "Day1_Low": calc_stats(p_low),
            "Day1_Close": calc_stats(p_close),
            "Close_Return_Pct": calc_stats(close_return_pct),
            "User_Cost_Basis": calc_stats(user_cost),
            "User_Open_Profit": calc_stats(profit_open),
            "User_Peak_Profit": calc_stats(profit_high),
            "User_Close_Profit": calc_stats(profit_close)
        }
        df_stats = pd.DataFrame(stats_data).T
        
        # Risk Metrics
        pct_5_loss = float(np.percentile(profit_close, 5))
        var_95 = -pct_5_loss if pct_5_loss < 0 else 0.0
        
        risk_metrics = {
            "Simulations_Run": n,
            "P_Close_Above_Offer_WinRate": f"{float(np.mean(p_close > p_offer) * 100.0):.2f}%",
            "P_Broken_IPO_Close_Below_Offer": f"{float(np.mean(p_close < p_offer) * 100.0):.2f}%",
            "P_Open_Pop_Greater_20Pct": f"{float(np.mean(pop_open_pct > 20.0) * 100.0):.2f}%",
            "P_Open_Pop_Greater_30Pct": f"{float(np.mean(pop_open_pct > 30.0) * 100.0):.2f}%",
            "P_Open_Pop_Greater_50Pct": f"{float(np.mean(pop_open_pct > 50.0) * 100.0):.2f}%",
            "VaR_95_Dollar_Loss": f"${var_95:.2f}",
            "Expected_Position_Return_Close": f"{float(np.mean(close_return_pct)):.2f}%",
            "Sharpe_Ratio_Day1": float(np.mean(close_return_pct) / np.std(close_return_pct))
        }
        
        return df_runs, df_stats, risk_metrics

if __name__ == "__main__":
    deal = AnthropicDealConfig()
    sim = SimulationConfig(n_simulations=1000)
    engine = MonteCarloEngine(deal, sim)
    runs, stats, risk = engine.run()
    print("Monte Carlo Vectorized Run OK:")
    print(stats)
    print(risk)
