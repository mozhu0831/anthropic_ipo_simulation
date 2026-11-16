"""
Quantitative Retail Execution Optimizer.
Evaluates intraday execution strategies (MOO, Momentum Limit, Trailing Volatility Stop, Afternoon VWAP)
to maximize risk-adjusted P&L on allocated retail shares.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any

class ExecutionOptimizer:
    def __init__(self, raw_sim_runs: pd.DataFrame, shares: int = 5):
        self.runs = raw_sim_runs.copy()
        self.shares = shares
        
    def evaluate_strategies(self) -> pd.DataFrame:
        p_offer = self.runs["Offer_Price"].values
        p_open = self.runs["Open_Price"].values
        p_high = self.runs["Day1_High"].values
        p_low = self.runs["Day1_Low"].values
        p_close = self.runs["Day1_Close"].values
        n = len(self.runs)
        
        # Strategy 1: Immediate Market-on-Open (MOO)
        strat1_fill = p_open
        strat1_pnl = (strat1_fill - p_offer) * self.shares
        
        # Strategy 2: Golden Window Limit Order (Target = Open * 1.06)
        target_limit = p_open * 1.06
        filled_limit = p_high >= target_limit
        # If not filled at limit, sold at close
        strat2_fill = np.where(filled_limit, target_limit, p_close)
        strat2_pnl = (strat2_fill - p_offer) * self.shares
        strat2_fill_rate = float(np.mean(filled_limit) * 100.0)
        
        # Strategy 3: Trailing Volatility Stop (4% trailing below peak)
        trail_fill = p_high * 0.96
        # Fill is max of trail stop and low
        strat3_fill = np.maximum(trail_fill, p_low)
        strat3_pnl = (strat3_fill - p_offer) * self.shares
        
        # Strategy 4: Afternoon Institutional VWAP Exit
        strat4_fill = p_close
        strat4_pnl = (strat4_fill - p_offer) * self.shares
        
        results = [
            {
                "Strategy": "1. Immediate Open Cross (MOO)",
                "Execution_Window": "10:15 - 10:30 AM ET",
                "Mean_Fill_Price": f"${np.mean(strat1_fill):.2f}",
                "Expected_Profit": f"+${np.mean(strat1_pnl):.2f}",
                "Fill_Rate": "100.0%",
                "Return_Pct": f"+{np.mean((strat1_fill - p_offer) / p_offer)*100:.1f}%",
                "Std_Dev_Profit": f"${np.std(strat1_pnl):.2f}",
                "Min_Profit": f"${np.min(strat1_pnl):.2f}",
                "Max_Profit": f"${np.max(strat1_pnl):.2f}",
                "Sharpe_Ratio": f"{np.mean(strat1_pnl) / (np.std(strat1_pnl) + 1e-6):.2f}",
                "Description": "Sell on the first trade. Zero execution risk, but foregoes morning momentum spike."
            },
            {
                "Strategy": "2. Golden Window Limit Order",
                "Execution_Window": "11:00 - 11:45 AM ET",
                "Mean_Fill_Price": f"${np.mean(strat2_fill):.2f}",
                "Expected_Profit": f"+${np.mean(strat2_pnl):.2f}",
                "Fill_Rate": f"{strat2_fill_rate:.1f}%",
                "Return_Pct": f"+{np.mean((strat2_fill - p_offer) / p_offer)*100:.1f}%",
                "Std_Dev_Profit": f"${np.std(strat2_pnl):.2f}",
                "Min_Profit": f"${np.min(strat2_pnl):.2f}",
                "Max_Profit": f"${np.max(strat2_pnl):.2f}",
                "Sharpe_Ratio": f"{np.mean(strat2_pnl) / (np.std(strat2_pnl) + 1e-6):.2f}",
                "Description": "Place limit order at +6% above open. Captures peak retail FOMO before midday flipping."
            },
            {
                "Strategy": "3. Trailing Volatility Stop (4%)",
                "Execution_Window": "Intraday Dynamic",
                "Mean_Fill_Price": f"${np.mean(strat3_fill):.2f}",
                "Expected_Profit": f"+${np.mean(strat3_pnl):.2f}",
                "Fill_Rate": "100.0%",
                "Return_Pct": f"+{np.mean((strat3_fill - p_offer) / p_offer)*100:.1f}%",
                "Std_Dev_Profit": f"${np.std(strat3_pnl):.2f}",
                "Min_Profit": f"${np.min(strat3_pnl):.2f}",
                "Max_Profit": f"${np.max(strat3_pnl):.2f}",
                "Sharpe_Ratio": f"{np.mean(strat3_pnl) / (np.std(strat3_pnl) + 1e-6):.2f}",
                "Description": "Dynamic trailing stop follows high watermark, locking in upside while hedging afternoon drop."
            },
            {
                "Strategy": "4. Afternoon Institutional VWAP",
                "Execution_Window": "2:30 - 3:45 PM ET",
                "Mean_Fill_Price": f"${np.mean(strat4_fill):.2f}",
                "Expected_Profit": f"+${np.mean(strat4_pnl):.2f}",
                "Fill_Rate": "100.0%",
                "Return_Pct": f"+{np.mean((strat4_fill - p_offer) / p_offer)*100:.1f}%",
                "Std_Dev_Profit": f"${np.std(strat4_pnl):.2f}",
                "Min_Profit": f"${np.min(strat4_pnl):.2f}",
                "Max_Profit": f"${np.max(strat4_pnl):.2f}",
                "Sharpe_Ratio": f"{np.mean(strat4_pnl) / (np.std(strat4_pnl) + 1e-6):.2f}",
                "Description": "Hold past the midday flipping dip; sell into institutional VWAP rebalancing before close."
            }
        ]
        return pd.DataFrame(results)

if __name__ == "__main__":
    raw_path = r"D:\Professional\Personal Projects\Anthropic_IPO\monte_carlo_10k_raw_runs.csv"
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        opt = ExecutionOptimizer(df_raw, shares=5)
        df_strat = opt.evaluate_strategies()
        print("Execution Strategy Optimization Results:")
        print(df_strat[["Strategy", "Expected_Profit", "Return_Pct", "Sharpe_Ratio", "Fill_Rate"]].to_string())
