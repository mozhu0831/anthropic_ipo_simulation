import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_monte_carlo_10000():
    output_dir = r"D:\Professional\Personal Projects\Anthropic_IPO"
    os.makedirs(output_dir, exist_ok=True)
    
    n_sims = 10_000
    np.random.seed(42)
    
    # -------------------------------------------------------------------------
    # 1. Stochastic Inputs (Calibrated from ECM Syndicate Data & Microstructure)
    # -------------------------------------------------------------------------
    # A. Macro Tech Sentiment Shock Z_market ~ N(0, 1)
    z_market = np.random.normal(0, 1, n_sims)
    
    # B. Offer Price Distribution (P_offer)
    # Syndicate behavior on 7x oversubscribed deal with $70-$78 range:
    # 65% chance priced at $78.00 (cap)
    # 20% chance bumped above range ($80 - $84)
    # 15% chance priced inside range ($74 - $77)
    offer_regimes = np.random.choice(["top", "bumped", "inside"], size=n_sims, p=[0.65, 0.20, 0.15])
    p_offer = np.zeros(n_sims)
    for i, regime in enumerate(offer_regimes):
        if regime == "top":
            p_offer[i] = 78.00
        elif regime == "bumped":
            p_offer[i] = np.random.choice([80.00, 81.00, 82.00, 84.00], p=[0.4, 0.3, 0.2, 0.1])
        else: # inside
            p_offer[i] = np.random.uniform(74.00, 77.50)
            
    # C. Oversubscription realization (Base 7.0x, lognormal variation)
    oversub = np.random.lognormal(mean=np.log(7.0), sigma=0.12, size=n_sims)
    
    # D. Float & Size Drag Penalty Function
    # Float = $64.4B. Size drag dampens pop compared to a $2-3B SaaS IPO.
    # Base pop equation calibrated to historical comps (BABA, FB, ARM, RIVN, SNOW):
    retail_impact = np.random.normal(3.5, 1.2, n_sims) # 15-20% retail tranche creates early MOO surge
    idiosyncratic_shock = np.random.normal(0, 3.8, n_sims)
    
    pop_open_pct = 21.0 + 2.2 * (oversub - 7.0) + 4.5 * z_market + retail_impact + idiosyncratic_shock
    # Bound pop realistically: rarely below -5% on 7x oversubscribed book due to greenshoe defense
    pop_open_pct = np.clip(pop_open_pct, -6.0, 75.0)
    
    p_open = p_offer * (1.0 + pop_open_pct / 100.0)
    
    # -------------------------------------------------------------------------
    # 2. Intraday Dynamics: High, Low, and Close
    # -------------------------------------------------------------------------
    # A. Intraday High (Morning Momentum Overshoot):
    # First 45 mins retail/momentum rush pushes above open by gamma distribution
    high_overshoot_pct = np.random.gamma(shape=3.2, scale=2.4, size=n_sims) # Mean ~7.7% above open
    p_high = p_open * (1.0 + high_overshoot_pct / 100.0)
    
    # B. Intraday Low (Midday Retail/Fast-Money Flipping):
    pullback_pct = np.random.beta(a=3.0, b=12.0, size=n_sims) * 35.0 # Typical 6-15% pullback from high
    p_low_raw = p_high * (1.0 - pullback_pct / 100.0)
    greenshoe_support = p_offer * 0.975 # Max syndicate allowable discount before heavy stabilization
    p_low = np.maximum(np.minimum(p_low_raw, p_open * 0.99), greenshoe_support)
    
    # C. Day 1 Close (Institutional VWAP Absorption):
    reversion_weight = np.random.uniform(0.40, 0.75, size=n_sims)
    p_close_raw = p_open + reversion_weight * (p_low + p_high - 2.0 * p_open) + np.random.normal(0, 1.5, n_sims)
    p_close = np.clip(p_close_raw, p_low, p_high)
    
    day1_close_return_pct = (p_close - p_offer) / p_offer * 100.0
    
    # -------------------------------------------------------------------------
    # 3. User Position Metrics (5 Shares)
    # -------------------------------------------------------------------------
    user_shares = 5
    user_cost = user_shares * p_offer
    user_val_open = user_shares * p_open
    user_val_high = user_shares * p_high
    user_val_low = user_shares * p_low
    user_val_close = user_shares * p_close
    
    user_profit_open = user_val_open - user_cost
    user_profit_high = user_val_high - user_cost
    user_profit_close = user_val_close - user_cost

    # -------------------------------------------------------------------------
    # 4. Statistical Summary Function (Pure Numpy)
    # -------------------------------------------------------------------------
    def get_stats(arr):
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

    stats_dict = {
        "Offer_Price": get_stats(p_offer),
        "Open_Price": get_stats(p_open),
        "Open_Pop_Pct": get_stats(pop_open_pct),
        "Day1_High": get_stats(p_high),
        "Day1_Low": get_stats(p_low),
        "Day1_Close": get_stats(p_close),
        "Day1_Close_Return_Pct": get_stats(day1_close_return_pct),
        "User_Cost_Basis": get_stats(user_cost),
        "User_Open_Profit": get_stats(user_profit_open),
        "User_Peak_Profit": get_stats(user_profit_high),
        "User_Close_Profit": get_stats(user_profit_close)
    }

    # -------------------------------------------------------------------------
    # 5. Probabilistic Risk & Tail Metrics
    # -------------------------------------------------------------------------
    prob_close_above_offer = float(np.mean(p_close > p_offer) * 100.0)
    prob_broken_ipo = float(np.mean(p_close < p_offer) * 100.0)
    prob_pop_gt_20 = float(np.mean(pop_open_pct > 20.0) * 100.0)
    prob_pop_gt_30 = float(np.mean(pop_open_pct > 30.0) * 100.0)
    prob_pop_gt_50 = float(np.mean(pop_open_pct > 50.0) * 100.0)
    
    # Value at Risk (VaR 95% and 99%) on Close Profit for 5 shares
    pct_5_profit = float(np.percentile(user_profit_close, 5))
    var_95_close = -pct_5_profit if pct_5_profit < 0 else 0.0
    cvar_sub = user_profit_close[user_profit_close <= np.percentile(user_profit_close, 5)]
    cvar_95_close = -float(np.mean(cvar_sub)) if float(np.mean(cvar_sub)) < 0 else 0.0
    
    risk_metrics = {
        "Simulations_Run": n_sims,
        "P_Close_Above_Offer_WinRate": f"{prob_close_above_offer:.2f}%",
        "P_Broken_IPO_Close_Below_Offer": f"{prob_broken_ipo:.2f}%",
        "P_Open_Pop_Greater_20Pct": f"{prob_pop_gt_20:.2f}%",
        "P_Open_Pop_Greater_30Pct": f"{prob_pop_gt_30:.2f}%",
        "P_Open_Pop_Greater_50Pct": f"{prob_pop_gt_50:.2f}%",
        "VaR_95_Dollar_Loss": f"${var_95_close:.2f}",
        "CVaR_95_Expected_Shortfall": f"${cvar_95_close:.2f}"
    }

    # -------------------------------------------------------------------------
    # 6. Save Data & Tables
    # -------------------------------------------------------------------------
    df_raw = pd.DataFrame({
        "Offer_Price": p_offer,
        "Open_Price": p_open,
        "Open_Pop_Pct": pop_open_pct,
        "Day1_High": p_high,
        "Day1_Low": p_low,
        "Day1_Close": p_close,
        "Close_Return_Pct": day1_close_return_pct,
        "User_Cost": user_cost,
        "User_Profit_Open": user_profit_open,
        "User_Profit_High": user_profit_high,
        "User_Profit_Close": user_profit_close
    })
    raw_csv = os.path.join(output_dir, "monte_carlo_10k_raw_runs.csv")
    df_raw.to_csv(raw_csv, index=False)
    
    df_summary = pd.DataFrame(stats_dict).T
    summary_csv = os.path.join(output_dir, "monte_carlo_10k_statistics.csv")
    df_summary.to_csv(summary_csv)
    
    with open(os.path.join(output_dir, "monte_carlo_10k_risk_metrics.json"), "w") as f:
        json.dump(risk_metrics, f, indent=4)

    # -------------------------------------------------------------------------
    # 7. Visualization Plot (4-Panel High-Res Multi-Distribution)
    # -------------------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Opening Pop Distribution
    ax = axes[0, 0]
    ax.hist(pop_open_pct, bins=60, color="#1f77b4", edgecolor="black", alpha=0.75, density=True)
    ax.axvline(np.mean(pop_open_pct), color="red", linestyle="--", linewidth=2, label=f"Mean: +{np.mean(pop_open_pct):.1f}%")
    ax.axvline(np.median(pop_open_pct), color="green", linestyle=":", linewidth=2, label=f"Median: +{np.median(pop_open_pct):.1f}%")
    ax.axvline(np.percentile(pop_open_pct, 5), color="gray", linestyle="-.", label=f"5th Pct: +{np.percentile(pop_open_pct, 5):.1f}%")
    ax.axvline(np.percentile(pop_open_pct, 95), color="gray", linestyle="-.", label=f"95th Pct: +{np.percentile(pop_open_pct, 95):.1f}%")
    ax.set_title("1. Opening Cross Pop Distribution (% vs. Offer Price)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Opening Pop (%)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Panel 2: Day 1 Close Return Distribution
    ax = axes[0, 1]
    ax.hist(day1_close_return_pct, bins=60, color="#2ca02c", edgecolor="black", alpha=0.75, density=True)
    ax.axvline(np.mean(day1_close_return_pct), color="red", linestyle="--", linewidth=2, label=f"Mean: +{np.mean(day1_close_return_pct):.1f}%")
    ax.axvline(np.median(day1_close_return_pct), color="blue", linestyle=":", linewidth=2, label=f"Median: +{np.median(day1_close_return_pct):.1f}%")
    ax.axvline(0, color="black", linestyle="-", linewidth=1.5, label="Break-Even (0%)")
    ax.set_title(f"2. Day-1 Close Return Distribution (Win Rate: {prob_close_above_offer:.1f}%)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Close Return vs. Offer Price (%)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Panel 3: Net Profit for User (5 Shares)
    ax = axes[1, 0]
    ax.hist(user_profit_close, bins=60, color="#ff7f0e", edgecolor="black", alpha=0.75, density=True)
    ax.axvline(np.mean(user_profit_close), color="red", linestyle="--", linewidth=2, label=f"Expected Profit: +${np.mean(user_profit_close):.2f}")
    ax.axvline(np.median(user_profit_close), color="blue", linestyle=":", linewidth=2, label=f"Median Profit: +${np.median(user_profit_close):.2f}")
    ax.axvline(np.percentile(user_profit_close, 5), color="purple", linestyle="-.", label=f"5th Pct (Worst 5%): ${np.percentile(user_profit_close, 5):.2f}")
    ax.axvline(np.percentile(user_profit_close, 95), color="darkgreen", linestyle="-.", label=f"95th Pct (Top 5%): +${np.percentile(user_profit_close, 95):.2f}")
    ax.set_title("3. User Position Dollar P&L on Day 1 Close (5 Shares Allocation)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Dollar Profit / Loss (USD)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Panel 4: CDF of Day 1 Close Share Price
    ax = axes[1, 1]
    sorted_close = np.sort(p_close)
    cdf = np.arange(1, n_sims + 1) / n_sims
    ax.plot(sorted_close, cdf, color="#9467bd", linewidth=2.5, label="Empirical CDF")
    ax.axvline(78.0, color="gray", linestyle=":", label="Offer Reference ($78.00)")
    ax.axvline(np.median(p_close), color="blue", linestyle="--", label=f"Median Price: ${np.median(p_close):.2f}")
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.5)
    ax.axhline(0.9, color="green", linestyle=":", alpha=0.5, label="90% Confidence Threshold")
    ax.set_title("4. Cumulative Distribution Function (CDF) of Day-1 Close Price", fontsize=13, fontweight='bold')
    ax.set_xlabel("Day 1 Close Share Price ($)", fontsize=11)
    ax.set_ylabel("Cumulative Probability P(Price <= X)", fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "anthropic_monte_carlo_10k.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    print("Monte Carlo 10,000 Complete!")
    print(df_summary.to_string())
    print("\nRisk Metrics:")
    print(json.dumps(risk_metrics, indent=4))

if __name__ == "__main__":
    run_monte_carlo_10000()
