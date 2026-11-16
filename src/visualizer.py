"""
High-Resolution Publication Visualizer.
Generates institutional-grade figures for the GitHub research showcase:
1. Historical Comps & Size Drag Curve (figures/comps_size_drag_regression.png)
2. Nasdaq NOII Clearing Price Convergence (figures/nasdaq_noii_convergence.png)
3. Intraday Microstructure & VWAP (figures/intraday_microstructure_vwap.png)
4. Monte Carlo Distribution Dashboard (figures/monte_carlo_distribution_dashboard.png)
5. Optimal Execution Frontier (figures/optimal_execution_frontier.png)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ResearchVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.style_setup()
        
    def style_setup(self):
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })

    def plot_comps_and_size_drag(self, df_comps: pd.DataFrame, anthropic_deal_size: float = 64.35, anthropic_pred_pop: float = 24.6):
        fig, ax = plt.subplots(figsize=(12, 7))
        
        scatter = ax.scatter(
            df_comps["Deal_Size_B"], 
            df_comps["Pop_Open_Pct"], 
            c=df_comps["Oversubscription_x"], 
            cmap="viridis", 
            s=df_comps["Valuation_B"] * 1.5 + 40,
            alpha=0.85, 
            edgecolor="black",
            linewidth=1.2,
            zorder=3
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Oversubscription Multiple (x)", fontsize=11, fontweight='bold')
        
        labels_to_show = ["SNOW", "ABNB", "BABA", "FB", "ARM", "RIVN", "UBER", "ALAB", "RDDT"]
        for _, row in df_comps.iterrows():
            if row["Ticker"] in labels_to_show:
                ax.annotate(
                    f"{row['Ticker']} (+{row['Pop_Open_Pct']:.0f}%)",
                    (row["Deal_Size_B"], row["Pop_Open_Pct"]),
                    textcoords="offset points",
                    xytext=(8, 5),
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="gray")
                )
                
        x_curve = np.linspace(0.1, 75, 200)
        y_curve = 45.0 / (1.0 + 0.018 * x_curve)
        ax.plot(x_curve, y_curve, color="#d62728", linestyle="--", linewidth=2.2, label="Empirical Size Drag Curve", zorder=2)
        
        ax.scatter([anthropic_deal_size], [anthropic_pred_pop], color="#ff7f0e", marker="*", s=450, edgecolor="black", linewidth=1.5, label=f"Anthropic ANT (Est: +{anthropic_pred_pop:.1f}%)", zorder=5)
        ax.annotate(
            f"Anthropic ANT ($64.4B)\nEst. Pop: +{anthropic_pred_pop:.1f}%",
            (anthropic_deal_size, anthropic_pred_pop),
            textcoords="offset points",
            xytext=(-110, 15),
            fontsize=10,
            fontweight='bold',
            color="#d95f02",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff2e6", ec="#d95f02", lw=1.5),
            arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.5)
        )
        
        ax.set_title("Empirical Tech IPO Comps & The 'Size Drag' Law (Deal Size vs. Opening Pop %)", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("IPO Deal Size ($ Billions Raised)", fontsize=11, fontweight='bold')
        ax.set_ylabel("First Trade Opening Pop (%)", fontsize=11, fontweight='bold')
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)
        
        out_path = os.path.join(self.output_dir, "comps_size_drag_regression.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_noii_convergence(self, df_noii: pd.DataFrame):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        
        x = np.arange(len(df_noii))
        time_labels = df_noii["Time"]
        
        ax1.plot(x, df_noii["Far_Clearing_Price"], label="Far Clearing Price (Continuous Market Orders)", color="#d62728", linewidth=2.0, linestyle="--")
        ax1.plot(x, df_noii["Near_Clearing_Price"], label="Near Clearing Price (All Eligible Orders)", color="#1f77b4", linewidth=2.2)
        ax1.plot(x, df_noii["Reference_Price"], label="Current Reference Price (NBBO Mid)", color="#2ca02c", linewidth=1.8, linestyle=":")
        ax1.axhline(78.0, color="gray", linestyle="-.", label="Offer Price ($78.00)")
        
        final_price = df_noii["Near_Clearing_Price"].iloc[-1]
        ax1.scatter([x[-1]], [final_price], color="#1f77b4", s=120, zorder=5)
        ax1.annotate(f"Uncrossing Print:\n${final_price:.2f} (+24.4%)", (x[-1], final_price), textcoords="offset points", xytext=(-85, 10),
                     fontweight='bold', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#e6f2ff", ec="#1f77b4"))
        
        ax1.set_title("Nasdaq Opening Cross Simulation (NOII Price Discovery & Imbalance Convergence)", fontsize=13, fontweight='bold', pad=10)
        ax1.set_ylabel("Price (USD)", fontsize=11, fontweight='bold')
        ax1.legend(loc="upper left", frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.6)
        
        ax2.bar(x, df_noii["Paired_Shares"] / 1e6, width=0.6, color="#1f77b4", alpha=0.7, label="Paired Shares (Millions)")
        ax2.plot(x, df_noii["Imbalance_Shares"] / 1e6, color="#ff7f0e", linewidth=2.0, marker="o", markersize=4, label="Net Buy Imbalance (Millions)")
        ax2.set_ylabel("Shares (Millions)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Time (ET) - Pre-Opening Price Discovery Window", fontsize=11, fontweight='bold')
        ax2.set_xticks(x[::5])
        ax2.set_xticklabels(time_labels.iloc[::5])
        ax2.legend(loc="upper right", frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.6)
        
        out_path = os.path.join(self.output_dir, "nasdaq_noii_convergence.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_intraday_microstructure(self, df_intra: pd.DataFrame):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2.2, 1]}, sharex=True)
        
        x = np.arange(len(df_intra))
        time_labels = df_intra["Timestamp"]
        
        ax1.plot(x, df_intra["Price"], label="Intraday Spot Price", color="#1f77b4", linewidth=2.0)
        ax1.plot(x, df_intra["VWAP"], label="Intraday VWAP (Volume Weighted Average Price)", color="#d62728", linewidth=2.0, linestyle="--")
        ax1.axhline(78.0, color="gray", linestyle=":", label="Offer Price ($78.00)")
        
        ax1.axvspan(0, 45, color="#2ca02c", alpha=0.10, label="Phase 1: Retail & Momentum Rush (10:30-11:15)")
        ax1.axvspan(45, 180, color="#d62728", alpha=0.10, label="Phase 2: Flipping Wave & Lunch Lull (11:15-13:30)")
        ax1.axvspan(180, 300, color="#1f77b4", alpha=0.10, label="Phase 3: Institutional VWAP Accumulation (13:30-15:30)")
        ax1.axvspan(300, 330, color="#9467bd", alpha=0.10, label="Phase 4: MOC Rebalancing (15:30-16:00)")
        
        p_high_idx = df_intra["Price"].idxmax()
        p_high = df_intra["Price"].iloc[p_high_idx]
        ax1.annotate(f"Peak: ${p_high:.2f}", (p_high_idx, p_high), textcoords="offset points", xytext=(-25, 12),
                     fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="#ffffcc", ec="orange"),
                     arrowprops=dict(arrowstyle="->", color="orange", lw=1.5))
                     
        ax1.set_title("Anthropic (ANT) Intraday Microstructure & VWAP Trajectory (Day-1)", fontsize=13, fontweight='bold', pad=10)
        ax1.set_ylabel("Price (USD)", fontsize=11, fontweight='bold')
        ax1.legend(loc="upper left", frameon=True, fontsize=8.5)
        ax1.grid(True, linestyle="--", alpha=0.6)
        
        colors = ["#2ca02c" if i <= 45 else "#d62728" if i <= 180 else "#1f77b4" if i <= 300 else "#9467bd" for i in x]
        ax2.bar(x, df_intra["Volume"] / 1e6, color=colors, alpha=0.7, width=0.8)
        ax2.set_ylabel("Volume (M Shares)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Trading Time (ET)", fontsize=11, fontweight='bold')
        ax2.set_xticks(x[::30])
        ax2.set_xticklabels(time_labels.iloc[::30])
        ax2.grid(True, linestyle="--", alpha=0.6)
        
        out_path = os.path.join(self.output_dir, "intraday_microstructure_vwap.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_monte_carlo_dashboard(self, df_runs: pd.DataFrame, risk_metrics: dict):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        pop_open = df_runs["Open_Pop_Pct"].values
        close_return = df_runs["Close_Return_Pct"].values
        profit_close = df_runs["User_Profit_Close"].values
        p_close = df_runs["Day1_Close"].values
        n = len(df_runs)
        
        # Panel 1: Opening Pop Distribution
        ax = axes[0, 0]
        ax.hist(pop_open, bins=60, color="#1f77b4", edgecolor="black", alpha=0.75, density=True)
        ax.axvline(np.mean(pop_open), color="red", linestyle="--", linewidth=2, label=f"Mean: +{np.mean(pop_open):.1f}%")
        ax.axvline(np.median(pop_open), color="green", linestyle=":", linewidth=2, label=f"Median: +{np.median(pop_open):.1f}%")
        ax.axvline(np.percentile(pop_open, 5), color="gray", linestyle="-.", label=f"5th Pct: +{np.percentile(pop_open, 5):.1f}%")
        ax.axvline(np.percentile(pop_open, 95), color="gray", linestyle="-.", label=f"95th Pct: +{np.percentile(pop_open, 95):.1f}%")
        ax.set_title("1. Opening Cross Pop Distribution (% vs. Offer Price)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Opening Pop (%)", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # Panel 2: Day 1 Close Return Distribution
        ax = axes[0, 1]
        ax.hist(close_return, bins=60, color="#2ca02c", edgecolor="black", alpha=0.75, density=True)
        ax.axvline(np.mean(close_return), color="red", linestyle="--", linewidth=2, label=f"Mean: +{np.mean(close_return):.1f}%")
        ax.axvline(np.median(close_return), color="blue", linestyle=":", linewidth=2, label=f"Median: +{np.median(close_return):.1f}%")
        ax.axvline(0, color="black", linestyle="-", linewidth=1.5, label="Break-Even (0%)")
        ax.set_title(f"2. Day-1 Close Return Distribution (Win Rate: {risk_metrics['P_Close_Above_Offer_WinRate']})", fontsize=13, fontweight='bold')
        ax.set_xlabel("Close Return vs. Offer Price (%)", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # Panel 3: Net Profit for User (5 Shares)
        ax = axes[1, 0]
        ax.hist(profit_close, bins=60, color="#ff7f0e", edgecolor="black", alpha=0.75, density=True)
        ax.axvline(np.mean(profit_close), color="red", linestyle="--", linewidth=2, label=f"Expected Profit: +${np.mean(profit_close):.2f}")
        ax.axvline(np.median(profit_close), color="blue", linestyle=":", linewidth=2, label=f"Median Profit: +${np.median(profit_close):.2f}")
        ax.axvline(np.percentile(profit_close, 5), color="purple", linestyle="-.", label=f"5th Pct: ${np.percentile(profit_close, 5):.2f}")
        ax.axvline(np.percentile(profit_close, 95), color="darkgreen", linestyle="-.", label=f"95th Pct: +${np.percentile(profit_close, 95):.2f}")
        ax.set_title("3. User Position Dollar P&L on Day-1 Close (5 Shares Allocation)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Dollar Profit / Loss (USD)", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # Panel 4: CDF of Day 1 Close Share Price
        ax = axes[1, 1]
        sorted_close = np.sort(p_close)
        cdf = np.arange(1, n + 1) / n
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
        
        out_path = os.path.join(self.output_dir, "monte_carlo_distribution_dashboard.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_execution_frontier(self, df_strat: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(11, 6))
        
        strats = df_strat["Strategy"]
        profits = [float(p.replace("+$", "").replace("$", "")) for p in df_strat["Expected_Profit"]]
        sharpes = [float(s) for s in df_strat["Sharpe_Ratio"]]
        
        y_pos = np.arange(len(strats))
        bars = ax.barh(y_pos, profits, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"], alpha=0.85, edgecolor="black", height=0.55)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2.0, 
                    f"+${width:.2f}  (Sharpe: {sharpes[i]:.2f})", 
                    va='center', fontsize=10, fontweight='bold')
                    
        ax.set_yticks(y_pos)
        ax.set_yticklabels(strats, fontsize=10, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel("Expected Dollar Profit on 5 Shares Allocation ($)", fontsize=11, fontweight='bold')
        ax.set_title("Retail Execution Strategy Optimization Frontier (Risk-Adjusted P&L Comparison)", fontsize=13, fontweight='bold', pad=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(0, max(profits) * 1.25)
        
        out_path = os.path.join(self.output_dir, "optimal_execution_frontier.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path
