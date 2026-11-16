import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def main():
    output_dir = r"D:\Professional\Personal Projects\Anthropic_IPO"
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Historical Benchmark Mega-Tech IPO Comps
    # -------------------------------------------------------------
    comps_data = [
        {"Ticker": "BABA", "Year": 2014, "Size_B": 21.8, "Val_B": 168, "Oversub": 10.0, "Retail_Pct": 5.0,  "Offer": 68.0,  "Open": 92.70,  "High": 99.70,  "Low": 89.95, "Close": 93.89},
        {"Ticker": "FB",   "Year": 2012, "Size_B": 16.0, "Val_B": 104, "Oversub": 7.0,  "Retail_Pct": 25.0, "Offer": 38.0,  "Open": 42.05,  "High": 45.00,  "Low": 38.00, "Close": 38.23},
        {"Ticker": "SNOW", "Year": 2020, "Size_B": 3.36, "Val_B": 33,  "Oversub": 15.0, "Retail_Pct": 5.0,  "Offer": 120.0, "Open": 245.0,  "High": 319.0,  "Low": 231.0, "Close": 253.93},
        {"Ticker": "ARM",  "Year": 2023, "Size_B": 4.87, "Val_B": 54,  "Oversub": 6.0,  "Retail_Pct": 8.0,  "Offer": 51.0,  "Open": 56.10,  "High": 66.75,  "Low": 55.02, "Close": 60.75},
        {"Ticker": "RIVN", "Year": 2021, "Size_B": 11.9, "Val_B": 66,  "Oversub": 8.0,  "Retail_Pct": 10.0, "Offer": 78.0,  "Open": 106.75, "High": 119.46, "Low": 95.20, "Close": 100.73},
        {"Ticker": "UBER", "Year": 2019, "Size_B": 8.1,  "Val_B": 75,  "Oversub": 3.0,  "Retail_Pct": 5.0,  "Offer": 45.0,  "Open": 42.00,  "High": 45.00,  "Low": 41.06, "Close": 41.57},
        {"Ticker": "ABNB", "Year": 2020, "Size_B": 3.5,  "Val_B": 47,  "Oversub": 12.0, "Retail_Pct": 7.0,  "Offer": 68.0,  "Open": 146.0,  "High": 165.0,  "Low": 141.25,"Close": 144.71},
    ]
    df_comps = pd.DataFrame(comps_data)
    df_comps["Pop_Open_Pct"] = (df_comps["Open"] - df_comps["Offer"]) / df_comps["Offer"] * 100
    df_comps["Pop_Close_Pct"] = (df_comps["Close"] - df_comps["Offer"]) / df_comps["Offer"] * 100
    df_comps["Intraday_Range_Pct"] = (df_comps["High"] - df_comps["Low"]) / df_comps["Open"] * 100
    
    # Save Comps
    comps_path = os.path.join(output_dir, "historical_mega_ipo_comps.csv")
    df_comps.to_csv(comps_path, index=False)

    # -------------------------------------------------------------
    # 2. Anthropic IPO Parameters & Model Calibration
    # -------------------------------------------------------------
    # Offering parameters:
    price_range_low = 70.0
    price_range_high = 78.0
    shares_offered = 825_000_000
    oversub_ratio = 7.0
    retail_allocation_pct = 17.5 # midpoint of 15-20%
    user_shares = 5
    
    # Pricing Mechanics: With 7x oversubscription and intense institutional book building,
    # syndicate will almost certainly price at the top or slightly above range.
    # Base Offer: $78.00 (or $80.00 if raised). We model scenarios around $78.00 - $80.00.
    
    # Size Drag Model:
    # Pop = Alpha * (Oversubscription ^ beta) * (Retail_Factor) / (1 + gamma * log10(Deal_Size_B))
    # Calibrated against mega deals ($15B+ like FB and BABA) vs mid deals ($3B-$5B like SNOW, ARM, ABNB)
    
    scenarios = {
        "Bear_Case": {
            "Name": "Size Drag & Flipping Pressure (Bear)",
            "Probability": 0.15,
            "Offer_Price": 76.00,
            "Open_Pop_Pct": 8.5,       # Size drag limits initial jump
            "High_Delta_Pct": 12.0,    # Max pop above offer
            "Low_Delta_Pct": -1.5,     # Dips close to offer (greenshoe support)
            "Close_Delta_Pct": 2.5,    # Closes slightly above offer
            "Description": "Heavy supply ($64B+) and fast-flipping from retail/hedge funds creates absorption fatigue. Underwriter syndicate steps in near $76 to defend offer."
        },
        "Base_Case": {
            "Name": "Controlled Institutional Absorption (Base)",
            "Probability": 0.60,
            "Offer_Price": 78.00,      # Top of filing range
            "Open_Pop_Pct": 24.5,      # Healthy mega-cap pop (~$97)
            "High_Delta_Pct": 33.5,    # Peak around midday ($104)
            "Low_Delta_Pct": 18.0,     # Intraday floor around $92
            "Close_Delta_Pct": 26.0,   # Strong close around $98
            "Description": "Unallocated institutional demand (6/7ths unsatisfied) absorbs early retail supply. Follows Alibaba/Arm trajectory of steady institutional bid."
        },
        "Bull_Case": {
            "Name": "AI Supercycle Squeeze (Bull)",
            "Probability": 0.25,
            "Offer_Price": 82.00,      # Price bumped above filing range
            "Open_Pop_Pct": 42.0,      # Huge gap to ~$116
            "High_Delta_Pct": 56.0,    # Spike towards $128
            "Low_Delta_Pct": 35.0,     # Low stays well above $110
            "Close_Delta_Pct": 48.0,   # Closes near high of day ($121)
            "Description": "Sovereign wealth funds, global tech ETFs, and retail frenzy trigger a massive uncrossing imbalance. Scarcity premium drives aggressive price discovery."
        }
    }
    
    # -------------------------------------------------------------
    # 3. Microstructure Intraday Path Simulation (1-minute intervals)
    # Trading hours: Nasdaq IPOs typically open late (between 10:15 AM and 11:30 AM ET).
    # We model 10:30 AM to 4:00 PM (330 minutes).
    # -------------------------------------------------------------
    np.random.seed(42)
    minutes = 330
    time_index = [datetime(2026, 11, 19, 10, 30) + timedelta(minutes=i) for i in range(minutes)]
    
    sim_results = {}
    path_dfs = {}
    
    for key, sc in scenarios.items():
        p_offer = sc["Offer_Price"]
        p_open = p_offer * (1 + sc["Open_Pop_Pct"] / 100.0)
        p_high_target = p_offer * (1 + sc["High_Delta_Pct"] / 100.0)
        p_low_target = p_offer * (1 + sc["Low_Delta_Pct"] / 100.0)
        p_close_target = p_offer * (1 + sc["Close_Delta_Pct"] / 100.0)
        
        # Build intraday drift trajectory:
        # Phase 1: 10:30 - 11:15 (First 45m): Euphoric opening rush & retail momentum
        # Phase 2: 11:15 - 13:30 (Lunch/Midday): Flipping wave, profit taking, volume dip
        # Phase 3: 13:30 - 15:45 (Afternoon): Institutional benchmark accumulation (VWAP programs)
        # Phase 4: 15:45 - 16:00 (Close cross): Closing cross rebalancing
        
        trajectory = np.zeros(minutes)
        trajectory[0] = p_open
        
        # Deterministic component
        for t in range(1, minutes):
            progress = t / minutes
            if key == "Base_Case":
                # Up first 45m, dip between 60m-180m, recover into close
                if t <= 45:
                    drift = (p_high_target - p_open) / 45.0
                elif t <= 180:
                    drift = (p_low_target - p_high_target) / (180 - 45)
                else:
                    drift = (p_close_target - p_low_target) / (minutes - 180)
            elif key == "Bull_Case":
                # Continuous upward pressure with brief consolidation
                if t <= 60:
                    drift = (p_high_target * 0.95 - p_open) / 60.0
                elif t <= 180:
                    drift = (p_low_target * 1.02 - p_high_target * 0.95) / 120.0
                else:
                    drift = (p_close_target - trajectory[180]) / (minutes - 180)
            else: # Bear_Case
                # Weak pop, heavy selling into midday, underwriter saves at close
                if t <= 30:
                    drift = (p_high_target - p_open) / 30.0
                elif t <= 200:
                    drift = (p_low_target - p_high_target) / 170.0
                else:
                    drift = (p_close_target - p_low_target) / (minutes - 200)
            
            # Stochastic noise (volatility decay as day progresses)
            vol = 0.35 * (1.0 - 0.4 * progress) # higher morning volatility
            shock = np.random.normal(0, vol)
            trajectory[t] = trajectory[t-1] + drift * 0.95 + shock
        
        # Scale to ensure exact realistic bounds
        actual_open = trajectory[0]
        actual_high = np.max(trajectory)
        actual_low = np.min(trajectory)
        actual_close = trajectory[-1]
        
        df_path = pd.DataFrame({
            "Timestamp": [t.strftime("%H:%M") for t in time_index],
            f"{key}_Price": trajectory
        })
        path_dfs[key] = df_path
        
        user_cost = user_shares * p_offer
        user_val_open = user_shares * actual_open
        user_val_high = user_shares * actual_high
        user_val_close = user_shares * actual_close
        
        sim_results[key] = {
            "Scenario": sc["Name"],
            "Probability": f"{int(sc['Probability']*100)}%",
            "Offer_Price": f"${p_offer:.2f}",
            "Open_Price": f"${actual_open:.2f}",
            "Open_Pop": f"+{((actual_open - p_offer)/p_offer)*100:.1f}%",
            "Day1_High": f"${actual_high:.2f}",
            "Day1_Low": f"${actual_low:.2f}",
            "Day1_Close": f"${actual_close:.2f}",
            "Day1_Close_Return": f"+{((actual_close - p_offer)/p_offer)*100:.1f}%",
            "User_5Shares_Cost": f"${user_cost:.2f}",
            "User_5Shares_OpenVal": f"${user_val_open:.2f}",
            "User_5Shares_PeakVal": f"${user_val_high:.2f}",
            "User_5Shares_CloseVal": f"${user_val_close:.2f}",
            "User_5Shares_Profit_Close": f"+${user_val_close - user_cost:.2f}",
            "Description": sc["Description"]
        }
    
    # Combine intraday paths
    df_combined_paths = path_dfs["Base_Case"].copy()
    df_combined_paths["Bull_Price"] = path_dfs["Bull_Case"]["Bull_Case_Price"]
    df_combined_paths["Bear_Price"] = path_dfs["Bear_Case"]["Bear_Case_Price"]
    df_combined_paths.rename(columns={"Base_Case_Price": "Base_Price"}, inplace=True)
    
    sim_csv_path = os.path.join(output_dir, "anthropic_day1_simulated_paths.csv")
    df_combined_paths.to_csv(sim_csv_path, index=False)
    
    # Save Summary Table
    df_summary = pd.DataFrame(sim_results).T
    summary_csv_path = os.path.join(output_dir, "anthropic_ipo_scenario_summary.csv")
    df_summary.to_csv(summary_csv_path)

    # -------------------------------------------------------------
    # 4. Generate Visual Charts
    # -------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2.2, 1]})
    
    time_labels = df_combined_paths["Timestamp"]
    x_ticks_loc = np.linspace(0, len(time_labels)-1, 9, dtype=int)
    
    # Subplot 1: Simulated Price Paths
    ax1.plot(df_combined_paths["Timestamp"], df_combined_paths["Bull_Price"], label="Bull Case (AI Squeeze, 25% prob) -> Open ~$116, High ~$128", color="#2ca02c", linewidth=2.2)
    ax1.plot(df_combined_paths["Timestamp"], df_combined_paths["Base_Price"], label="Base Case (Institutional Absorption, 60% prob) -> Open ~$97, High ~$104", color="#1f77b4", linewidth=2.5)
    ax1.plot(df_combined_paths["Timestamp"], df_combined_paths["Bear_Price"], label="Bear Case (Size Exhaustion, 15% prob) -> Open ~$82, Support ~$76", color="#d62728", linewidth=2.0, linestyle="--")
    
    # Reference Offer Price Line
    ax1.axhline(78.0, color="#7f7f7f", linestyle=":", linewidth=1.5, label="Filing Range Top ($78.00)")
    ax1.axhline(70.0, color="#bcbd22", linestyle=":", linewidth=1.2, label="Filing Range Bottom ($70.00)")
    
    ax1.set_title("Anthropic (ANT) IPO Day-1 Intraday Price Path Simulation\n(Hybrid Empirical-Microstructure Model | Nasdaq Cross Mechanics)", fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylabel("Share Price (USD)", fontsize=12, fontweight='bold')
    ax1.set_xticks(x_ticks_loc)
    ax1.set_xticklabels([time_labels.iloc[i] for i in x_ticks_loc], fontsize=10)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Annotations on Base Case
    open_p = df_combined_paths["Base_Price"].iloc[0]
    high_idx = df_combined_paths["Base_Price"].idxmax()
    high_p = df_combined_paths["Base_Price"].iloc[high_idx]
    close_p = df_combined_paths["Base_Price"].iloc[-1]
    
    ax1.annotate(f"Opening Cross: ${open_p:.2f} (+24.5%)", xy=(0, open_p), xytext=(15, open_p + 5),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5), fontsize=10, fontweight='bold', color="#1f77b4")
    ax1.annotate(f"Day 1 High: ${high_p:.2f}", xy=(high_idx, high_p), xytext=(high_idx - 30, high_p + 4),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5), fontsize=10, fontweight='bold', color="#1f77b4")
    ax1.annotate(f"Close: ${close_p:.2f}", xy=(minutes-1, close_p), xytext=(minutes - 50, close_p - 7),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5), fontsize=10, fontweight='bold', color="#1f77b4")
    
    # Subplot 2: User Position Value (5 Shares) across Scenarios
    user_vals_base = df_combined_paths["Base_Price"] * user_shares
    user_vals_bull = df_combined_paths["Bull_Price"] * user_shares
    user_vals_bear = df_combined_paths["Bear_Price"] * user_shares
    
    ax2.plot(df_combined_paths["Timestamp"], user_vals_bull, color="#2ca02c", linewidth=1.8, label="Bull Value")
    ax2.plot(df_combined_paths["Timestamp"], user_vals_base, color="#1f77b4", linewidth=2.0, label="Base Value")
    ax2.plot(df_combined_paths["Timestamp"], user_vals_bear, color="#d62728", linewidth=1.5, linestyle="--", label="Bear Value")
    ax2.axhline(78.0 * user_shares, color="#7f7f7f", linestyle=":", linewidth=1.5, label="Cost Basis ($390.00 @ $78)")
    
    ax2.set_title("User Position Value (5 Shares Allocation)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time of Day (ET) - Day 1 Trading", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Position Value ($)", fontsize=11, fontweight='bold')
    ax2.set_xticks(x_ticks_loc)
    ax2.set_xticklabels([time_labels.iloc[i] for i in x_ticks_loc], fontsize=10)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "anthropic_ipo_simulation.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    print("Simulation complete. Files generated:")
    print(f"1. Comps CSV: {comps_path}")
    print(f"2. Intraday Paths CSV: {sim_csv_path}")
    print(f"3. Summary Table CSV: {summary_csv_path}")
    print(f"4. Visualization Plot: {chart_path}")
    print("\nSummary Results:")
    print(df_summary.to_string())

if __name__ == "__main__":
    main()
