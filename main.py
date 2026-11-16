"""
Anthropic (ANT) IPO Quantitative Pricing & Day-1 Microstructure Modeling Engine.
Main CLI entry point for full pipeline execution, simulation, optimization, and visualization.

Usage:
    python main.py --run-all
    python main.py --simulations 10000
    python main.py --optimize-execution
"""
import os
import sys
import argparse
import json
import pandas as pd

from src.config import AnthropicDealConfig, SimulationConfig
from src.comps_database import build_comps_dataset
from src.empirical_models import EmpiricalPricingModel
from src.microstructure import NasdaqNOIISimulator, IntradayMicrostructureEngine
from src.monte_carlo import MonteCarloEngine
from src.execution_optimizer import ExecutionOptimizer
from src.visualizer import ResearchVisualizer

def main():
    parser = argparse.ArgumentParser(description="Anthropic (ANT) IPO Quantitative Prediction Engine")
    parser.add_argument("--simulations", type=int, default=10_000, help="Number of Monte Carlo simulations")
    parser.add_argument("--run-all", action="store_true", help="Execute complete pipeline (Comps, NOII, Monte Carlo, Optimization, Plots)")
    parser.add_argument("--optimize-execution", action="store_true", help="Run retail execution strategy optimizer")
    args = parser.parse_args()
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "data")
    figures_dir = os.path.join(project_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 75)
    print("  ANTHROPIC (ANT) IPO QUANTITATIVE PRICING & MICROSTRUCTURE ENGINE")
    print("=" * 75)
    
    deal_config = AnthropicDealConfig()
    sim_config = SimulationConfig(n_simulations=args.simulations)
    visualizer = ResearchVisualizer(figures_dir)
    
    # -------------------------------------------------------------------------
    # Stage 1: Historical Comps & Empirical Regression
    # -------------------------------------------------------------------------
    print("\n[1/5] Building Empirical Tech IPO Comps Database (25 Benchmark Deals)...")
    comps_csv_path = os.path.join(data_dir, "historical_tech_ipos.csv")
    df_comps = build_comps_dataset(comps_csv_path)
    
    reg_model = EmpiricalPricingModel(df_comps)
    reg_metrics = reg_model.fit(ridge_alpha=1.5)
    print(f"      Regression Fit -> Open Pop R^2: {reg_metrics['r2_open']:.3f} | Close Return R^2: {reg_metrics['r2_close']:.3f}")
    
    pred_empirical = reg_model.predict_target(
        oversub=deal_config.reported_oversubscription,
        deal_size_b=deal_config.gross_proceeds_high_b,
        retail_pct=deal_config.retail_allocation_pct,
        float_pct=(deal_config.shares_offered * deal_config.filing_price_high / 1e9) / deal_config.implied_valuation_b * 100.0,
        vix=deal_config.pre_ipo_vxn
    )
    print(f"      Anthropic Empirical Forecast -> Raw Pop: +{pred_empirical['raw_pop_open_pct']:.1f}%")
    print(f"      Size Drag Penalty: -{pred_empirical['size_drag_penalty_pct']:.1f}% | Net Dampened Pop: +{pred_empirical['predicted_pop_open_pct']:.1f}%")
    
    fig1 = visualizer.plot_comps_and_size_drag(df_comps, deal_config.gross_proceeds_high_b, pred_empirical['predicted_pop_open_pct'])
    print(f"      Saved Figure: {fig1}")
    
    # -------------------------------------------------------------------------
    # Stage 2: Nasdaq Opening Cross & NOII Simulator
    # -------------------------------------------------------------------------
    print("\n[2/5] Simulating Nasdaq NOII Opening Cross Imbalance Convergence...")
    target_open = deal_config.filing_price_high * (1.0 + pred_empirical['predicted_pop_open_pct'] / 100.0 * 1.35) # include momentum/oversub
    noii = NasdaqNOIISimulator(offer_price=deal_config.filing_price_high, target_open_price=target_open, steps=45)
    df_noii = noii.simulate()
    df_noii.to_csv(os.path.join(data_dir, "nasdaq_noii_simulation.csv"), index=False)
    
    fig2 = visualizer.plot_noii_convergence(df_noii)
    print(f"      Uncrossing Print: ${target_open:.2f} at 10:45 AM ET")
    print(f"      Saved Figure: {fig2}")
    
    # -------------------------------------------------------------------------
    # Stage 3: Intraday Minute-by-Minute Microstructure & VWAP
    # -------------------------------------------------------------------------
    print("\n[3/5] Simulating 1-Minute Intraday Order Flow (10:30 AM - 4:00 PM ET)...")
    intra_engine = IntradayMicrostructureEngine(
        open_price=target_open,
        high_target=target_open * 1.078,
        low_target=target_open * 0.94,
        close_target=target_open * 1.03
    )
    df_intra = intra_engine.simulate_day1_path()
    df_intra.to_csv(os.path.join(data_dir, "intraday_microstructure_1min.csv"), index=False)
    
    fig3 = visualizer.plot_intraday_microstructure(df_intra)
    print(f"      Intraday Day-1 High: ${df_intra['Price'].max():.2f} | Low: ${df_intra['Price'].min():.2f} | Close: ${df_intra['Price'].iloc[-1]:.2f}")
    print(f"      Saved Figure: {fig3}")
    
    # -------------------------------------------------------------------------
    # Stage 4: Correlated Monte Carlo Simulation Engine (10,000 Runs)
    # -------------------------------------------------------------------------
    print(f"\n[4/5] Executing Vectorized Correlated Monte Carlo Engine ({args.simulations:,} Runs)...")
    mc_engine = MonteCarloEngine(deal_config, sim_config)
    df_runs, df_stats, risk_metrics = mc_engine.run()
    
    df_runs.to_csv(os.path.join(data_dir, "monte_carlo_raw_runs.csv"), index=False)
    df_stats.to_csv(os.path.join(data_dir, "monte_carlo_statistics.csv"))
    with open(os.path.join(data_dir, "monte_carlo_risk_metrics.json"), "w") as f:
        json.dump(risk_metrics, f, indent=4)
        
    print(f"      Monte Carlo Mean Open Pop: {df_stats.loc['Open_Pop_Pct', 'Mean']:+.2f}% (Median: {df_stats.loc['Open_Pop_Pct', 'Median']:+.2f}%)")
    print(f"      Win Rate P(Close > Offer): {risk_metrics['P_Close_Above_Offer_WinRate']}")
    print(f"      Probability of Pop > 20%: {risk_metrics['P_Open_Pop_Greater_20Pct']} | Pop > 30%: {risk_metrics['P_Open_Pop_Greater_30Pct']}")
    print(f"      95% Value at Risk (VaR): {risk_metrics['VaR_95_Dollar_Loss']}")
    
    fig4 = visualizer.plot_monte_carlo_dashboard(df_runs, risk_metrics)
    print(f"      Saved Figure: {fig4}")
    
    # -------------------------------------------------------------------------
    # Stage 5: Retail Execution Strategy Optimizer
    # -------------------------------------------------------------------------
    print("\n[5/5] Optimizing Retail Execution Strategies for 5-Share Position...")
    opt = ExecutionOptimizer(df_runs, shares=deal_config.user_shares_entered)
    df_strategies = opt.evaluate_strategies()
    df_strategies.to_csv(os.path.join(data_dir, "retail_execution_strategies.csv"), index=False)
    
    fig5 = visualizer.plot_execution_frontier(df_strategies)
    print(f"      Optimal Strategy: {df_strategies.iloc[1]['Strategy']} -> Expected Profit: {df_strategies.iloc[1]['Expected_Profit']}")
    print(f"      Saved Figure: {fig5}")
    
    print("\n" + "=" * 75)
    print("  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY")
    print(f"  Artifacts saved in: {project_dir}")
    print("=" * 75)

if __name__ == "__main__":
    main()
