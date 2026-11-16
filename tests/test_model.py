"""
Unit Tests for Anthropic IPO Quantitative Engine (Runs via python tests/test_model.py or pytest).
Validates numerical stability, arbitrage bounds, and simulation integrity.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from src.config import AnthropicDealConfig, SimulationConfig
from src.comps_database import build_comps_dataset
from src.empirical_models import EmpiricalPricingModel
from src.microstructure import NasdaqNOIISimulator, IntradayMicrostructureEngine
from src.monte_carlo import MonteCarloEngine
from src.execution_optimizer import ExecutionOptimizer

def test_deal_config():
    config = AnthropicDealConfig()
    assert config.filing_price_low > 0
    assert config.filing_price_high > config.filing_price_low
    assert config.shares_offered > 0
    assert config.user_shares_entered == 5
    print("test_deal_config PASSED")

def test_comps_database():
    df = build_comps_dataset()
    assert len(df) >= 20
    assert "Pop_Open_Pct" in df.columns
    assert (df["Offer_Price"] > 0).all()
    assert (df["Day1_High"] >= df["Day1_Low"]).all()
    print("test_comps_database PASSED")

def test_empirical_pricing_model():
    df = build_comps_dataset()
    model = EmpiricalPricingModel(df)
    fit_res = model.fit(ridge_alpha=1.0)
    assert "r2_open" in fit_res
    assert fit_res["r2_open"] > 0.30
    
    pred = model.predict_target(oversub=7.0, deal_size_b=64.35, retail_pct=17.5, float_pct=4.6, vix=18.5)
    assert 5.0 <= pred["predicted_pop_open_pct"] <= 50.0
    assert pred["size_drag_penalty_pct"] >= 0.0
    print("test_empirical_pricing_model PASSED")

def test_nasdaq_noii_simulation():
    noii = NasdaqNOIISimulator(offer_price=78.0, target_open_price=97.4, steps=30)
    df = noii.simulate()
    assert len(df) == 30
    assert abs(df["Near_Clearing_Price"].iloc[-1] - 97.4) < 0.05
    assert abs(df["Far_Clearing_Price"].iloc[-1] - 97.4) < 0.05
    assert df["Imbalance_Shares"].iloc[-1] == 0
    print("test_nasdaq_noii_simulation PASSED")

def test_monte_carlo_engine():
    deal = AnthropicDealConfig()
    sim = SimulationConfig(n_simulations=500)
    engine = MonteCarloEngine(deal, sim)
    runs, stats, risk = engine.run()
    
    assert len(runs) == 500
    assert (runs["Open_Price"] > 0).all()
    assert (runs["Day1_High"] >= runs["Day1_Low"]).all()
    assert "P_Close_Above_Offer_WinRate" in risk
    print("test_monte_carlo_engine PASSED")

def test_execution_optimizer():
    deal = AnthropicDealConfig()
    sim = SimulationConfig(n_simulations=200)
    engine = MonteCarloEngine(deal, sim)
    runs, _, _ = engine.run()
    
    opt = ExecutionOptimizer(runs, shares=5)
    df_strat = opt.evaluate_strategies()
    assert len(df_strat) == 4
    assert "Strategy" in df_strat.columns
    print("test_execution_optimizer PASSED")

if __name__ == "__main__":
    print("Running suite of unit tests...")
    test_deal_config()
    test_comps_database()
    test_empirical_pricing_model()
    test_nasdaq_noii_simulation()
    test_monte_carlo_engine()
    test_execution_optimizer()
    print("\nALL 6 TEST SUITES PASSED SUCCESSFULLY!")
