"""
Anthropic (ANT) IPO Modeling Configuration
Centralized parameters for deal structure, macro environment, and retail positions.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class AnthropicDealConfig:
    # Basic Offering Details
    ticker: str = "ANT"
    company_name: str = "Anthropic, Inc."
    exchange: str = "Nasdaq Global Select Market"
    expected_pricing_date: str = "2026-11-18"
    expected_trading_date: str = "2026-11-19"
    
    # Capital Structure & Range
    filing_price_low: float = 70.00
    filing_price_high: float = 78.00
    shares_offered: int = 825_000_000       # Class A shares in public IPO
    private_placement_b: float = 26.0       # $26B concurrent strategic placement (Google, Nvidia)
    greenshoe_pct: float = 0.15             # Standard 15% overallotment
    
    # Valuation & Demand Indicators
    implied_valuation_b: float = 1400.0     # ~$1.4T targeted market cap
    reported_oversubscription: float = 7.0  # ~7x book coverage (~$450B indicative demand)
    retail_allocation_pct: float = 17.5     # Midpoint of 15-20% retail tranche
    strategic_backing_lockup_months: int = 12
    
    # Market & Macro Regime
    pre_ipo_vxn: float = 18.5               # Nasdaq-100 Volatility Index benchmark
    macro_tech_sentiment_score: float = 0.75 # Scale 0.0 to 1.0 (Bullish AI backdrop)
    
    # Retail User Position
    user_shares_entered: int = 5
    
    @property
    def midpoint_price(self) -> float:
        return (self.filing_price_low + self.filing_price_high) / 2.0
    
    @property
    def gross_proceeds_high_b(self) -> float:
        return (self.shares_offered * self.filing_price_high) / 1e9

@dataclass
class SimulationConfig:
    n_simulations: int = 10_000
    random_seed: int = 42
    trading_minutes: int = 330              # 10:30 AM to 4:00 PM ET
    noii_steps: int = 45                    # 10:00 AM to 10:45 AM ET (pre-opening cross)
    confidence_levels: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])
