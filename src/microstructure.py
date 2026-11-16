"""
Nasdaq Opening Cross & Intraday Market Microstructure Engine.
Simulates:
1. Nasdaq NOII (Net Order Imbalance Indicator) clearing price convergence.
2. 1-minute order book dynamics, VWAP, and volume profile across trading regimes.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

class NasdaqNOIISimulator:
    """
    Simulates the Nasdaq Halt Cross / Opening Cross price discovery mechanism.
    Tracks Far Clearing Price, Near Clearing Price, Matched Volume, and Order Imbalance.
    """
    def __init__(self, offer_price: float = 78.0, target_open_price: float = 97.4, steps: int = 45):
        self.offer_price = offer_price
        self.target_open = target_open_price
        self.steps = steps # Minutes from 10:00 to 10:45 AM ET
        
    def simulate(self, random_seed: int = 42) -> pd.DataFrame:
        np.random.seed(random_seed)
        t = np.linspace(0, 1, self.steps)
        
        # Far clearing price begins erratic and volatile with early market-only orders
        far_noise = np.random.normal(0, 4.0, self.steps) * (1.0 - t * 0.8)
        far_price = self.target_open + (self.offer_price - self.target_open) * 0.4 * (1.0 - t)**1.5 + far_noise
        far_price = np.maximum(far_price, self.offer_price)
        
        # Near clearing price steadily converges to the final crossing print
        near_noise = np.random.normal(0, 1.2, self.steps) * (1.0 - t)
        near_price = self.offer_price + (self.target_open - self.offer_price) * (1.0 - np.exp(-3.5 * t)) + near_noise
        
        # Ensure exact convergence at the cross
        far_price[-1] = self.target_open
        near_price[-1] = self.target_open
        
        # Current reference price (Nasdaq best bid/offer midpoint during halt)
        ref_price = (far_price + near_price) / 2.0
        
        # Matched shares scale up non-linearly as deadline approaches
        total_float = 825_000_000
        paired_shares = (15_000_000 + 45_000_000 * (t**2.5) + np.random.normal(0, 500_000, self.steps)).astype(int)
        paired_shares = np.maximum(paired_shares, 1_000_000)
        
        # Imbalance shares (starts with high Buy Imbalance 'B', shrinks to 0 at the cross)
        imbalance_shares = (35_000_000 * (1.0 - t)**1.8 + np.random.normal(0, 800_000, self.steps)).astype(int)
        imbalance_shares = np.maximum(imbalance_shares, 0)
        imbalance_shares[-1] = 0
        
        minutes_str = [f"10:{i:02d} AM" for i in range(self.steps)]
        
        df = pd.DataFrame({
            "Time": minutes_str,
            "Reference_Price": ref_price,
            "Near_Clearing_Price": near_price,
            "Far_Clearing_Price": far_price,
            "Paired_Shares": paired_shares,
            "Imbalance_Shares": imbalance_shares,
            "Imbalance_Side": ["B" if s > 0 else "N" for s in imbalance_shares]
        })
        return df

class IntradayMicrostructureEngine:
    """
    Simulates minute-by-minute trading path from 10:30 AM to 4:00 PM (330 minutes).
    Models regime-shifting order arrivals, volume profile, bid-ask spread, and VWAP.
    """
    def __init__(self, open_price: float, high_target: float, low_target: float, close_target: float, total_minutes: int = 330):
        self.p_open = open_price
        self.p_high = high_target
        self.p_low = low_target
        self.p_close = close_target
        self.minutes = total_minutes
        
    def simulate_day1_path(self, random_seed: int = 42) -> pd.DataFrame:
        np.random.seed(random_seed)
        prices = np.zeros(self.minutes)
        volumes = np.zeros(self.minutes)
        spreads = np.zeros(self.minutes)
        
        prices[0] = self.p_open
        
        # Base intraday regimes:
        # Phase 1: 0 - 45 min (10:30 - 11:15 AM) Euphoria / MOO rush
        # Phase 2: 45 - 180 min (11:15 - 1:30 PM) Flipping / Lunch lull
        # Phase 3: 180 - 300 min (1:30 - 3:30 PM) Institutional VWAP accumulation
        # Phase 4: 300 - 330 min (3:30 - 4:00 PM) Closing cross MOC rebalance
        
        for t in range(1, self.minutes):
            if t <= 45:
                # Upward drift toward high
                drift = (self.p_high - self.p_open) / 45.0
                volatility = 0.40
                base_vol = 450_000 + np.random.exponential(250_000)
                spread = 0.08 + 0.05 * (1.0 - t / 45.0)
            elif t <= 180:
                # Downward drift toward low
                drift = (self.p_low - self.p_high) / (180 - 45)
                volatility = 0.28
                base_vol = 180_000 + np.random.exponential(100_000)
                spread = 0.04 + np.random.uniform(0, 0.03)
            elif t <= 300:
                # Upward drift toward close
                drift = (self.p_close - self.p_low) / (300 - 180)
                volatility = 0.22
                base_vol = 280_000 + np.random.exponential(150_000)
                spread = 0.03 + np.random.uniform(0, 0.02)
            else:
                # Final closing cross positioning
                drift = (self.p_close - prices[t-1]) / (self.minutes - t + 1)
                volatility = 0.18
                base_vol = 600_000 + np.random.exponential(400_000) # Closing volume surge
                spread = 0.02 + np.random.uniform(0, 0.02)
                
            shock = np.random.normal(0, volatility)
            prices[t] = prices[t-1] + drift * 0.95 + shock
            volumes[t] = base_vol
            spreads[t] = spread
            
        volumes[0] = 60_000_000 # Uncrossing print volume
        spreads[0] = 0.15 # Wide opening spread
        
        # Calculate VWAP
        cum_dollar_vol = np.cumsum(prices * volumes)
        cum_vol = np.cumsum(volumes)
        vwap = cum_dollar_vol / cum_vol
        
        # Timestamps
        from datetime import datetime, timedelta
        start_time = datetime(2026, 11, 19, 10, 30)
        timestamps = [(start_time + timedelta(minutes=i)).strftime("%H:%M") for i in range(self.minutes)]
        
        df = pd.DataFrame({
            "Timestamp": timestamps,
            "Price": prices,
            "VWAP": vwap,
            "Volume": volumes.astype(int),
            "Bid_Ask_Spread": spreads
        })
        return df

if __name__ == "__main__":
    noii = NasdaqNOIISimulator()
    df_noii = noii.simulate()
    print("NOII Simulation Sample:")
    print(df_noii.head(5))
    print(df_noii.tail(5))
    
    intra = IntradayMicrostructureEngine(open_price=97.4, high_target=105.0, low_target=95.0, close_target=100.4)
    df_intra = intra.simulate_day1_path()
    print("\nIntraday Microstructure Sample:")
    print(df_intra.head(5))
    print(df_intra.tail(5))
