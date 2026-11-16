"""
Comprehensive Historical Tech & Mega-Cap IPO Database (25 Benchmark Offerings).
Provides empirical data for regression calibration and size-drag estimation.
"""
import os
import pandas as pd
from typing import Optional

def build_comps_dataset(output_path: Optional[str] = None) -> pd.DataFrame:
    comps = [
        # Mega Offerings ($10B+ Float)
        {"Ticker": "BABA", "Company": "Alibaba Group", "Year": 2014, "Deal_Size_B": 21.8, "Valuation_B": 168.0, "Float_Pct": 13.0, "Oversubscription_x": 10.0, "Retail_Pct": 5.0,  "VIX_Pricing": 12.8, "Offer_Price": 68.0,  "Open_Price": 92.70,  "Day1_High": 99.70,  "Day1_Low": 89.95,  "Day1_Close": 93.89},
        {"Ticker": "FB",   "Company": "Meta Platforms", "Year": 2012, "Deal_Size_B": 16.0, "Valuation_B": 104.0, "Float_Pct": 15.4, "Oversubscription_x": 7.0,  "Retail_Pct": 25.0, "VIX_Pricing": 21.4, "Offer_Price": 38.0,  "Open_Price": 42.05,  "Day1_High": 45.00,  "Day1_Low": 38.00,  "Day1_Close": 38.23},
        {"Ticker": "RIVN", "Company": "Rivian Automotive", "Year": 2021, "Deal_Size_B": 11.9, "Valuation_B": 66.5, "Float_Pct": 17.9, "Oversubscription_x": 8.0,  "Retail_Pct": 10.0, "VIX_Pricing": 17.6, "Offer_Price": 78.0,  "Open_Price": 106.75, "Day1_High": 119.46, "Day1_Low": 95.20,  "Day1_Close": 100.73},
        {"Ticker": "V",    "Company": "Visa Inc.", "Year": 2008, "Deal_Size_B": 17.9, "Valuation_B": 86.0,  "Float_Pct": 20.8, "Oversubscription_x": 5.5,  "Retail_Pct": 6.0,  "VIX_Pricing": 25.6, "Offer_Price": 44.0,  "Open_Price": 59.50,  "Day1_High": 69.00,  "Day1_Low": 56.50,  "Day1_Close": 56.50},
        {"Ticker": "UBER", "Company": "Uber Technologies", "Year": 2019, "Deal_Size_B": 8.1,  "Valuation_B": 75.5,  "Float_Pct": 10.7, "Oversubscription_x": 3.0,  "Retail_Pct": 5.0,  "VIX_Pricing": 19.1, "Offer_Price": 45.0,  "Open_Price": 42.00,  "Day1_High": 45.00,  "Day1_Low": 41.06,  "Day1_Close": 41.57},
        {"Ticker": "ARM",  "Company": "Arm Holdings", "Year": 2023, "Deal_Size_B": 4.87, "Valuation_B": 54.5,  "Float_Pct": 9.4,  "Oversubscription_x": 6.0,  "Retail_Pct": 8.0,  "VIX_Pricing": 13.5, "Offer_Price": 51.0,  "Open_Price": 56.10,  "Day1_High": 66.75,  "Day1_Low": 55.02,  "Day1_Close": 60.75},
        
        # High-Flyer SaaS & AI Precedents
        {"Ticker": "SNOW", "Company": "Snowflake Inc.", "Year": 2020, "Deal_Size_B": 3.36, "Valuation_B": 33.3,  "Float_Pct": 10.1, "Oversubscription_x": 15.0, "Retail_Pct": 4.0,  "VIX_Pricing": 26.0, "Offer_Price": 120.0, "Open_Price": 245.0,  "Day1_High": 319.0,  "Day1_Low": 231.0,  "Day1_Close": 253.93},
        {"Ticker": "ABNB", "Company": "Airbnb, Inc.", "Year": 2020, "Deal_Size_B": 3.50, "Valuation_B": 47.0,  "Float_Pct": 7.4,  "Oversubscription_x": 12.0, "Retail_Pct": 7.0,  "VIX_Pricing": 22.5, "Offer_Price": 68.0,  "Open_Price": 146.00, "Day1_High": 165.00, "Day1_Low": 141.25, "Day1_Close": 144.71},
        {"Ticker": "DASH", "Company": "DoorDash, Inc.", "Year": 2020, "Deal_Size_B": 3.37, "Valuation_B": 39.0,  "Float_Pct": 8.6,  "Oversubscription_x": 11.0, "Retail_Pct": 6.0,  "VIX_Pricing": 20.8, "Offer_Price": 102.0, "Open_Price": 182.00, "Day1_High": 195.50, "Day1_Low": 163.80, "Day1_Close": 189.51},
        {"Ticker": "GOOG", "Company": "Alphabet (Google)", "Year": 2004, "Deal_Size_B": 1.67, "Valuation_B": 23.1,  "Float_Pct": 7.2,  "Oversubscription_x": 4.5,  "Retail_Pct": 18.0, "VIX_Pricing": 15.2, "Offer_Price": 85.0,  "Open_Price": 100.01, "Day1_High": 104.06, "Day1_Low": 95.96,  "Day1_Close": 100.34},
        
        # Recent High-Profile AI & Semiconductor Listings (2023-2024)
        {"Ticker": "ALAB", "Company": "Astera Labs", "Year": 2024, "Deal_Size_B": 0.71, "Valuation_B": 5.5,   "Float_Pct": 13.0, "Oversubscription_x": 14.0, "Retail_Pct": 5.0,  "VIX_Pricing": 13.0, "Offer_Price": 36.0,  "Open_Price": 52.56,  "Day1_High": 63.50,  "Day1_Low": 50.61,  "Day1_Close": 62.03},
        {"Ticker": "RDDT", "Company": "Reddit, Inc.", "Year": 2024, "Deal_Size_B": 0.75, "Valuation_B": 6.4,   "Float_Pct": 11.7, "Oversubscription_x": 5.0,  "Retail_Pct": 8.0,  "VIX_Pricing": 13.1, "Offer_Price": 34.0,  "Open_Price": 47.00,  "Day1_High": 57.80,  "Day1_Low": 45.05,  "Day1_Close": 50.44},
        {"Ticker": "TEM",  "Company": "Tempus AI", "Year": 2024, "Deal_Size_B": 0.41, "Valuation_B": 6.1,   "Float_Pct": 6.7,  "Oversubscription_x": 6.5,  "Retail_Pct": 5.0,  "VIX_Pricing": 12.6, "Offer_Price": 37.0,  "Open_Price": 40.00,  "Day1_High": 44.40,  "Day1_Low": 38.00,  "Day1_Close": 40.25},
        {"Ticker": "RBRK", "Company": "Rubrik, Inc.", "Year": 2024, "Deal_Size_B": 0.75, "Valuation_B": 5.6,   "Float_Pct": 13.4, "Oversubscription_x": 7.0,  "Retail_Pct": 5.0,  "VIX_Pricing": 15.2, "Offer_Price": 32.0,  "Open_Price": 38.60,  "Day1_High": 40.00,  "Day1_Low": 36.50,  "Day1_Close": 37.00},
        {"Ticker": "CART", "Company": "Instacart (Maple)", "Year": 2023, "Deal_Size_B": 0.66, "Valuation_B": 9.9,   "Float_Pct": 6.7,  "Oversubscription_x": 4.5,  "Retail_Pct": 6.0,  "VIX_Pricing": 14.1, "Offer_Price": 30.0,  "Open_Price": 42.00,  "Day1_High": 42.95,  "Day1_Low": 33.70,  "Day1_Close": 33.70},
        
        # High Retail Tranche & Direct Listing Comparables
        {"Ticker": "HOOD", "Company": "Robinhood Markets", "Year": 2021, "Deal_Size_B": 2.10, "Valuation_B": 32.0,  "Float_Pct": 6.5,  "Oversubscription_x": 4.0,  "Retail_Pct": 23.0, "VIX_Pricing": 18.0, "Offer_Price": 38.0,  "Open_Price": 38.00,  "Day1_High": 40.25,  "Day1_Low": 33.25,  "Day1_Close": 34.82},
        {"Ticker": "CPNG", "Company": "Coupang, Inc.", "Year": 2021, "Deal_Size_B": 4.60, "Valuation_B": 60.0,  "Float_Pct": 7.7,  "Oversubscription_x": 8.0,  "Retail_Pct": 5.0,  "VIX_Pricing": 21.9, "Offer_Price": 35.0,  "Open_Price": 63.50,  "Day1_High": 69.00,  "Day1_Low": 49.25,  "Day1_Close": 49.52},
        {"Ticker": "GTLB", "Company": "GitLab Inc.", "Year": 2021, "Deal_Size_B": 0.65, "Valuation_B": 11.0,  "Float_Pct": 5.9,  "Oversubscription_x": 9.5,  "Retail_Pct": 4.0,  "VIX_Pricing": 16.5, "Offer_Price": 77.0,  "Open_Price": 94.25,  "Day1_High": 104.99, "Day1_Low": 91.00,  "Day1_Close": 103.89},
        {"Ticker": "HCP",  "Company": "HashiCorp", "Year": 2021, "Deal_Size_B": 1.20, "Valuation_B": 14.0,  "Float_Pct": 8.6,  "Oversubscription_x": 7.5,  "Retail_Pct": 4.0,  "VIX_Pricing": 19.8, "Offer_Price": 80.0,  "Open_Price": 81.16,  "Day1_High": 89.50,  "Day1_Low": 79.50,  "Day1_Close": 85.19},
        {"Ticker": "PLTR", "Company": "Palantir (Direct)", "Year": 2020, "Deal_Size_B": 2.50, "Valuation_B": 21.0,  "Float_Pct": 12.0, "Oversubscription_x": 5.0,  "Retail_Pct": 15.0, "VIX_Pricing": 26.3, "Offer_Price": 7.25,  "Open_Price": 10.00,  "Day1_High": 11.41,  "Day1_Low": 9.11,   "Day1_Close": 9.50},
        {"Ticker": "COIN", "Company": "Coinbase (Direct)", "Year": 2021, "Deal_Size_B": 3.00, "Valuation_B": 65.0,  "Float_Pct": 8.0,  "Oversubscription_x": 6.0,  "Retail_Pct": 20.0, "VIX_Pricing": 17.0, "Offer_Price": 250.0, "Open_Price": 381.00, "Day1_High": 429.54, "Day1_Low": 310.00, "Day1_Close": 328.28},
        {"Ticker": "TWLO", "Company": "Twilio Inc.", "Year": 2016, "Deal_Size_B": 0.15, "Valuation_B": 1.2,   "Float_Pct": 12.5, "Oversubscription_x": 10.0, "Retail_Pct": 4.0,  "VIX_Pricing": 17.3, "Offer_Price": 15.0,  "Open_Price": 23.99,  "Day1_High": 28.99,  "Day1_Low": 23.66,  "Day1_Close": 28.79},
        {"Ticker": "MDB",  "Company": "MongoDB, Inc.", "Year": 2017, "Deal_Size_B": 0.19, "Valuation_B": 1.2,   "Float_Pct": 15.8, "Oversubscription_x": 8.0,  "Retail_Pct": 4.0,  "VIX_Pricing": 10.1, "Offer_Price": 24.0,  "Open_Price": 33.00,  "Day1_High": 34.00,  "Day1_Low": 30.50,  "Day1_Close": 32.07},
        {"Ticker": "CRWD", "Company": "CrowdStrike", "Year": 2019, "Deal_Size_B": 0.61, "Valuation_B": 6.7,   "Float_Pct": 9.1,  "Oversubscription_x": 12.0, "Retail_Pct": 4.0,  "VIX_Pricing": 16.0, "Offer_Price": 34.0,  "Open_Price": 63.50,  "Day1_High": 67.00,  "Day1_Low": 56.00,  "Day1_Close": 58.00},
        {"Ticker": "ZM",   "Company": "Zoom Video", "Year": 2019, "Deal_Size_B": 0.35, "Valuation_B": 9.2,   "Float_Pct": 3.8,  "Oversubscription_x": 14.0, "Retail_Pct": 5.0,  "VIX_Pricing": 12.3, "Offer_Price": 36.0,  "Open_Price": 65.00,  "Day1_High": 66.00,  "Day1_Low": 60.32,  "Day1_Close": 62.28},
    ]
    df = pd.DataFrame(comps)
    df["Pop_Open_Pct"] = (df["Open_Price"] - df["Offer_Price"]) / df["Offer_Price"] * 100.0
    df["Pop_Close_Pct"] = (df["Day1_Close"] - df["Offer_Price"]) / df["Offer_Price"] * 100.0
    df["Intraday_Overshoot_Pct"] = (df["Day1_High"] - df["Open_Price"]) / df["Open_Price"] * 100.0
    df["Intraday_Drawdown_Pct"] = (df["Day1_High"] - df["Day1_Low"]) / df["Day1_High"] * 100.0
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
    return df

if __name__ == "__main__":
    db_path = r"D:\Professional\Personal Projects\Anthropic_IPO\data\historical_tech_ipos.csv"
    df = build_comps_dataset(db_path)
    print(f"Loaded {len(df)} comps to {db_path}")
