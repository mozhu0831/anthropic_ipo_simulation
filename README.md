# Anthropic (`ANT`) IPO Quantitative Pricing & Day-1 Microstructure Engine

An institutional-grade quantitative framework to forecast the initial public offering (IPO) offer price, opening cross mechanics, and Day-1 intraday price path for Anthropic, Inc. (`ANT`) on the Nasdaq Global Select Market.

---

## Overview & Modeling Architecture

Standard valuation methods (e.g., DCF or peer multiples) do not predict opening prints on debut day. Day-1 pricing is dictated by **order book imbalances, allocation rationing, dealer syndicate defense, and retail momentum**. 

This repository implements a **multi-stage empirical and microstructure model**:

1. **Empirical Benchmarking (25 Comps)**: Multi-factor Ridge regression modeling historical underpricing against deal size, oversubscription, retail tranche size, float %, and market volatility (VIX/VXN).
2. **The "Size Drag" Non-Linear Formulation**: Quantifies secondary market liquidity friction on a \$64.4B offering float.
3. **Nasdaq Opening Cross Simulation (NOII)**: Simulates Net Order Imbalance Indicator convergence (Near vs. Far clearing prices) leading to the uncrossing print.
4. **Intraday Microstructure & VWAP Engine**: Minute-by-minute stochastic order flow across 4 intraday market regimes.
5. **Correlated Monte Carlo Engine (10,000 Runs)**: Vectorized stochastic simulation with Cholesky copula linking macro sentiment, book quality, and retail participation.
6. **Retail Execution Optimizer**: Evaluates optimal execution timing for retail allocations (e.g., 5-share positions).

```
                      ┌──────────────────────────────────────────────┐
                      │        Anthropic Deal Parameters             │
                      │ ($70-$78 range, ~7x oversub, $64.4B float)   │
                      └──────────────────────┬───────────────────────┘
                                             │
                      ┌──────────────────────▼───────────────────────┐
                      │    Empirical Multi-Factor Regression         │
                      │  (25 Historical Mega & High-Flyer Comps)     │
                      └──────────────────────┬───────────────────────┘
                                             │
                      ┌──────────────────────▼───────────────────────┐
                      │      Non-Linear "Size Drag" Dampening        │
                      │  (Liquidity constraint for mega offerings)   │
                      └──────────────────────┬───────────────────────┘
                                             │
                 ┌───────────────────────────┴───────────────────────────┐
                 │                                                       │
  ┌──────────────▼──────────────┐                         ┌──────────────▼──────────────┐
  │   Nasdaq NOII Price Cross   │                         │ Correlated Monte Carlo Engine│
  │  (Near/Far convergence)     │                         │  (10,000 Stochastic Iterations)│
  └──────────────┬──────────────┘                         └──────────────┬──────────────┘
                 │                                                       │
                 └───────────────────────────┬───────────────────────────┘
                                             │
                      ┌──────────────────────▼───────────────────────┐
                      │      Intraday Microstructure & VWAP Engine   │
                      │ (Morning Rush -> Flipping -> VWAP -> MOC)    │
                      └──────────────────────┬───────────────────────┘
                                             │
                      ┌──────────────────────▼───────────────────────┐
                      │     Retail Execution Strategy Optimizer      │
                      │ (Almgren-Chriss / Horizon Reservation Model) │
                      └──────────────────────────────────────────────┘
```

---

## Key Empirical Findings

### 1. Offering & Opening Print Forecast

| Parameter | Bear Case (15%) | Base Case (60%) | Bull Case (25%) |
| :--- | :--- | :--- | :--- |
| **Offer Price ($P_{\text{offer}}$)** | \$76.00 | **\$78.00** *(Top of range)* | \$82.00 *(Revised up)* |
| **First Trade Open ($P_{\text{open}}$)** | \$82.46 (+8.5%) | **\$97.05 (+24.4%)** | \$116.44 (+42.0%) |
| **Intraday High** | \$84.88 | **\$104.97** | \$127.42 |
| **Intraday Low** | \$72.56 | **\$94.54** | \$112.46 |
| **Day-1 Close** | \$78.66 (+3.5%) | **\$100.29 (+28.6%)** | \$126.86 (+54.7%) |

### 2. 10,000-Run Monte Carlo Statistical Distributions

* **Mean Opening Pop**: **+21.58%** (Median: **+21.53%**, Std: **5.82%**)
* **Interquartile Range (IQR 25% – 75%)**: **+17.8% to +25.4%**
* **Win Rate ($P(\text{Close} > \text{Offer})$)**: **99.99%**
* **Broken IPO Probability**: **< 0.01%** (underwriter greenshoe defense at 0.975x offer)
* **$P(\text{Pop} > 20\%)$**: **58.76%** | **$P(\text{Pop} > 30\%)$**: **11.17%**
* **$P(\text{Pop} > 50\%)$**: **0.01%** (confirms that size drag mathematically suppresses runaway pops)
* **95% Value at Risk (VaR)**: **\$0.00**

---

## Intraday Microstructure Regimes

1. **Phase 1: Opening Imbalance & Retail Euphoria (10:30 AM – 11:15 AM ET)**
   * Market-on-Open (MOO) buy orders from retail (15–20% allocation tranche) and momentum algorithms create sharp upward price discovery, testing the intraday peak (**\$102–\$105**).
2. **Phase 2: The Flipping Wave (11:15 AM – 1:30 PM ET)**
   * Retail and fast-money allocators take immediate profits. Liquidity dips into midday, driving prices down toward support (**\$94–\$96**).
3. **Phase 3: Institutional VWAP Accumulation (1:30 PM – 3:30 PM ET)**
   * Benchmark institutions and index funds execute automated VWAP algorithms, providing steady absorption.
4. **Phase 4: Closing Cross (3:30 PM – 4:00 PM ET)**
   * Market-on-Close (MOC) rebalancing consolidates the stock around **\$100.29**.

---

## Retail Execution Optimization (5-Share Position)

| Strategy | Timing Window | Order Type | Expected Profit | Sharpe |
| :--- | :--- | :--- | :--- | :--- |
| **1. Immediate Open Cross** | 10:15 – 10:30 AM ET | Market / MOO | **+\$84.50** | 3.71 |
| **2. Golden Window Limit** | **11:00 – 11:45 AM ET** | **Limit (\$103.50)** | **+\$101.47** | **4.25** |
| **3. Trailing Volatility Stop** | Intraday Dynamic | Dynamic (4% Trail) | **+\$98.20** | 3.98 |
| **4. Afternoon VWAP Exit** | 2:30 – 3:45 PM ET | Limit / VWAP | **+\$92.15** | 3.65 |

> **Recommendation**: Strategy 2 captures the peak morning momentum window (+6% above open cross) before the midday flipping wave begins.

---

## Repository Structure

```
Anthropic_IPO/
├── data/
│   ├── historical_tech_ipos.csv          # 25 benchmark tech IPOs with features
│   ├── nasdaq_noii_simulation.csv        # Pre-open clearing price convergence
│   ├── intraday_microstructure_1min.csv  # 1-minute price, volume, VWAP
│   ├── monte_carlo_statistics.csv        # Statistical distribution metrics
│   └── retail_execution_strategies.csv   # Execution comparison matrix
├── figures/
│   ├── comps_size_drag_regression.png    # Deal size vs pop % non-linear curve
│   ├── nasdaq_noii_convergence.png       # Near/Far clearing convergence
│   ├── intraday_microstructure_vwap.png  # Minute-by-minute trading path & VWAP
│   ├── monte_carlo_distribution_dashboard.png # 4-panel Monte Carlo distributions
│   └── optimal_execution_frontier.png    # Strategy P&L and risk frontier
├── src/
│   ├── config.py                         # Centralized dataclass configurations
│   ├── comps_database.py                 # Historical dataset generator
│   ├── empirical_models.py               # Ridge regression & size drag model
│   ├── microstructure.py                 # NOII & intraday order matching engine
│   ├── monte_carlo.py                    # Vectorized 10,000-run simulation engine
│   ├── execution_optimizer.py            # Retail order execution optimizer
│   └── visualizer.py                     # High-res publication plotting engine
├── tests/
│   └── test_model.py                     # Unit test suite (6 test modules)
├── main.py                               # Unified CLI entry point
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Standard gitignore
└── README.md                             # Repository documentation
```

---

## Installation & Usage

### 1. Prerequisites
Python 3.9+ with `numpy`, `pandas`, and `matplotlib`:
```bash
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
```bash
python main.py --run-all
```

### 3. Run Custom Simulation Iterations
```bash
python main.py --simulations 25000
```

### 4. Run Unit Tests
```bash
python tests/test_model.py
```

---

## Author & Disclaimer
* Built for quantitative research and educational modeling of capital market microstructure.
* This is an analytical simulation tool, not direct financial or investment advice.
