# Regime-Integrated Cross-Asset Allocation: A Machine Learning Overlay with Alternative Macro Data

Traditional portfolio implementations suffer from well-documented flaws: Mean-Variance Optimization (MVO) acts as an "error maximizer" vulnerable to parameter instability, while Risk Parity frameworks inherently ignore return expectations. This research bridges the gap by mapping alternative macro indicators into regime probabilities, filtering for conviction, and injecting those signals directly into classic allocation optimizers.

## The full research report is available [HERE](https://github.com/yoho369/regime-integrated-asset-allocation/blob/main/Regime-Integrated%20Cross-Asset%20Allocation%20Using%20Alternative%20Macro%20Data.pdf) (last updated on June 03, 2026) 

---

## 1. Core Methodology

The execution engine translates raw macroeconomic data into executable portfolio weights through three operational layers:

### Feature Space: Alternative Macro Data
Instead of relying purely on backward-looking price action, the model ingests a 16-factor feature space covering real-time and forward-looking economic indicators. 
* **Growth:** Real-time Sahm Rule Recession Indicator.
* **Uncertainty:** Global Economic Policy Uncertainty (GEPU) Index.
* **Tail Risk & Volatility:** CBOE Skew Index and ICE BofA MOVE Index.
* **Liquidity & Inflation:** Chicago Fed NFCI and 10-Year Breakeven Inflation.

### The Predictive Engine: Soft-Voting Ensemble
Financial data is noisy and prone to regime shifts that can break single-architecture models. To generate a stable Information Coefficient (IC), the engine utilizes a heterogeneous soft-voting ensemble:
* **Logistic Regression (20%):** Provides a regularized, linear baseline.
* **Random Forest (40%):** Utilizes bagging for variance reduction.
* **XGBoost (40%):** Captures non-linear macro-to-price interactions.

### Friction Management: The Conviction Deadband
To protect downstream optimizers from trading on low-confidence noise, the framework implements a 10% Conviction Deadband. If a predicted regime probability falls within the threshold, the signal is zeroed out. Combined with a portfolio inertia parameter, this ensures capital is only rotated during mathematically significant regime shifts.


---
## 2. Key Empirical Findings (2016 - 2026)

The framework was tested using a walk-forward out-of-sample approach across a 12-asset global universe, applying a conservative 10 bps transaction fee to all rebalancing turns.

### Generating "Crisis Alpha" Post-2020
During the low-volatility 2016-2019 window, the machine learning architectures closely mirrored passive baselines. However, following the shocks in 2020 and 2022, the macro landscape became more volatile. The regime framework thrived in this chaotic environment, executing timely capital rotations that generated sustained outperformance.

### Net Performance Comparison
The integration of the 10% Filtered ML signal delivered systematic out-of-sample improvements across portfolio architectures. 

| Strategy Framework | Signal Type | Net Ann. Return | Net Sharpe | Max Drawdown | Ann. Turnover |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MVO (Long-Short)** | Baseline | 10.68% | 0.56 | -36.26% | 690.2% |
| **MVO (Long-Short)** | ML Filtered | 12.45% | 0.60 | -30.89% | 1235.0% |
| **Volatility Risk Parity** | Baseline | 7.24% | 0.74 | -19.34% | 14.5% |
| **Volatility Risk Parity** | ML Filtered | 9.26% | 0.83 | -20.76% | 163.6% |
| **Pure ML Stat-Arb** | 1/N Baseline | 9.63% | 0.74 | -24.85% | 13.8% |
| **Pure ML Stat-Arb** | ML Filtered | **11.82%** | **0.91** | **-22.69%** | 320.5% |

---


## 3. Repository Directory Structure

* `src/data_engineering.py`: Ingestion and normalization of the 16-factor macroeconomic feature space and asset returns.
* `src/ml_ensemble.py`: Walk-forward cross-validation, XGBoost/RF/LogReg model training, and SHAP value generation.
* `src/portfolio_optimizers.py`: Mathematical solvers for MVO, ERC, and VRP, including alpha-mapping logic.
* `src/execution_engine.py`: Deadband filtering, turnover friction simulation, and inertia tracking.
* `notebooks/`: Jupyter notebooks containing exploratory data analysis, cross-asset IC evaluation, and equity curve generation.

## 4. Reproduction Guide

### Prerequisites
Clone the repository and initialize the environment:
```bash
git clone [https://github.com/yoho369/regime-integrated-asset-allocation.git](https://github.com/yoho369/regime-integrated-asset-allocation.git)
cd regime-integrated-asset-allocation
python3 -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt
