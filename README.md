# Data-Driven Portfolio Optimization

Walk-forward, out-of-sample backtest of constrained mean-variance portfolios on real
S&P 500 daily prices, benchmarked against an equal-weight (1/N) baseline.

## What it does
At every monthly rebalance, the optimizer estimates a rolling covariance matrix from the
trailing year of returns and solves a constrained quadratic program to pick portfolio
weights. Those weights are then applied to the *next* month's returns — so every number
reported is genuinely out-of-sample, with no look-ahead.

## Data
- Source: S&P 500 daily OHLCV, Feb 2013 – Feb 2018 (`plotly/datasets`, fetched at runtime)
- Universe: 50 most-liquid names (by median dollar volume) with complete history
- 1,259 trading days × 50 assets = 62,950 price points

## Method
| Component | Choice |
|---|---|
| Returns | daily log returns |
| Estimation window | 252 trading days (rolling) |
| Rebalance | every 21 trading days (~monthly), 48 rebalances |
| Constraints | long-only, weights sum to 1, 10% per-name cap |
| Solver | SLSQP (`scipy.optimize`) |
| Strategies | min-variance QP, max-Sharpe QP, equal-weight baseline |
| Evaluation | 1,006 out-of-sample trading days |

## Results (out-of-sample)
| Strategy | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|
| Equal-weight (1/N) | 9.43% | 13.15% | 0.717 | −17.31% |
| Min-variance QP | 6.73% | 10.48% | 0.642 | −13.88% |
| Max-Sharpe QP | 11.85% | 14.59% | **0.812** | −15.27% |

- **Max-Sharpe** improved risk-adjusted return (Sharpe) by **13.3%** over the 1/N baseline.
- **Min-variance** cut volatility by **20.3%** and reduced max drawdown by ~3.4pp, at the
  cost of some return — a pure risk-control profile.

## Honest caveats (read before quoting numbers in an interview)
- Results are from a **single historical period** (2014–2018, a strong bull market for the
  back half). A backtest is not a forecast.
- The optimizers' edge is **regime-dependent**: splitting the out-of-sample window shows both
  optimizers beat 1/N decisively in the choppier first half (Sharpe 0.38 / 0.33 vs 0.02) and
  *lose* to it in the raging second-half bull market. Risk control costs return when
  everything goes up.
- Max-Sharpe relies on noisy expected-return estimates and is less robust than min-variance;
  the min-variance volatility reduction is the more dependable finding.
- No transaction costs, slippage, or shorting modelled.

## Run
```bash
pip install pandas numpy scipy
python portfolio_optimizer.py
```
Deterministic — reproduces the table above on every run.
