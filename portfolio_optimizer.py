"""
Data-Driven Portfolio Optimization
----------------------------------
Walk-forward (out-of-sample) backtest of constrained mean-variance portfolios
on real S&P 500 daily prices, benchmarked against an equal-weight (1/N) baseline.

Methodology
-----------
- Universe : 50 most-liquid S&P 500 names (by median dollar volume) with full history
- Returns  : daily log returns
- Estimation : 252-day (1y) rolling window for covariance & mean
- Rebalance  : every 21 trading days (~monthly)
- Optimizer  : constrained QP (long-only, sum(w)=1, w<=10% cap) via SLSQP
                 * min-variance  : min w'Σw          (no reliance on mean estimates)
                 * max-Sharpe    : max (w'μ)/sqrt(w'Σw)
- Baseline   : equal-weight 1/N, same rebalance schedule
- Evaluation : out-of-sample only — weights come from the trailing window and are
               applied to the *forward* holding period (no look-ahead).

Run:  python portfolio_optimizer.py
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize

DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"
TRADING, LOOKBACK, REBAL, CAP, N_ASSETS = 252, 252, 21, 0.10, 50


def load_universe(n=N_ASSETS):
    df = pd.read_csv(DATA_URL, parse_dates=["date"])
    close = df.pivot(index="date", columns="Name", values="close").sort_index()
    vol   = df.pivot(index="date", columns="Name", values="volume")
    full  = close.columns[close.notna().all()]            # complete history only
    liq   = (close[full] * vol[full]).median().sort_values(ascending=False)
    universe = sorted(liq.head(n).index.tolist())          # deterministic selection
    return close[universe]


def solve_qp(cov, mu=None, objective="minvar"):
    n = len(cov)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    bnds = [(0, CAP)] * n
    x0 = np.repeat(1 / n, n)
    if objective == "minvar":
        f = lambda w: w @ cov @ w
    else:  # max-Sharpe -> minimise negative Sharpe (rf = 0)
        def f(w):
            v = np.sqrt(w @ cov @ w)
            return -(w @ mu) / v if v > 0 else 0.0
    return minimize(f, x0, method="SLSQP", bounds=bnds, constraints=cons,
                    options={"maxiter": 500, "ftol": 1e-12}).x


def backtest(rets, strategy):
    n = rets.shape[1]
    points = range(LOOKBACK, len(rets), REBAL)
    oos, weights = [], []
    for t in points:
        train = rets.iloc[t - LOOKBACK:t]
        cov, mu = train.cov().values * TRADING, train.mean().values * TRADING
        if   strategy == "equal":     w = np.repeat(1 / n, n)
        elif strategy == "minvar":    w = solve_qp(cov, objective="minvar")
        elif strategy == "maxsharpe": w = solve_qp(cov, mu, objective="maxsharpe")
        hold = rets.iloc[t:t + REBAL]
        if len(hold) == 0:
            break
        oos.extend((hold.values @ w).tolist())
        weights.append(w)
    return np.array(oos), np.array(weights)


def metrics(daily, rf=0.0):
    ann_ret = daily.mean() * TRADING
    ann_vol = daily.std(ddof=1) * np.sqrt(TRADING)
    cum = np.cumprod(1 + daily)
    mdd = ((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min()
    return dict(ann_ret=ann_ret, ann_vol=ann_vol,
                sharpe=(ann_ret - rf) / ann_vol, max_dd=mdd, n_days=len(daily))


def main():
    prices = load_universe()
    rets = np.log(prices / prices.shift(1)).dropna()
    print(f"Universe: {prices.shape[1]} assets | {len(prices)} trading days | "
          f"{prices.size:,} price points\n")

    labels = {"equal": "Equal-weight (1/N)", "minvar": "Min-variance QP",
              "maxsharpe": "Max-Sharpe QP"}
    res = {s: backtest(rets, s) for s in labels}

    print(f"{'Strategy':<22}{'AnnRet':>9}{'AnnVol':>9}{'Sharpe':>9}{'MaxDD':>9}")
    for s, name in labels.items():
        m = metrics(res[s][0])
        print(f"{name:<22}{m['ann_ret']*100:>8.2f}%{m['ann_vol']*100:>8.2f}%"
              f"{m['sharpe']:>9.3f}{m['max_dd']*100:>8.2f}%")

    sh = {s: metrics(res[s][0])["sharpe"] for s in labels}
    vol = {s: metrics(res[s][0])["ann_vol"] for s in labels}
    print(f"\nMax-Sharpe vs 1/N : Sharpe {(sh['maxsharpe']/sh['equal']-1)*100:+.1f}%")
    print(f"Min-var  vs 1/N : volatility {(1-vol['minvar']/vol['equal'])*100:+.1f}% (lower)")


if __name__ == "__main__":
    main()
