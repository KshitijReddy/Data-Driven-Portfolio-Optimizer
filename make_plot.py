import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from portfolio_optimizer import load_universe, backtest, metrics, TRADING

prices = load_universe()
rets = np.log(prices/prices.shift(1)).dropna()

labels = {"equal":"Equal-weight (1/N)","minvar":"Min-variance QP","maxsharpe":"Max-Sharpe QP"}
colors = {"equal":"#9aa0a6","minvar":"#1a73e8","maxsharpe":"#0f9d58"}

# rebuild the out-of-sample dates so the x-axis is real calendar time
points = range(252, len(rets), 21)
oos_dates = []
for t in points:
    hold = rets.iloc[t:t+21]
    if len(hold)==0: break
    oos_dates.extend(list(hold.index))

series = {}
for s in labels:
    daily,_ = backtest(rets, s)
    series[s] = (np.cumprod(1+daily)-1)*100  # cumulative % return

idx = pd.to_datetime(oos_dates[:len(next(iter(series.values())))])

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(11,5.5), dpi=140)
for s in ["equal","minvar","maxsharpe"]:
    m = metrics(backtest(rets,s)[0])
    ax.plot(idx, series[s], label=f"{labels[s]}  (Sharpe {m['sharpe']:.2f})",
            color=colors[s], linewidth=2 if s!="equal" else 1.6,
            linestyle="--" if s=="equal" else "-")
ax.set_title("Out-of-Sample Cumulative Return — S&P 500 (50 assets, 2014–2018)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Cumulative Return (%)"); ax.set_xlabel("")
ax.legend(frameon=True, fontsize=10, loc="upper left")
ax.axhline(0, color="#444", linewidth=0.8)
fig.text(0.99,0.01,"Walk-forward backtest · 252-day rolling window · monthly rebalance · long-only, 10% cap",
         ha="right", fontsize=7.5, color="#777")
plt.tight_layout()
import os; os.makedirs("results",exist_ok=True)
plt.savefig("results/performance.png", bbox_inches="tight")
print("saved results/performance.png")
print("final cumulative returns:", {s: round(float(series[s][-1]),1) for s in labels})
