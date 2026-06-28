# Data-Driven-Portfolio-Optimizer-
At every monthly rebalance, the optimizer estimates a rolling covariance matrix from the trailing year of returns and solves a constrained quadratic program to pick portfolio weights. Those weights are then applied to the next month's returns — so every number reported is genuinely out-of-sample, with no look-ahead.
