# Stock Backtesting & Analysis Tool

A quantitative finance research tool built in Python that implements a moving average crossover strategy, evaluates performance through rigorous statistical analysis, and simulates future market scenarios through Monte Carlo simulations.

## Features

- **Moving Average Crossover Strategy** — Implements a configurable SMA crossover strategy with buy/sell signal generation
- **Parameter Optimization** — Systematically tests 100+ SMA window combinations to find optimal parameters on in-sample data
- **Overfitting Analysis** — Splits data into in-sample and out-of-sample periods and compares Sharpe ratio degradation to expose overfitting
- **Risk Metrics** — Calculates Sharpe ratio, maximum drawdown, and win rate to provide a complete picture of strategy performance
- **Monte Carlo Simulation** — Runs 1000 simulated market scenarios using Geometric Brownian Motion, applying the full strategy logic to each path to evaluate true strategy probability of profit
- **Visualizations** — Produces clean, professional charts for strategy performance and Monte Carlo results

## Example

The following is an example analysis of the AAPL stock from 2020-2024 using 2022 as the in/out of sample split date:

<img width="1397" height="699" alt="image" src="https://github.com/user-attachments/assets/5aafc41b-3d59-4dab-85f5-7ae766e97932" />


From the visualization, we find that:
- The Sharpe Ratio for the out sample when using the optimized parameters for the in sample drastically drops from 1.63 to 0.2, demonstrating that optimized parameters do not necessarily correlate to a trading edge
- Our crossover strategy would have been outperformed by a buy-and-hold strategy, demonstrating that even with a positive win rate, the strategy is not effective


The Following is a two-figure visualization providing more analysis on our strategy.

Figure 1 is a Monte Carlo visualization of 1000 different paths that the AAPL stock price could have taken using the Geometric Brownian Motion. 

Figure 2 demonstrates how our crossover strategy would have performed in each path.

<img width="1590" height="696" alt="image" src="https://github.com/user-attachments/assets/8dc5b68e-28a7-4151-9a2d-4a0f8a32ae08" />
