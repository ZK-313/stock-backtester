import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class StockBacktester:
    def __init__(self, symbol, start_date, end_date, split_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.split_date = split_date
        self.short_window = 20
        self.long_window = 50
        self.data = None
        self.in_sample = None
        self.out_sample = None


    def optimize_parameters(self, short_windows, long_windows, dataset):
        results = []
        
        for short in short_windows:
            for long in long_windows:
                if short >= long:
                    continue
                    
                data = dataset.copy()
                data[f'SMA{short}'] = data['Close'].rolling(window=short).mean()
                data[f'SMA{long}'] = data['Close'].rolling(window=long).mean()
                
                signal = (data[f'SMA{short}'] > data[f'SMA{long}']).astype(int)
                strategy_returns = data['Close'].pct_change() * signal.shift(1)
                
                excess_returns = strategy_returns - (0.02 / 252)
                sharpe = (excess_returns.mean() / excess_returns.std()) * (252 ** 0.5)
                
                results.append({
                    'short_window': short,
                    'long_window': long,
                    'sharpe': sharpe
                })
        
        return pd.DataFrame(results)


    def fetch_data(self):
        self.data = yf.download(self.symbol, self.start_date, self.end_date)
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.droplevel(1)
        
        self.in_sample = self.data[self.data.index < self.split_date]
        self.out_sample = self.data[self.data.index >= self.split_date]


    def generate_signals(self):
        self.data['SMA_Short'] = self.data['Close'].rolling(window=self.short_window).mean()
        self.data['SMA_Long'] = self.data['Close'].rolling(window=self.long_window).mean()
        
        self.data['Signal'] = 0
        self.data['Signal'] = (self.data['SMA_Short'] > self.data['SMA_Long']).astype(int)
        self.data['Position'] = self.data['Signal'].diff()
        
        self.data['Strategy_Returns'] = self.data['Close'].pct_change() * self.data['Signal'].shift(1)
        self.data['Buy_Hold_Returns'] = self.data['Close'].pct_change()


    def calculate_metrics(self):
        print(f"\n--- Performance Metrics for {self.symbol} ---")
        strategy_returns = self.data['Strategy_Returns'].dropna()

        # Win rate - percentage of days with positive returns while in a trade
        winning_days = strategy_returns[strategy_returns > 0]
        losing_days = strategy_returns[strategy_returns < 0]
        total_active_days = len(winning_days) + len(losing_days)

        win_rate = len(winning_days) / total_active_days * 100
        print(f"Win Rate: {win_rate:.2f}%")

        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")

        risk_free_rate = 0.02 / 252  # Daily risk free rate (assuming 2% annual)
        excess_returns = strategy_returns - risk_free_rate
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * (252 ** 0.5)
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")


    def monte_carlo(self, num_simulations=1000, days=252):
        daily_returns = self.data['Close'].pct_change().dropna()
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        last_price = self.data['Close'].iloc[-1]
        
        simulation_results = []
        price_paths = np.zeros((days, num_simulations))
        
        for i in range(num_simulations):
            # Generate simulated price path
            prices = [last_price]
            for _ in range(days):
                random_return = np.random.normal(mean_return, std_return)
                prices.append(prices[-1] * (1 + random_return))
            
            price_series = pd.Series(prices[1:])
            price_paths[:, i] = price_series.values
            
            # Run strategy on simulated path
            sma_short = price_series.rolling(window=self.short_window).mean()
            sma_long = price_series.rolling(window=self.long_window).mean()
            
            signal = (sma_short > sma_long).astype(int)
            strategy_returns = price_series.pct_change() * signal.shift(1)
            
            cumulative_return = (1 + strategy_returns.fillna(0)).cumprod().iloc[-1]
            simulation_results.append(cumulative_return)
        
        return price_paths, np.array(simulation_results)

    def plot_results(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        
        split_point = pd.to_datetime(self.split_date)

        in_returns = self.data.loc[self.data.index < split_point, 'Strategy_Returns'].fillna(0)
        out_returns = self.data.loc[self.data.index >= split_point, 'Strategy_Returns'].fillna(0)
        buyhold_returns = self.data['Buy_Hold_Returns'].fillna(0)
        
        cumulative_in = (1 + in_returns).cumprod()

        out_start = cumulative_in.iloc[-1]
        cumulative_out = (1 + out_returns).cumprod() * out_start
        cumulative_buyhold = (1 + buyhold_returns).cumprod()
        
        strategy_returns = self.data['Strategy_Returns'].dropna()
        
        winning_days = strategy_returns[strategy_returns > 0]
        losing_days = strategy_returns[strategy_returns < 0]
        total_active_days = len(winning_days) + len(losing_days)
        win_rate = len(winning_days) / total_active_days * 100
        
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        risk_free_rate = 0.02 / 252
        excess_returns = strategy_returns - risk_free_rate
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * (252 ** 0.5)

        in_strategy = self.data.loc[self.data.index < split_point, 'Strategy_Returns'].dropna()
        out_strategy = self.data.loc[self.data.index >= split_point, 'Strategy_Returns'].dropna()
        
        in_excess = in_strategy - risk_free_rate
        out_excess = out_strategy - risk_free_rate
        in_sharpe = (in_excess.mean() / in_excess.std()) * (252 ** 0.5)
        out_sharpe = (out_excess.mean() / out_excess.std()) * (252 ** 0.5)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.plot(cumulative_in.index, cumulative_in,
                color='#9467bd', linewidth=2, label='Strategy (In-sample)')
        ax.plot(cumulative_out.index, cumulative_out,
                color='#9467bd', linewidth=2, linestyle='--', label='Strategy (Out-of-sample)')
        ax.plot(cumulative_buyhold.index, cumulative_buyhold,
                color='#8c564b', linewidth=2, alpha=0.85, label='Buy and Hold')
        
        ax.fill_between(cumulative_in.index, cumulative_in, 1, alpha=0.1, color='#9467bd')
        
        ax.axvline(split_point, color='black', linestyle=':',
                linewidth=1.5, label='In/Out Sample Split')
        
        ax.plot([], [], ' ', label=f'── Metrics ──')
        ax.plot([], [], ' ', label=f'Win Rate: {win_rate:.1f}%')
        ax.plot([], [], ' ', label=f'Max Drawdown: {max_drawdown:.1f}%')
        ax.plot([], [], ' ', label=f'Sharpe (Full): {sharpe_ratio:.2f}')
        ax.plot([], [], ' ', label=f'Sharpe (In-sample): {in_sharpe:.2f}')
        ax.plot([], [], ' ', label=f'Sharpe (Out-of-sample): {out_sharpe:.2f}')
        
        ax.annotate('In-Sample', xy=(self.data.index[10], ax.get_ylim()[1] * 0.95),
                    fontsize=10, color='gray')
        ax.annotate('Out-of-Sample', xy=(split_point, ax.get_ylim()[1] * 0.95),
                    fontsize=10, color='gray')
        
        ax.set_title(
            f'{self.symbol}: Strategy vs Buy and Hold (SMA{self.short_window}/SMA{self.long_window})',
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Growth (x)')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.legend(frameon=True, loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        

    def plot_monte_carlo(self, simulations, simulation_results):
        last_price = self.data['Close'].iloc[-1]
        
        prob_profit = (simulation_results > 1.0).mean() * 100
        median_return = np.median(simulation_results)
        p95_return = np.percentile(simulation_results, 95)
        p5_return = np.percentile(simulation_results, 5)
        
        fig, (ax_paths, ax_hist) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left plot — price paths
        ax_paths.plot(simulations, alpha=0.05, color='blue', linewidth=0.5)
        ax_paths.plot(np.percentile(simulations, 95, axis=1),
                    color='green', linewidth=2, label='95th Percentile')
        ax_paths.plot(np.percentile(simulations, 50, axis=1),
                    color='orange', linewidth=2, label='Median')
        ax_paths.plot(np.percentile(simulations, 5, axis=1),
                    color='red', linewidth=2, label='5th Percentile')
        ax_paths.axhline(y=last_price, color='black',
                        linestyle='--', linewidth=1.5, label=f'Starting Price (${last_price:.2f})')
        ax_paths.set_title(f'{self.symbol} Simulated Price Paths\n({simulations.shape[1]} simulations, {simulations.shape[0]} days)',
                        fontsize=13, fontweight='bold')
        ax_paths.set_xlabel('Days')
        ax_paths.set_ylabel('Price ($)')
        ax_paths.legend(frameon=True)
        ax_paths.grid(True, alpha=0.3)
        
        # Right plot — strategy return distribution
        ax_hist.hist(simulation_results, bins=50, color='#9467bd',
                    alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_hist.axvline(x=1.0, color='black', linestyle='--',
                        linewidth=1.5, label='Break Even')
        ax_hist.axvline(x=median_return, color='orange',
                        linewidth=2, label=f'Median ({median_return:.2f}x)')
        ax_hist.axvline(x=p95_return, color='green',
                        linewidth=2, label=f'95th Percentile ({p95_return:.2f}x)')
        ax_hist.axvline(x=p5_return, color='red',
                        linewidth=2, label=f'5th Percentile ({p5_return:.2f}x)')
        ax_hist.plot([], [], ' ', label=f'Probability of Profit: {prob_profit:.1f}%')
        
        ax_hist.set_title(f'Strategy Return Distribution\n(SMA{self.short_window}/SMA{self.long_window})',
                        fontsize=13, fontweight='bold')
        ax_hist.set_xlabel('Cumulative Return (x)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.legend(frameon=True, loc='upper right')
        ax_hist.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def run(self):
        self.fetch_data()
        
        short_windows = range(5, 50, 5)
        long_windows = range(10, 200, 10)
        
        in_sample_results = self.optimize_parameters(short_windows, long_windows, self.in_sample)
        # print(in_sample_results.sort_values('sharpe', ascending=False))
        best = in_sample_results.sort_values('sharpe', ascending=False).iloc[0]
        self.short_window = int(best.short_window)
        self.long_window = int(best.long_window)
        print(f"\nBest in-sample parameters: SMA{int(best.short_window)}/SMA{int(best.long_window)} with Sharpe {best.sharpe:.2f}")
        
        out_sample_results = self.optimize_parameters(
            [self.short_window], 
            [self.long_window], 
            self.out_sample
        )
        print(f"Out-of-sample Sharpe with same parameters: {out_sample_results.iloc[0].sharpe:.2f}")
        
        self.generate_signals()
        self.plot_results()
        price_paths, simulation_results = self.monte_carlo()
        self.plot_monte_carlo(price_paths, simulation_results)
        self.calculate_metrics()
        


def main():
    backtester = StockBacktester(
        symbol='AAPL',
        start_date='2020-01-01',
        end_date='2024-01-01',
        split_date='2022-01-01'
    )
    backtester.run()


if __name__ == '__main__':
    main()
