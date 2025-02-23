import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class PortfolioRiskAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.02
        self.confidence_level = 0.95
        self.simulation_runs = 10000
        self.historical_window = 252  # Trading days in a year
        self.risk_metrics = {}
        
    def calculate_basic_metrics(self, returns):
        """Calculate basic risk metrics"""
        metrics = {
            'mean_return': np.mean(returns),
            'volatility': np.std(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'var_95': np.percentile(returns, 5),
            'var_99': np.percentile(returns, 1),
            'max_drawdown': self._calculate_max_drawdown(returns)
        }
        return metrics

    def calculate_sharpe_ratio(self, returns, periods_per_year=252):
        """Enhanced Sharpe ratio calculation"""
        excess_returns = returns - self.risk_free_rate/periods_per_year
        sharpe_ratio = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(returns)
        
        # Calculate rolling Sharpe ratio
        rolling_sharpe = np.sqrt(periods_per_year) * (
            returns.rolling(window=periods_per_year).mean() - self.risk_free_rate
        ) / returns.rolling(window=periods_per_year).std()
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'rolling_sharpe': rolling_sharpe
        }

    def calculate_sortino_ratio(self, returns, periods_per_year=252):
        """Calculate Sortino ratio"""
        excess_returns = returns - self.risk_free_rate/periods_per_year
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns)
        
        sortino_ratio = np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std
        return sortino_ratio

    def calculate_var(self, portfolio_value, returns, method='historical'):
        """Enhanced Value at Risk calculation with multiple methods"""
        methods = {
            'historical': self._historical_var(returns),
            'parametric': self._parametric_var(returns),
            'monte_carlo': self._monte_carlo_var(returns)
        }
        
        var_results = {}
        for method_name, var_calc in methods.items():
            var_results[method_name] = portfolio_value * var_calc
            
        return var_results

    def _historical_var(self, returns):
        """Historical VaR calculation"""
        return np.percentile(returns, (1 - self.confidence_level) * 100)

    def _parametric_var(self, returns):
        """Parametric VaR calculation"""
        mean = np.mean(returns)
        std = np.std(returns)
        return stats.norm.ppf(1 - self.confidence_level, mean, std)

    def _monte_carlo_var(self, returns):
        """Monte Carlo VaR calculation"""
        mean = np.mean(returns)
        std = np.std(returns)
        simulated_returns = np.random.normal(mean, std, self.simulation_runs)
        return np.percentile(simulated_returns, (1 - self.confidence_level) * 100)

    def calculate_expected_shortfall(self, returns):
        """Calculate Expected Shortfall (CVaR)"""
        var = self._historical_var(returns)
        return np.mean(returns[returns <= var])

    def monte_carlo_simulation(self, initial_value, mu, sigma, periods=252):
        """Enhanced Monte Carlo simulation with multiple scenarios"""
        scenarios = []
        final_values = []
        
        for _ in range(self.simulation_runs):
            # Generate random returns
            returns = np.random.normal(mu/252, sigma/np.sqrt(252), periods)
            
            # Calculate price path
            price_path = initial_value * np.exp(np.cumsum(returns))
            scenarios.append(price_path)
            final_values.append(price_path[-1])
            
        results = {
            'scenarios': np.array(scenarios),
            'final_values': np.array(final_values),
            'confidence_intervals': {
                '95%': np.percentile(final_values, [2.5, 97.5]),
                '99%': np.percentile(final_values, [0.5, 99.5])
            },
            'summary_statistics': {
                'mean': np.mean(final_values),
                'std': np.std(final_values),
                'min': np.min(final_values),
                'max': np.max(final_values)
            }
        }
        
        return results

    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns/running_max - 1
        return np.min(drawdowns)

    def stress_test_portfolio(self, portfolio, scenarios):
        """Perform stress testing on portfolio"""
        stress_results = {}
        
        # Historical scenarios
        historical_events = {
            '2008_crisis': -0.40,
            'covid_crash': -0.30,
            'tech_bubble': -0.35,
            'rate_hike': -0.15
        }
        
        for event, shock in historical_events.items():
            stress_results[event] = portfolio * (1 + shock)
            
        return stress_results

    def calculate_beta(self, returns, market_returns):
        """Calculate portfolio beta"""
        covariance = np.cov(returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance

    def generate_risk_report(self, portfolio_data):
        """Generate comprehensive risk report"""
        returns = portfolio_data['returns']
        value = portfolio_data['value']
        
        report = {
            'basic_metrics': self.calculate_basic_metrics(returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'var_analysis': self.calculate_var(value, returns),
            'expected_shortfall': self.calculate_expected_shortfall(returns),
            'stress_test': self.stress_test_portfolio(value, None)
        }
        
        return pd.DataFrame(report)

    def plot_risk_metrics(self, returns):
        """Generate visualization of risk metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Returns distribution
        sns.histplot(returns, kde=True, ax=axes[0,0])
        axes[0,0].set_title('Returns Distribution')
        
        # QQ plot
        stats.probplot(returns, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('Q-Q Plot')
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)
        rolling_vol.plot(ax=axes[1,0])
        axes[1,0].set_title('Rolling Volatility')
        
        # Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns/running_max - 1
        drawdowns.plot(ax=axes[1,1])
        axes[1,1].set_title('Drawdown')
        
        plt.tight_layout()
        return fig

class CorrelationAnalyzer:
    def __init__(self):
        self.correlation_matrix = None
        
    def calculate_correlations(self, returns_df):
        """Calculate correlation matrix"""
        self.correlation_matrix = returns_df.corr()
        return self.correlation_matrix
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Asset Correlation Heatmap')
        return plt.gcf()