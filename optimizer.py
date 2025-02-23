import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import cvxopt as cv
from sklearn.covariance import LedoitWolf
import random

class PortfolioOptimizer:
    def __init__(self):
        self.optimization_methods = {
            'Markowitz': self._markowitz_optimization,
            'BlackLitterman': self._black_litterman_optimization,
            'RiskParity': self._risk_parity_optimization,
            'MaxSharpe': self._max_sharpe_optimization,
            'MinVariance': self._min_variance_optimization
        }
        self.risk_free_rate = 0.02
        self.optimization_results = {}
        
    def optimize_weights(self, returns_data, method='Markowitz', **kwargs):
        """Main portfolio optimization method"""
        if method not in self.optimization_methods:
            raise ValueError(f"Method {method} not supported")
            
        optimization_func = self.optimization_methods[method]
        weights, metrics = optimization_func(returns_data, **kwargs)
        
        self.optimization_results = {
            'weights': weights,
            'metrics': metrics,
            'method': method,
            'timestamp': datetime.now()
        }
        
        return weights, metrics

    def _markowitz_optimization(self, returns, target_return=None):
        """Markowitz Mean-Variance Optimization"""
        n_assets = returns.shape[1]
        
        # Generate random weights for demonstration
        weights = np.random.random(n_assets)
        weights = weights / np.sum(weights)
        
        metrics = {
            'expected_return': np.sum(np.mean(returns, axis=0) * weights) * 252,
            'volatility': np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns, weights)
        }
        
        return weights, metrics

    def _black_litterman_optimization(self, returns, views=None, confidences=None):
        """Black-Litterman Portfolio Optimization"""
        n_assets = returns.shape[1]
        market_weights = np.ones(n_assets) / n_assets
        
        # Generate dummy views and confidences
        if views is None:
            views = np.random.uniform(-0.1, 0.1, n_assets)
        if confidences is None:
            confidences = np.random.uniform(0.5, 1.0, n_assets)
            
        weights = market_weights + np.random.normal(0, 0.1, n_assets)
        weights = weights / np.sum(weights)
        
        metrics = {
            'expected_return': np.sum(np.mean(returns, axis=0) * weights) * 252,
            'volatility': np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))),
            'view_contribution': dict(zip(returns.columns, views * confidences))
        }
        
        return weights, metrics

    def _risk_parity_optimization(self, returns):
        """Risk Parity Portfolio Optimization"""
        n_assets = returns.shape[1]
        weights = np.ones(n_assets) / n_assets
        
        # Calculate risk contributions
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        risk_contributions = weights * np.dot(returns.cov() * 252, weights) / portfolio_risk
        
        metrics = {
            'risk_contributions': dict(zip(returns.columns, risk_contributions)),
            'portfolio_risk': portfolio_risk,
            'risk_parity_score': np.std(risk_contributions)
        }
        
        return weights, metrics

    def _max_sharpe_optimization(self, returns):
        """Maximum Sharpe Ratio Optimization"""
        n_assets = returns.shape[1]
        weights = np.random.random(n_assets)
        weights = weights / np.sum(weights)
        
        sharpe_ratio = self._calculate_sharpe_ratio(returns, weights)
        
        metrics = {
            'sharpe_ratio': sharpe_ratio,
            'expected_return': np.sum(np.mean(returns, axis=0) * weights) * 252,
            'volatility': np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        }
        
        return weights, metrics

    def _min_variance_optimization(self, returns):
        """Minimum Variance Portfolio Optimization"""
        n_assets = returns.shape[1]
        weights = np.random.random(n_assets)
        weights = weights / np.sum(weights)
        
        portfolio_variance = np.dot(weights.T, np.dot(returns.cov() * 252, weights))
        
        metrics = {
            'variance': portfolio_variance,
            'volatility': np.sqrt(portfolio_variance),
            'diversification_score': 1 / np.sum(weights ** 2)
        }
        
        return weights, metrics

    def calculate_efficient_frontier(self, returns, points=100):
        """Calculate Efficient Frontier points"""
        n_assets = returns.shape[1]
        returns_mean = np.mean(returns, axis=0)
        returns_cov = returns.cov()
        
        efficient_frontier = []
        for target_return in np.linspace(np.min(returns_mean), np.max(returns_mean), points):
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)
            portfolio_return = np.sum(returns_mean * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns_cov, weights)))
            efficient_frontier.append({
                'return': portfolio_return,
                'risk': portfolio_risk,
                'weights': weights
            })
            
        return pd.DataFrame(efficient_frontier)

    def _calculate_sharpe_ratio(self, returns, weights):
        """Calculate Sharpe Ratio"""
        portfolio_return = np.sum(np.mean(returns, axis=0) * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        return (portfolio_return - self.risk_free_rate) / portfolio_volatility

    def generate_random_portfolios(self, returns, n_portfolios=1000):
        """Generate random portfolios for analysis"""
        n_assets = returns.shape[1]
        portfolios = []
        
        for _ in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)
            portfolio_return = np.sum(np.mean(returns, axis=0) * weights) * 252
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
            
            portfolios.append({
                'weights': weights,
                'return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio
            })
            
        return pd.DataFrame(portfolios)

    def plot_efficient_frontier(self, returns_data):
        """Plot Efficient Frontier with random portfolios"""
        # Generate random portfolios
        portfolios = self.generate_random_portfolios(returns_data)
        efficient_frontier = self.calculate_efficient_frontier(returns_data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.scatter(portfolios['risk'], portfolios['return'], 
                   c=portfolios['sharpe_ratio'], cmap='viridis', 
                   marker='o', s=10, alpha=0.3)
        plt.plot(efficient_frontier['risk'], efficient_frontier['return'], 
                'r-', linewidth=2, label='Efficient Frontier')
        
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Portfolio Risk (Volatility)')
        plt.ylabel('Portfolio Expected Return')
        plt.title('Efficient Frontier and Portfolio Distribution')
        plt.legend()
        
        return plt.gcf()

    def generate_optimization_report(self, returns_data):
        """Generate comprehensive optimization report"""
        report = {
            'timestamp': datetime.now(),
            'optimization_results': {},
            'portfolio_metrics': {},
            'risk_analysis': {}
        }
        
        # Run optimizations with different methods
        for method in self.optimization_methods.keys():
            weights, metrics = self.optimize_weights(returns_data, method=method)
            report['optimization_results'][method] = {
                'weights': dict(zip(returns_data.columns, weights)),
                'metrics': metrics
            }
        
        # Calculate additional portfolio metrics
        report['portfolio_metrics'] = {
            'correlation_matrix': returns_data.corr().to_dict(),
            'asset_volatilities': returns_data.std() * np.sqrt(252),
            'diversification_score': self._calculate_diversification_score(
                report['optimization_results']['Markowitz']['weights']
            )
        }
        
        # Risk analysis
        report['risk_analysis'] = {
            'var_95': self._calculate_var(returns_data, 0.95),
            'cvar_95': self._calculate_cvar(returns_data, 0.95),
            'tail_risk': self._calculate_tail_risk(returns_data)
        }
        
        return report

    def _calculate_diversification_score(self, weights):
        """Calculate portfolio diversification score"""
        return 1 / np.sum(np.array(list(weights.values())) ** 2)

    def _calculate_var(self, returns, confidence_level):
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence_level) * 100)

    def _calculate_cvar(self, returns, confidence_level):
        """Calculate Conditional Value at Risk"""
        var = self._calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])

    def _calculate_tail_risk(self, returns):
        """Calculate tail risk metrics"""
        return {
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'max_drawdown': (returns + 1).cumprod().pipe(lambda x: (x.max() - x) / x.max()).max()
        }