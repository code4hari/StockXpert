import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class MarketPredictor:
    def __init__(self):
        self.models = {
            'LSTM': self._create_dummy_lstm(),
            'RandomForest': RandomForestRegressor(n_estimators=100),
            'XGBoost': xgb.XGBRegressor(),
            'GradientBoosting': GradientBoostingRegressor()
        }
        self.scaler = MinMaxScaler()
        self.predictions = {}
        self.performance_metrics = {}
        
    def _create_dummy_lstm(self):
        """Create dummy LSTM model"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def generate_price_forecast(self, ticker, history_df, forecast_days=30):
        """Generate price forecasts using multiple models"""
        base_price = history_df['Close'].iloc[-1]
        dates = pd.date_range(start=datetime.now(), periods=forecast_days, freq='B')
        
        forecasts = {}
        for model_name in self.models.keys():
            forecast_prices = self._generate_model_forecast(base_price, forecast_days)
            forecasts[model_name] = {
                'prices': forecast_prices,
                'confidence_intervals': self._calculate_confidence_intervals(forecast_prices),
                'volatility': np.std(forecast_prices),
                'trend': self._determine_trend(forecast_prices)
            }
            
        ensemble_forecast = self._generate_ensemble_forecast(forecasts)
        
        return {
            'individual_forecasts': forecasts,
            'ensemble_forecast': ensemble_forecast,
            'forecast_dates': dates,
            'metadata': {
                'ticker': ticker,
                'generation_time': datetime.now(),
                'forecast_horizon': forecast_days
            }
        }

    def _generate_model_forecast(self, base_price, forecast_days):
        """Generate dummy forecast for a single model"""
        trend = np.random.uniform(-0.001, 0.001)
        volatility = np.random.uniform(0.01, 0.02)
        
        prices = [base_price]
        for _ in range(forecast_days - 1):
            change = np.random.normal(trend, volatility)
            next_price = prices[-1] * (1 + change)
            prices.append(next_price)
            
        return np.array(prices)

    def _generate_ensemble_forecast(self, individual_forecasts):
        """Generate ensemble forecast from individual model predictions"""
        all_forecasts = np.array([f['prices'] for f in individual_forecasts.values()])
        
        ensemble_forecast = {
            'mean_forecast': np.mean(all_forecasts, axis=0),
            'median_forecast': np.median(all_forecasts, axis=0),
            'std_forecast': np.std(all_forecasts, axis=0),
            'model_weights': self._calculate_model_weights(individual_forecasts)
        }
        
        return ensemble_forecast

    def _calculate_confidence_intervals(self, forecast, confidence_levels=[0.95, 0.99]):
        """Calculate confidence intervals for forecasts"""
        intervals = {}
        std_dev = np.std(forecast)
        
        for level in confidence_levels:
            z_score = norm.ppf((1 + level) / 2)
            margin = z_score * std_dev
            intervals[f'{level*100}%'] = {
                'upper': forecast + margin,
                'lower': forecast - margin
            }
            
        return intervals

    def _determine_trend(self, prices):
        """Determine trend direction and strength"""
        returns = np.diff(prices) / prices[:-1]
        trend_strength = np.mean(returns)
        
        if trend_strength > 0.001:
            direction = 'Bullish'
        elif trend_strength < -0.001:
            direction = 'Bearish'
        else:
            direction = 'Neutral'
            
        return {
            'direction': direction,
            'strength': abs(trend_strength),
            'consistency': np.std(returns)
        }

    def _calculate_model_weights(self, forecasts):
        """Calculate weights for ensemble model"""
        weights = {
            'LSTM': random.uniform(0.2, 0.3),
            'RandomForest': random.uniform(0.2, 0.3),
            'XGBoost': random.uniform(0.2, 0.3),
            'GradientBoosting': random.uniform(0.2, 0.3)
        }
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def evaluate_predictions(self, actual, predicted):
        """Evaluate prediction accuracy"""
        metrics = {
            'MSE': mean_squared_error(actual, predicted),
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'MAE': mean_absolute_error(actual, predicted),
            'R2': r2_score(actual, predicted)
        }
        return metrics

    def generate_technical_signals(self, data):
        """Generate technical trading signals"""
        signals = {
            'MA_Crossover': random.choice(['Buy', 'Sell', 'Hold']),
            'RSI': random.uniform(0, 100),
            'MACD': {
                'Signal': random.choice(['Buy', 'Sell', 'Hold']),
                'Histogram': random.uniform(-1, 1)
            },
            'Bollinger_Bands': {
                'Position': random.uniform(0, 1),
                'Signal': random.choice(['Buy', 'Sell', 'Hold'])
            }
        }
        return signals

    def plot_forecasts(self, forecast_data):
        """Plot forecasts with confidence intervals"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Individual model forecasts
        for model, forecast in forecast_data['individual_forecasts'].items():
            ax1.plot(forecast_data['forecast_dates'], forecast['prices'], label=model)
        ax1.set_title('Model Forecasts Comparison')
        ax1.legend()
        
        # Ensemble forecast with confidence intervals
        ensemble = forecast_data['ensemble_forecast']
        ax2.plot(forecast_data['forecast_dates'], ensemble['mean_forecast'], 'b-', label='Ensemble Mean')
        
        # Plot confidence intervals
        ci = forecast_data['individual_forecasts']['LSTM']['confidence_intervals']
        ax2.fill_between(forecast_data['forecast_dates'],
                        ci['95%']['lower'],
                        ci['95%']['upper'],
                        alpha=0.2,
                        label='95% CI')
        
        ax2.set_title('Ensemble Forecast with Confidence Intervals')
        ax2.legend()
        
        plt.tight_layout()
        return fig

    def generate_forecast_report(self, ticker, forecast_data):
        """Generate comprehensive forecast report"""
        report = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'forecast_summary': {
                'short_term_outlook': self._determine_trend(forecast_data['ensemble_forecast']['mean_forecast']),
                'confidence_level': random.uniform(0.7, 0.9),
                'risk_assessment': self._assess_forecast_risk(forecast_data)
            },
            'technical_indicators': self.generate_technical_signals(forecast_data),
            'model_performance': self.performance_metrics,
            'forecast_data': forecast_data
        }
        return report

    def _assess_forecast_risk(self, forecast_data):
        """Assess risk levels in forecasts"""
        volatility = np.std(forecast_data['ensemble_forecast']['std_forecast'])
        return {
            'volatility_risk': 'High' if volatility > 0.02 else 'Medium' if volatility > 0.01 else 'Low',
            'uncertainty_score': random.uniform(0, 1),
            'risk_factors': random.sample(['Market Volatility', 'Economic Indicators', 'Technical Levels'], 2)
        }