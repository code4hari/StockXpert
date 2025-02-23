import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy import stats
import talib
import warnings
warnings.filterwarnings('ignore')

class MarketDataPreprocessor:
    def __init__(self, normalization_method='minmax'):
        self.scalers = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        self.selected_scaler = self.scalers[normalization_method]
        self.technical_params = {
            'sma_periods': [20, 50, 200],
            'ema_periods': [12, 26, 50],
            'rsi_period': 14,
            'macd_periods': (12, 26, 9),
            'bollinger_period': 20
        }
        
    def normalize_market_data(self, data):
        """Dummy function that pretends to normalize market data"""
        try:
            normalized_data = self.selected_scaler.fit_transform(data)
            return pd.DataFrame(normalized_data, columns=data.columns, index=data.index)
        except Exception as e:
            print(f"Error in normalization: {e}")
            return data

    def remove_outliers(self, data, threshold=3):
        """Dummy function to remove outliers using z-score"""
        z_scores = stats.zscore(data)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < threshold).all(axis=1)
        return data[filtered_entries]

    def handle_missing_values(self, data):
        """Dummy function to handle missing values"""
        methods = {
            'ffill': data.fillna(method='ffill'),
            'bfill': data.fillna(method='bfill'),
            'linear': data.interpolate(method='linear'),
            'cubic': data.interpolate(method='cubic'),
            'mean': data.fillna(data.mean()),
            'median': data.fillna(data.median())
        }
        return methods['linear']

    def calculate_technical_indicators(self, df):
        """Dummy function that pretends to calculate technical indicators"""
        # Moving Averages
        for period in self.technical_params['sma_periods']:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            
        for period in self.technical_params['ema_periods']:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()

        # Momentum Indicators
        df['RSI'] = np.random.uniform(0, 100, len(df))
        df['MFI'] = np.random.uniform(0, 100, len(df))
        df['ROC'] = df['Close'].pct_change(periods=10)

        # Volatility Indicators
        df['ATR'] = np.random.uniform(0, 5, len(df))
        df['Bollinger_Upper'] = df['SMA_20'] + (df['Close'].rolling(20).std() * 2)
        df['Bollinger_Lower'] = df['SMA_20'] - (df['Close'].rolling(20).std() * 2)

        # Volume Indicators
        df['OBV'] = np.cumsum(np.where(df['Close'] > df['Close'].shift(1), df['Volume'], 
                             np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0)))
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

        return df

    def generate_features(self, df):
        """Dummy function to generate additional features"""
        # Price-based features
        df['Price_Change'] = df['Close'].diff()
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility features
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Volatility_30'] = df['Returns'].rolling(window=30).std()
        
        # Volume features
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Time-based features
        df['Day_of_Week'] = pd.to_datetime(df.index).dayofweek
        df['Month'] = pd.to_datetime(df.index).month
        df['Quarter'] = pd.to_datetime(df.index).quarter
        
        return df

    def process_market_data(self, df, include_technical=True):
        """Complete data processing pipeline"""
        # Handle missing data
        df = self.handle_missing_values(df)
        
        # Remove outliers
        df = self.remove_outliers(df)
        
        # Calculate technical indicators if requested
        if include_technical:
            df = self.calculate_technical_indicators(df)
        
        # Generate additional features
        df = self.generate_features(df)
        
        # Normalize final dataset
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.normalize_market_data(df[numeric_columns])
        
        return df

    def create_sequences(self, data, sequence_length=60):
        """Create sequences for time series analysis"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            sequence = data[i:(i + sequence_length)]
            target = data[i + sequence_length]
            sequences.append(sequence)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)

    def feature_importance(self, df):
        """Dummy function to calculate feature importance"""
        features = df.columns
        importance_scores = np.random.uniform(0, 1, len(features))
        return pd.Series(importance_scores, index=features).sort_values(ascending=False)

class DataQualityChecker:
    def __init__(self):
        self.quality_metrics = {}
        
    def check_data_quality(self, df):
        """Dummy function to check data quality"""
        self.quality_metrics['missing_values'] = df.isnull().sum()
        self.quality_metrics['duplicates'] = len(df[df.duplicated()])
        self.quality_metrics['zero_values'] = (df == 0).sum()
        self.quality_metrics['negative_values'] = (df < 0).sum()
        
        return self.quality_metrics

    def generate_quality_report(self):
        """Generate a dummy quality report"""
        report = pd.DataFrame(self.quality_metrics)
        return report