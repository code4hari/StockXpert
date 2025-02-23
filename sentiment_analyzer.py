import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from textblob import TextBlob
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class MarketSentimentAnalyzer:
    def __init__(self):
        self.sentiment_sources = {
            'twitter': 0.3,  # weight for each source
            'news': 0.4,
            'reddit': 0.2,
            'stocktwits': 0.1
        }
        self.sentiment_history = defaultdict(list)
        self.sentiment_keywords = {
            'bullish': ['buy', 'bullish', 'upward', 'growth', 'positive'],
            'bearish': ['sell', 'bearish', 'downward', 'decline', 'negative'],
            'neutral': ['hold', 'stable', 'steady', 'maintain']
        }
        
    def analyze_social_media_sentiment(self, ticker, timeframe='1d'):
        """Enhanced social media sentiment analysis"""
        platforms = {
            'twitter': self._analyze_twitter_sentiment(ticker),
            'reddit': self._analyze_reddit_sentiment(ticker),
            'stocktwits': self._analyze_stocktwits_sentiment(ticker)
        }
        
        sentiment_scores = {}
        for platform, data in platforms.items():
            sentiment_scores[platform] = {
                'bullish_score': random.uniform(0, 1),
                'bearish_score': random.uniform(0, 1),
                'neutral_score': random.uniform(0, 1),
                'volume': random.randint(100, 10000),
                'trending_topics': self._generate_trending_topics(),
                'key_influencers': self._generate_key_influencers()
            }
            
        return self._aggregate_sentiment_scores(sentiment_scores)

    def _analyze_twitter_sentiment(self, ticker):
        """Dummy Twitter sentiment analysis"""
        return {
            'sentiment_score': random.uniform(-1, 1),
            'tweet_volume': random.randint(1000, 50000),
            'verified_mentions': random.randint(10, 100),
            'hashtag_trends': self._generate_hashtag_trends()
        }

    def _analyze_reddit_sentiment(self, ticker):
        """Dummy Reddit sentiment analysis"""
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        return {
            'sentiment_score': random.uniform(-1, 1),
            'post_volume': random.randint(100, 5000),
            'comment_volume': random.randint(1000, 20000),
            'subreddit_distribution': {sub: random.randint(10, 100) for sub in subreddits}
        }

    def _analyze_stocktwits_sentiment(self, ticker):
        """Dummy StockTwits sentiment analysis"""
        return {
            'sentiment_score': random.uniform(-1, 1),
            'message_volume': random.randint(100, 5000),
            'watchers': random.randint(1000, 50000),
            'trending_rank': random.randint(1, 100)
        }

    def get_news_sentiment(self, ticker):
        """Enhanced news sentiment analysis"""
        news_sources = ['Bloomberg', 'Reuters', 'CNBC', 'WSJ', 'MarketWatch']
        
        articles = []
        for _ in range(random.randint(10, 30)):
            source = random.choice(news_sources)
            articles.append({
                'source': source,
                'title': f"Dummy article title about {ticker}",
                'timestamp': datetime.now() - timedelta(hours=random.randint(1, 72)),
                'sentiment_score': random.uniform(-1, 1),
                'relevance_score': random.uniform(0.5, 1),
                'impact_score': random.uniform(0, 1)
            })
        
        return {
            'articles': articles,
            'aggregate_sentiment': {
                'positive_articles': sum(1 for a in articles if a['sentiment_score'] > 0.2),
                'negative_articles': sum(1 for a in articles if a['sentiment_score'] < -0.2),
                'neutral_articles': sum(1 for a in articles if abs(a['sentiment_score']) <= 0.2),
                'sentiment_score': np.mean([a['sentiment_score'] for a in articles]),
                'source_distribution': {source: random.randint(1, 10) for source in news_sources}
            }
        }

    def analyze_earnings_calls(self, ticker):
        """Dummy earnings call sentiment analysis"""
        transcript_metrics = {
            'positive_statements': random.randint(10, 50),
            'negative_statements': random.randint(5, 30),
            'neutral_statements': random.randint(20, 60),
            'key_topics': self._generate_earnings_topics(),
            'executive_tone': random.uniform(-1, 1),
            'qa_session_sentiment': random.uniform(-1, 1)
        }
        return transcript_metrics

    def generate_sentiment_report(self, ticker):
        """Generate comprehensive sentiment report"""
        social_sentiment = self.analyze_social_media_sentiment(ticker)
        news_sentiment = self.get_news_sentiment(ticker)
        earnings_sentiment = self.analyze_earnings_calls(ticker)
        
        report = {
            'timestamp': datetime.now(),
            'ticker': ticker,
            'overall_sentiment': self._calculate_overall_sentiment(
                social_sentiment,
                news_sentiment,
                earnings_sentiment
            ),
            'social_media_metrics': social_sentiment,
            'news_metrics': news_sentiment,
            'earnings_call_metrics': earnings_sentiment,
            'sentiment_trends': self._generate_sentiment_trends(),
            'key_drivers': self._identify_sentiment_drivers()
        }
        
        return report

    def _generate_trending_topics(self):
        """Generate dummy trending topics"""
        topics = ['earnings', 'merger', 'growth', 'innovation', 'market share']
        return random.sample(topics, random.randint(2, 4))

    def _generate_key_influencers(self):
        """Generate dummy key influencers"""
        influencers = ['@trader_pro', '@market_guru', '@stock_expert']
        return random.sample(influencers, random.randint(1, 3))

    def _generate_hashtag_trends(self):
        """Generate dummy hashtag trends"""
        hashtags = ['#investing', '#trading', '#stocks', '#wallstreet']
        return {tag: random.randint(100, 1000) for tag in random.sample(hashtags, 3)}

    def _generate_earnings_topics(self):
        """Generate dummy earnings call topics"""
        topics = ['revenue growth', 'market expansion', 'cost reduction', 'innovation']
        return {topic: random.uniform(0, 1) for topic in random.sample(topics, 3)}

    def _calculate_overall_sentiment(self, social, news, earnings):
        """Calculate weighted overall sentiment"""
        weights = {'social': 0.3, 'news': 0.5, 'earnings': 0.2}
        
        social_score = social.get('sentiment_score', 0)
        news_score = news['aggregate_sentiment']['sentiment_score']
        earnings_score = earnings['executive_tone']
        
        overall_score = (
            weights['social'] * social_score +
            weights['news'] * news_score +
            weights['earnings'] * earnings_score
        )
        
        return {
            'score': overall_score,
            'classification': self._classify_sentiment(overall_score),
            'confidence': random.uniform(0.6, 0.9)
        }

    def _classify_sentiment(self, score):
        """Classify sentiment score"""
        if score > 0.2:
            return 'Bullish'
        elif score < -0.2:
            return 'Bearish'
        else:
            return 'Neutral'

    def _generate_sentiment_trends(self):
        """Generate dummy sentiment trends"""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        return {
            'sentiment_scores': [random.uniform(-1, 1) for _ in dates],
            'volume_trends': [random.randint(1000, 10000) for _ in dates],
            'dates': dates.tolist()
        }

    def _identify_sentiment_drivers(self):
        """Identify dummy sentiment drivers"""
        drivers = ['product launch', 'earnings report', 'market conditions']
        return {driver: random.uniform(0, 1) for driver in drivers}

    def plot_sentiment_trends(self, sentiment_data):
        """Plot sentiment trends"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Sentiment scores plot
        dates = sentiment_data['dates']
        scores = sentiment_data['sentiment_scores']
        ax1.plot(dates, scores)
        ax1.set_title('Sentiment Score Trend')
        ax1.set_ylabel('Sentiment Score')
        
        # Volume trend plot
        volumes = sentiment_data['volume_trends']
        ax2.bar(dates, volumes)
        ax2.set_title('Volume Trend')
        ax2.set_ylabel('Volume')
        
        plt.tight_layout()
        return fig