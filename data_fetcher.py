#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time cryptocurrency data fetcher using CCXT
"""

import ccxt
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetch real-time OHLCV data from multiple exchanges
    
    Features:
    - Multi-exchange support with fallback
    - Caching to avoid excessive API calls
    - Technical indicator calculation
    - Support for multiple timeframes
    """
    
    def __init__(self, cache_duration: int = 60):
        self.exchanges = self._init_exchanges()
        self.cache = {}  # symbol_timeframe -> DataFrame
        self.cache_time = {}  # symbol_timeframe -> timestamp
        self.cache_duration = cache_duration
    
    def _init_exchanges(self) -> Dict:
        """
        Initialize multiple exchanges with fallback
        """
        exchanges = {}
        
        # Primary: Binance
        try:
            exchanges['binance'] = ccxt.binance({
                'enableRateLimit': True,
                'rateLimit': 1000,
            })
            logger.info("✓ Binance initialized")
        except Exception as e:
            logger.warning(f"⚠️  Binance failed: {e}")
        
        # Fallback: Bybit
        try:
            exchanges['bybit'] = ccxt.bybit({
                'enableRateLimit': True,
                'rateLimit': 1000,
            })
            logger.info("✓ Bybit initialized")
        except Exception as e:
            logger.warning(f"⚠️  Bybit failed: {e}")
        
        # Fallback: OKX
        try:
            exchanges['okx'] = ccxt.okx({
                'enableRateLimit': True,
                'rateLimit': 1000,
            })
            logger.info("✓ OKX initialized")
        except Exception as e:
            logger.warning(f"⚠️  OKX failed: {e}")
        
        # Fallback: Kraken
        try:
            exchanges['kraken'] = ccxt.kraken({
                'enableRateLimit': True,
                'rateLimit': 3000,
            })
            logger.info("✓ Kraken initialized")
        except Exception as e:
            logger.warning(f"⚠️  Kraken failed: {e}")
        
        return exchanges
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from exchanges
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (1m, 5m, 1h, 4h, 1d)
            limit: Number of candles to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.cache:
            time_diff = datetime.now() - self.cache_time[cache_key]
            if time_diff.total_seconds() < self.cache_duration:
                return self.cache[cache_key]
        
        # Try exchanges in order
        exchange_order = ['binance', 'bybit', 'okx', 'kraken']
        
        for exchange_name in exchange_order:
            if exchange_name not in self.exchanges:
                continue
            
            try:
                exchange = self.exchanges[exchange_name]
                logger.debug(f"Fetching {symbol} from {exchange_name}...")
                
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df.astype(float)
                
                # Cache
                self.cache[cache_key] = df
                self.cache_time[cache_key] = datetime.now()
                
                logger.debug(f"✓ Fetched {len(df)} candles for {symbol}")
                return df
            
            except Exception as e:
                logger.debug(f"⚠️  {exchange_name} failed for {symbol}: {str(e)[:80]}")
                continue
        
        logger.error(f"✗ Failed to fetch {symbol} from all exchanges")
        return None
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for a symbol
        """
        for exchange_name in ['binance', 'bybit', 'okx', 'kraken']:
            if exchange_name not in self.exchanges:
                continue
            
            try:
                exchange = self.exchanges[exchange_name]
                ticker = exchange.fetch_ticker(symbol)
                return float(ticker['last'])
            except:
                continue
        
        return None


class TechnicalAnalyzer:
    """
    Calculate technical indicators and trading signals
    """
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """
        Calculate support and resistance levels
        """
        try:
            recent = df.tail(window)
            support = recent['low'].min()
            resistance = recent['high'].max()
            return float(support), float(resistance)
        except:
            return 0, float('inf')
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate RSI indicator
        """
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate MACD indicator
        """
        try:
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            return float(macd.iloc[-1]), float(signal.iloc[-1]), float(histogram.iloc[-1])
        except:
            return 0, 0, 0
    
    @staticmethod
    def build_feature_vector(df: pd.DataFrame) -> np.ndarray:
        """
        Build 44-dimensional feature vector for model input
        """
        try:
            features = []
            
            # OHLCV (5)
            current = df.iloc[-1]
            features.extend([
                float(current['open']),
                float(current['high']),
                float(current['low']),
                float(current['close']),
                float(current['volume'])
            ])
            
            # Price changes (10)
            close_prices = df['close'].values
            for period in [1, 5, 10, 20, 50]:
                if len(df) >= period:
                    pct_change = (close_prices[-1] - close_prices[-period]) / close_prices[-period]
                    features.append(float(pct_change))
            
            # Moving averages (12)
            try:
                features.append(float(df['close'].rolling(5).mean().iloc[-1]))
                features.append(float(df['close'].rolling(10).mean().iloc[-1]))
                features.append(float(df['close'].rolling(20).mean().iloc[-1]))
                features.append(float(df['close'].rolling(50).mean().iloc[-1]))
                features.append(float(df['close'].ewm(span=5).mean().iloc[-1]))
                features.append(float(df['close'].ewm(span=12).mean().iloc[-1]))
                features.append(float(df['close'].ewm(span=26).mean().iloc[-1]))
                features.append(float(df['close'].rolling(20).std().iloc[-1]))
                features.append(float(df['close'].rolling(50).std().iloc[-1]))
                
                features.append(float((df['high'].rolling(20).max() - df['low'].rolling(20).min()).iloc[-1]))
                features.append(float(df['high'].iloc[-1] - df['low'].iloc[-1]))
                features.append(float((df['close'] - df['open']).abs().mean()))
            except:
                features.extend([0.0] * 12)
            
            # Momentum indicators (12)
            try:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                features.append(float(rsi.iloc[-1]))
                
                exp1 = df['close'].ewm(span=12).mean()
                exp2 = df['close'].ewm(span=26).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9).mean()
                histogram = macd - signal
                features.append(float(macd.iloc[-1]))
                features.append(float(signal.iloc[-1]))
                features.append(float(histogram.iloc[-1]))
                
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(14).mean()
                features.append(float(atr.iloc[-1]))
                
                sma = df['close'].rolling(20).mean()
                std = df['close'].rolling(20).std()
                bb_upper = sma + (std * 2)
                bb_lower = sma - (std * 2)
                bb_position = (df['close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                features.append(float(bb_position))
                
                lowest_low = df['low'].rolling(14).min()
                highest_high = df['high'].rolling(14).max()
                k_percent = 100 * ((df['close'].iloc[-1] - lowest_low.iloc[-1]) / (highest_high.iloc[-1] - lowest_low.iloc[-1]))
                features.append(float(k_percent))
                
                volume_sma = df['volume'].rolling(20).mean()
                volume_ratio = df['volume'].iloc[-1] / (volume_sma.iloc[-1] + 1e-8)
                features.append(float(volume_ratio))
                
                obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
                features.append(float(obv.iloc[-1]))
            except:
                features.extend([0.0] * 12)
            
            # Volatility (5)
            try:
                returns = df['close'].pct_change()
                features.append(float(returns.std()))
                features.append(float(returns.mean()))
                features.append(float((df['high'] - df['low']).mean()))
                features.append(float((df['high'] - df['low']).std()))
                
                min_price = df['close'].rolling(50).min().iloc[-1]
                max_price = df['close'].rolling(50).max().iloc[-1]
                price_position = (df['close'].iloc[-1] - min_price) / (max_price - min_price)
                features.append(float(price_position))
            except:
                features.extend([0.0] * 5)
            
            # Ensure exactly 44 features
            features = features[:44]
            while len(features) < 44:
                features.append(0.0)
            
            return np.array(features, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Error building feature vector: {e}")
            return np.zeros(44, dtype=np.float32)
