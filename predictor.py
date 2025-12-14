#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Price Predictor - Generate trading signals based on LSTM predictions
"""

import torch
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from model_manager import ModelManager
from data_fetcher import DataFetcher, TechnicalAnalyzer

logger = logging.getLogger(__name__)


class CryptoPredictor:
    """
    Main predictor class that combines:
    - Model loading
    - Data fetching
    - Feature engineering
    - Signal generation
    """
    
    def __init__(self, hf_repo: str = "zongowo111/crypto_model", hf_folder: str = "model"):
        self.model_manager = ModelManager(hf_repo=hf_repo, hf_folder=hf_folder)
        self.data_fetcher = DataFetcher()
        self.analyzer = TechnicalAnalyzer()
        self.predictions_cache = {}  # symbol -> latest prediction
        
        logger.info("🤖 CryptoPredictor initialized")
    
    def initialize(self) -> List[str]:
        """
        Initialize predictor by loading all models
        
        Returns:
            List of available symbols
        """
        logger.info("\n" + "="*60)
        logger.info("📦 Loading all models from Hugging Face...")
        logger.info("="*60)
        
        symbols = self.model_manager.list_available_models()
        
        if symbols:
            self.model_manager.load_all_models(symbols)
            logger.info(f"\n✓ Successfully loaded {len(self.model_manager.models)} models")
        else:
            logger.warning("\n⚠️  No models found")
        
        return symbols
    
    def get_model_info(self) -> Dict:
        """
        Get detailed information about all loaded models
        
        Returns:
            Dictionary with model specifications
        """
        info_dict = {}
        
        for symbol, model_info in self.model_manager.get_all_model_info().items():
            info_dict[symbol] = {
                'input_size': model_info['input_size'],
                'hidden_size': model_info['hidden_size'],
                'num_layers': model_info['num_layers'],
                'output_size': model_info['output_size']
            }
        
        return info_dict
    
    def predict(self, symbol: str, timeframe: str = '1h') -> Optional[Dict]:
        """
        Generate trading prediction and signals for a symbol
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            timeframe: Candle timeframe (default: '1h')
        
        Returns:
            Dictionary with prediction results and trading signals
        """
        try:
            symbol = symbol.upper()
            trading_pair = f"{symbol}/USDT"
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Analyzing {symbol}...")
            logger.info(f"{'='*60}")
            
            # 1. Fetch data
            df = self.data_fetcher.fetch_ohlcv(trading_pair, timeframe, limit=100)
            if df is None or df.empty:
                logger.error(f"Failed to fetch data for {symbol}")
                return None
            
            current_price = float(df['close'].iloc[-1])
            logger.info(f"💰 Current Price: ${current_price:,.2f}")
            
            # 2. Build features
            logger.info(f"\n📈 Building feature vector...")
            features = self.analyzer.build_feature_vector(df)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # 3. Make prediction
            logger.info(f"🤖 Running LSTM model...")
            predicted_price = self.model_manager.predict(symbol, features_tensor)
            
            if predicted_price is None:
                logger.error(f"Prediction failed for {symbol}")
                return None
            
            predicted_price = float(predicted_price)
            price_change = ((predicted_price - current_price) / current_price) * 100
            
            logger.info(f"  Predicted Price: ${predicted_price:,.2f}")
            logger.info(f"  Expected Change: {price_change:+.2f}%")
            
            # 4. Calculate technical indicators
            logger.info(f"\n📈 Technical Indicators:")
            support, resistance = self.analyzer.calculate_support_resistance(df)
            rsi = self.analyzer.calculate_rsi(df)
            macd, signal, histogram = self.analyzer.calculate_macd(df)
            
            logger.info(f"  Support: ${support:,.2f}")
            logger.info(f"  Resistance: ${resistance:,.2f}")
            logger.info(f"  RSI(14): {rsi:.2f}")
            logger.info(f"  MACD: {macd:.6f}")
            
            # 5. Predict next 3-5 candles and find entry points
            logger.info(f"\n🔏 Analyzing next 3-5 candles...")
            
            # Simulate future price movements
            predicted_prices = [current_price]
            for i in range(5):
                next_price = predicted_price + (price_change / 100 * current_price) * (i+1) / 5
                predicted_prices.append(float(next_price))
            
            # Find high and low points
            predicted_high = max(predicted_prices[1:4])  # 3 candles
            predicted_low = min(predicted_prices[1:4])
            
            logger.info(f"  Predicted High (3-5 candles): ${predicted_high:,.2f}")
            logger.info(f"  Predicted Low (3-5 candles): ${predicted_low:,.2f}")
            
            # 6. Generate trading signals
            logger.info(f"\n📄 Trading Signals:")
            
            # Determine signal type
            if price_change > 0:
                # Uptrend - look for buy signal
                if rsi < 30:
                    signal_type = "STRONG_BUY"
                    entry_point = support
                    recommendation = f"🔜 STRONG BUY: Oversold signal at ${support:,.2f}"
                elif current_price < support * 1.05:
                    signal_type = "BUY"
                    entry_point = support * 1.02
                    recommendation = f"📈 BUY: Near support at ${entry_point:,.2f}"
                else:
                    signal_type = "HOLD"
                    entry_point = current_price
                    recommendation = f"⏸️  HOLD: In uptrend"
            else:
                # Downtrend - look for sell signal
                if rsi > 70:
                    signal_type = "STRONG_SELL"
                    entry_point = resistance
                    recommendation = f"📉 STRONG SELL: Overbought signal at ${resistance:,.2f}"
                elif current_price > resistance * 0.95:
                    signal_type = "SELL"
                    entry_point = resistance * 0.98
                    recommendation = f"📉 SELL: Near resistance at ${entry_point:,.2f}"
                else:
                    signal_type = "HOLD"
                    entry_point = current_price
                    recommendation = f"⏸️  HOLD: In downtrend"
            
            # Risk management
            stop_loss = predicted_low * 0.98
            take_profit = predicted_high * 1.02
            risk_reward_ratio = (take_profit - entry_point) / (entry_point - stop_loss)
            
            logger.info(f"  Entry: ${entry_point:,.2f}")
            logger.info(f"  Stop Loss: ${stop_loss:,.2f}")
            logger.info(f"  Take Profit: ${take_profit:,.2f}")
            logger.info(f"  Risk/Reward: {risk_reward_ratio:.2f}x")
            logger.info(f"  {recommendation}")
            
            # 7. Calculate confidence
            confidence = self._calculate_confidence(rsi, macd, histogram, price_change)
            logger.info(f"  Confidence: {confidence*100:.1f}%")
            
            # Build result
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_percent': price_change,
                'predicted_high_3_5': predicted_high,
                'predicted_low_3_5': predicted_low,
                'support': support,
                'resistance': resistance,
                'signal_type': signal_type,
                'recommendation': recommendation,
                'entry_point': entry_point,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio,
                'rsi': rsi,
                'macd': macd,
                'confidence': confidence,
            }
            
            # Cache result
            self.predictions_cache[symbol] = result
            
            logger.info(f"{'='*60}\n")
            return result
        
        except Exception as e:
            logger.error(f"✗ Prediction failed for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def predict_all(self) -> Dict[str, Dict]:
        """
        Generate predictions for all available models
        
        Returns:
            Dictionary of all predictions
        """
        results = {}
        
        symbols = list(self.model_manager.models.keys())
        logger.info(f"\n👀 Generating predictions for {len(symbols)} symbols...\n")
        
        for symbol in sorted(symbols):
            result = self.predict(symbol)
            if result:
                results[symbol] = result
        
        logger.info(f"\n✓ Completed {len(results)} predictions")
        return results
    
    def get_latest_predictions(self) -> Dict[str, Dict]:
        """
        Get cached predictions
        """
        return self.predictions_cache
    
    @staticmethod
    def _calculate_confidence(rsi: float, macd: float, histogram: float, price_change: float) -> float:
        """
        Calculate confidence score (0-1)
        """
        confidence_factors = []
        
        # RSI confidence
        if rsi < 30 or rsi > 70:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # MACD confidence
        if (macd > 0 and price_change > 0) or (macd < 0 and price_change < 0):
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Histogram confidence
        if histogram > 0 and price_change > 0:
            confidence_factors.append(0.8)
        elif histogram < 0 and price_change < 0:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        return float(np.mean(confidence_factors))


if __name__ == '__main__':
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test
    predictor = CryptoPredictor()
    predictor.initialize()
    
    # Get model info
    print("\n📄 Model Information:")
    for symbol, info in predictor.get_model_info().items():
        print(f"  {symbol}: {info}")
    
    # Make predictions
    results = predictor.predict_all()
