# 🚀 Crypto Price Prediction Discord Bot

Advanced Discord bot for real-time cryptocurrency price predictions using LSTM neural networks. Features automatic model downloading from Hugging Face, real-time Binance data fetching, and an interactive web dashboard.

## ✨ Features

### 🤖 Model Management
- **Auto-download** LSTM models from Hugging Face (`zongowo111/crypto_model`)
- **Auto-detect** model architecture (input_size, hidden_size)
- **CPU-only** PyTorch (cost-effective)
- **Adaptive** model loading
- **Model Info Display** for debugging

### 📊 Real-Time Data Fetching
- **Multi-exchange**: Binance → Bybit → OKX → Kraken
- **1H timeframe** OHLCV data
- **Caching** to minimize API calls
- **44-dimensional** feature vectors

### 🏷️ Prediction & Trading Signals
- **3-5 candle forecasts**
- **Entry/Exit points**
- **Risk management** (SL, TP)
- **Risk/Reward ratios**
- **Confidence scoring**

### 💬 Discord Integration
- **Slash commands** (no conflicts)
- **Real-time predictions**
- **Batch predictions**
- **Auto-update** every hour
- **Embed formatting**

### 🌐 Web Dashboard
- **Real-time visualization**
- **Interactive filters**
- **Search functionality**
- **Export data** as JSON
- **Mobile-responsive**
- **Dark theme**

## 📋 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/caizongxun/crypto-prediction-discord-bot.git
cd crypto-prediction-discord-bot
```

### 2. Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Discord token
```

### 3. Run

```bash
# Discord Bot
python discord_bot.py

# Web Dashboard
python web_dashboard.py
# Visit http://localhost:5000
```

## 🔧 Discord Commands

- `/predict <symbol>` - Prediction for symbol (e.g., `/predict BTC`)
- `/predict_all` - All predictions
- `/models` - Available models
- `/info <symbol>` - Model info
- `/price <symbol>` - Current price
- `/help_crypto` - Help

## 📊 Model Architecture

- **Input**: 44-dimensional feature vector
  - OHLCV (5)
  - Price changes (10)
  - Moving averages (12)
  - Momentum indicators (12)
  - Volatility measures (5)

- **Model**: Bidirectional LSTM (2 layers) + Dense regressor
- **Output**: Price prediction (next 3-5 candles)
- **Framework**: PyTorch (CPU-only mode)

## 🌐 Web Dashboard

### Features
- View all predictions in real-time
- Filter by signal type (BUY/SELL/HOLD)
- Search by cryptocurrency symbol
- See support/resistance levels
- View entry/exit points and risk metrics
- Auto-refresh every 5 minutes
- Export predictions as JSON

### Access
```bash
python web_dashboard.py
# Open http://localhost:5000
```

## 📈 Trading Signals

### Signal Types
- **🟢 STRONG_BUY**: Oversold (RSI < 30) + Uptrend
- **🟢 BUY**: Near support + Positive momentum
- **🟡 HOLD**: Sideways / No strong signals
- **🔴 SELL**: Near resistance + Negative momentum
- **🔴 STRONG_SELL**: Overbought (RSI > 70) + Downtrend

### Risk Management
- Entry point from support/resistance
- Stop loss: 2% below predicted low
- Take profit: 2% above predicted high
- Risk/Reward ratio calculation

## 🛠️ Configuration

### .env File
```env
# Discord
DISCORD_BOT_TOKEN=your_token_here

# Hugging Face
HF_REPO=zongowo111/crypto_model
HF_FOLDER=model

# Web
WEB_PORT=5000
WEB_HOST=0.0.0.0

# Update Interval (seconds)
UPDATE_INTERVAL=3600
```

## 📁 Project Structure

```
crypto-prediction-discord-bot/
├── model_manager.py      # Model downloading & loading
├── data_fetcher.py       # Real-time data & indicators
├── predictor.py          # Prediction engine
├── discord_bot.py        # Discord bot commands
├── web_dashboard.py      # Flask web server
├── templates/
│   └── dashboard.html    # Web UI
├── requirements.txt      # Dependencies
├── .env.example          # Config template
└── README.md             # This file
```

## 🛠️ Troubleshooting

### Models Not Loading
- Check internet connection (HuggingFace)
- Verify HF_REPO and HF_FOLDER
- Check models directory permissions
- Run with DEBUG=true

### Data Fetching Issues
- Check internet connection
- Verify Binance/exchange status
- Try different timeframe
- Wait for rate limit reset

### Bot Not Responding
- Verify DISCORD_BOT_TOKEN
- Check bot permissions on server
- Verify message content intent enabled
- Check Discord Developer Portal

## 👩‍💻 Development

### Adding New Symbols
Place models in HuggingFace repo with pattern:
```
model/SYMBOL_model_v8.pth
```
Bot auto-detects and loads!

### Custom Indicators
Modify `TechnicalAnalyzer.build_feature_vector()` in `data_fetcher.py`

### Deployment
```bash
# Linux/Unix (systemd)
sudo systemctl start crypto-bot

# Docker (optional)
docker build -t crypto-bot .
docker run -d --env-file .env crypto-bot
```

## 🐐 Security

- 🔒 Discord token in `.env` (not in git)
- 🔒 Models cached locally
- 🔒 CPU-only (no GPU requirements)
- 🔒 Read-only data fetching
- 🔒 No orders placed (prediction only)

## 📝 License

MIT License - See LICENSE file

## 🎯 Status

✅ Model auto-download from HuggingFace
✅ Auto-detect model architecture
✅ Real-time Binance data fetching
✅ LSTM price predictions
✅ Trading signal generation
✅ Discord bot commands
✅ Web dashboard
✅ Multi-exchange fallback
✅ Technical indicators (RSI, MACD, ATR, etc.)
✅ Risk management (SL, TP, R:R ratio)
✅ Batch predictions
✅ Auto-update every hour
✅ Model debugging info

## 🤟 Contributing

Contributions welcome!

1. Fork repository
2. Create feature branch (`git checkout -b feature/Amazing`)
3. Commit changes (`git commit -m 'Add Amazing'`)
4. Push to branch (`git push origin feature/Amazing`)
5. Open Pull Request

## 📧 Support

For issues:
1. Check existing GitHub issues
2. Review troubleshooting section
3. Create new issue with error logs

## 🌠 Example Output

### Discord Prediction
```
🔍 BTC Price Prediction
💰 Current: $43,250.00
🤖 Predicted: $44,120.50
📈 Change: +1.98%

🔏 3-5 Candles
High: $44,500.00 / Low: $42,800.00

🎯 Entry: $42,100.00
⚠️ SL: $41,258.00 / TP: $45,306.00
R/R: 3.45x

📈 RSI: 28.5 | Confidence: 85%
🔟 STRONG BUY Signal
```

---

**Made with ❤️ for crypto traders**

Star ⭐ if this helps!
