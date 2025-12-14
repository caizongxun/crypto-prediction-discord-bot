# ⚡ Quick Start Guide

## 5-Minute Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/caizongxun/crypto-prediction-discord-bot.git
cd crypto-prediction-discord-bot
```

### Step 2: Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Bot

```bash
# Copy example config
cp .env.example .env

# Edit .env and add your Discord token
# (Get token from https://discord.com/developers/applications)
nano .env
```

### Step 4: Run Bot

```bash
# Open 2 terminals

# Terminal 1: Discord Bot
python discord_bot.py

# Terminal 2: Web Dashboard
python web_dashboard.py
```

### Step 5: Test in Discord

```
/predict BTC
/predict_all
/models
/help_crypto
```

Dashboard: http://localhost:5000

---

## What Happens on First Run

1. 📑 **Bot loads** → Reads `.env` file
2. 👁 **Detects models** → Lists available cryptocurrencies from HuggingFace
3. 📦 **Downloads models** → Auto-caches them locally
4. 🎯 **Loads models** → Auto-detects architecture
5. ✍️ **Ready** → Accepts Discord commands

---

## Available Commands

| Command | Description | Example |
|---------|-------------|----------|
| `/predict <symbol>` | Get prediction | `/predict BTC` |
| `/predict_all` | All predictions | `/predict_all` |
| `/models` | List models | `/models` |
| `/info <symbol>` | Model info | `/info ETH` |
| `/price <symbol>` | Current price | `/price SOL` |
| `/help_crypto` | Help menu | `/help_crypto` |

---

## Supported Cryptocurrencies

Bot auto-detects which symbols are available from your HuggingFace models.

Typically includes:
- **Major**: BTC, ETH, BNB, SOL
- **Altcoins**: ADA, DOGE, XRP, LINK, POLKA, etc.
- **Layer2**: ARB, OP, MATIC
- **Memes**: SHIB, PEPE, DOGE

Run `/models` to see exact list.

---

## Example Output

### Discord Prediction

```
🔍 BTC Price Prediction
💰 Price Info
Current: $43,250.00
Predicted: $44,120.50
Change: +1.98%

🔏 3-5 Candle Forecast
High: $44,500.00
Low: $42,800.00

🎯 Support/Resistance
Support: $42,100.00
Resistance: $44,900.00

📄 Trading Signal
Signal: STRONG_BUY
Recommendation: 🔟 STRONG BUY: Oversold signal at $42,100.00

⚠️ Risk Management
Entry: $42,100.00
Stop Loss: $41,258.00
Take Profit: $45,306.00
Risk/Reward: 3.45x

📈 Indicators
RSI(14): 28.50
MACD: 0.000542
Confidence: 85%
```

### Web Dashboard

- View all predictions in real-time
- Filter by signal (BUY/SELL/HOLD)
- Search by symbol
- See entry/exit points
- Export as JSON

---

## Common Questions

### Q: What's my Discord bot token?

A: Go to https://discord.com/developers/applications
1. Click "New Application"
2. Go to "Bot" → "Add Bot"
3. Copy the token
4. Paste in `.env`

### Q: How do I invite the bot to my server?

A: In Developer Portal:
1. Go to "OAuth2" → "URL Generator"
2. Select scopes: `bot`
3. Select permissions: `Send Messages`, `Embed Links`
4. Copy generated URL
5. Open in browser to invite

### Q: Why are some symbols missing?

A: Models must exist on HuggingFace. Check if `symbol_model_v8.pth` is uploaded.

### Q: Can I add more models?

A: Yes! Upload to HuggingFace in format:
```
zongowo111/crypto_model/model/SYMBOL_model_v8.pth
```
Bot auto-detects on next load.

### Q: Can I change the update interval?

A: Yes! In `.env`:
```env
UPDATE_INTERVAL=3600  # seconds (1 hour)
```

### Q: How do I stop the bot?

A: Press `Ctrl+C` in terminal

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'discord'"

```bash
# Activate venv and install
source venv/bin/activate
pip install discord.py==2.3.2
```

### "DISCORD_BOT_TOKEN not found"

```bash
# Make sure .env exists
ls -la .env

# Check content
cat .env
```

### "Failed to fetch predictions"

```bash
# Check internet connection
ping google.com

# Check Binance availability
# Try different timeframe
/predict BTC 4h
```

### "No model found for BTC"

```bash
# Check available models
/models

# Check HuggingFace has models
# https://huggingface.co/zongowo111/crypto_model
```

---

## Next Steps

- 📚 Read full [README.md](README.md)
- 🚀 Deploy to server ([DEPLOYMENT.md](DEPLOYMENT.md))
- 🛠️ Customize indicators ([data_fetcher.py](data_fetcher.py))
- 🤟 Contribute improvements

---

## Need Help?

1. Check [README.md](README.md) troubleshooting section
2. Review logs: `python discord_bot.py > bot.log 2>&1`
3. Open GitHub issue with error details

---

**Happy trading! 🚀**
