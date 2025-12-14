#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discord Bot for Cryptocurrency Price Predictions

Commands:
  /predict <symbol> - Get prediction for a symbol
  /predict_all - Get predictions for all available symbols
  /models - List all available models
  /info <symbol> - Get model information for a symbol
  /price <symbol> - Get current price
"""

import discord
from discord.ext import commands, tasks
import logging
from typing import Optional, Dict, List
from dotenv import load_dotenv
import os
import asyncio
from predictor import CryptoPredictor
from datetime import datetime, timedelta

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)
predictor = None


@bot.event
async def on_ready():
    """
    Called when bot is ready
    """
    logger.info(f"✓ Bot connected as {bot.user}")
    await bot.change_presence(
        activity=discord.Activity(type=discord.ActivityType.watching, name="crypto prices 💰")
    )
    
    # Start background tasks
    if not auto_predict.is_running():
        auto_predict.start()


@bot.event
async def on_command_error(ctx, error):
    """
    Error handler
    """
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("⚠️  Command not found. Use `/` for slash commands.")
    else:
        logger.error(f"Command error: {error}")
        await ctx.send(f"⚠️  Error: {str(error)[:100]}")


# ============ SLASH COMMANDS ============

@bot.tree.command(name="predict", description="Get price prediction for a cryptocurrency")
async def predict(interaction: discord.Interaction, symbol: str, timeframe: str = "1h"):
    """
    Get prediction for a symbol
    """
    await interaction.response.defer()
    
    try:
        symbol = symbol.upper()
        result = predictor.predict(symbol, timeframe)
        
        if not result:
            await interaction.followup.send(f"⚠️  Failed to get prediction for {symbol}")
            return
        
        # Build embed
        embed = discord.Embed(
            title=f"🔍 {symbol} Price Prediction",
            color=discord.Color.blue(),
            timestamp=datetime.now()
        )
        
        # Price info
        embed.add_field(
            name="💰 Price Info",
            value=f"Current: ${result['current_price']:,.2f}\n"
                  f"Predicted: ${result['predicted_price']:,.2f}\n"
                  f"Change: {result['price_change_percent']:+.2f}%",
            inline=False
        )
        
        # Predictions
        embed.add_field(
            name="🔏 3-5 Candle Forecast",
            value=f"High: ${result['predicted_high_3_5']:,.2f}\n"
                  f"Low: ${result['predicted_low_3_5']:,.2f}",
            inline=False
        )
        
        # Technical Levels
        embed.add_field(
            name="🎯 Support/Resistance",
            value=f"Support: ${result['support']:,.2f}\n"
                  f"Resistance: ${result['resistance']:,.2f}",
            inline=False
        )
        
        # Trading Signal
        embed.add_field(
            name="📄 Trading Signal",
            value=f"Signal: **{result['signal_type']}**\n"
                  f"Recommendation: {result['recommendation']}",
            inline=False
        )
        
        # Risk Management
        embed.add_field(
            name="⚠️  Risk Management",
            value=f"Entry: ${result['entry_point']:,.2f}\n"
                  f"Stop Loss: ${result['stop_loss']:,.2f}\n"
                  f"Take Profit: ${result['take_profit']:,.2f}\n"
                  f"Risk/Reward: {result['risk_reward_ratio']:.2f}x",
            inline=False
        )
        
        # Indicators
        embed.add_field(
            name="📈 Indicators",
            value=f"RSI(14): {result['rsi']:.2f}\n"
                  f"MACD: {result['macd']:.6f}\n"
                  f"Confidence: {result['confidence']*100:.1f}%",
            inline=False
        )
        
        embed.set_footer(text=f"Timeframe: {timeframe} | Confidence: {result['confidence']*100:.0f}%")
        
        await interaction.followup.send(embed=embed)
    
    except Exception as e:
        logger.error(f"Predict command error: {e}")
        await interaction.followup.send(f"✗ Error: {str(e)[:100]}")


@bot.tree.command(name="predict_all", description="Get predictions for all available cryptocurrencies")
async def predict_all(interaction: discord.Interaction):
    """
    Get predictions for all symbols
    """
    await interaction.response.defer()
    
    try:
        results = predictor.predict_all()
        
        if not results:
            await interaction.followup.send("⚠️  No predictions available")
            return
        
        # Build summary embed
        embed = discord.Embed(
            title="👀 All Cryptocurrency Predictions",
            color=discord.Color.green(),
            timestamp=datetime.now()
        )
        
        # Group by signal type
        buy_signals = []
        sell_signals = []
        hold_signals = []
        
        for symbol, result in sorted(results.items()):
            signal = result['signal_type']
            price_str = f"${result['current_price']:,.2f} → ${result['predicted_price']:,.2f}"
            
            if 'BUY' in signal:
                buy_signals.append(f"{symbol}: {price_str} ({result['price_change_percent']:+.2f}%)")
            elif 'SELL' in signal:
                sell_signals.append(f"{symbol}: {price_str} ({result['price_change_percent']:+.2f}%)")
            else:
                hold_signals.append(f"{symbol}: {price_str} ({result['price_change_percent']:+.2f}%)")
        
        # Add fields
        if buy_signals:
            embed.add_field(
                name="🔜 BUY Signals (" + str(len(buy_signals)) + ")",
                value="\n".join(buy_signals[:10]),
                inline=False
            )
        
        if sell_signals:
            embed.add_field(
                name="📉 SELL Signals (" + str(len(sell_signals)) + ")",
                value="\n".join(sell_signals[:10]),
                inline=False
            )
        
        if hold_signals:
            embed.add_field(
                name="⏸️  HOLD Signals (" + str(len(hold_signals)) + ")",
                value="\n".join(hold_signals[:10]),
                inline=False
            )
        
        embed.set_footer(text=f"Total: {len(results)} symbols | Generated at {datetime.now().strftime('%H:%M:%S UTC')}")
        
        await interaction.followup.send(embed=embed)
    
    except Exception as e:
        logger.error(f"Predict all command error: {e}")
        await interaction.followup.send(f"✗ Error: {str(e)[:100]}")


@bot.tree.command(name="models", description="List all available models")
async def models(interaction: discord.Interaction):
    """
    List all available models
    """
    try:
        symbols = sorted(list(predictor.model_manager.models.keys()))
        
        if not symbols:
            await interaction.response.send_message("⚠️  No models loaded")
            return
        
        # Paginate if too many
        embed = discord.Embed(
            title="🤖 Available Models",
            color=discord.Color.purple(),
            description=f"Total: {len(symbols)} models"
        )
        
        # Add symbols in chunks
        chunk_size = 20
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            embed.add_field(
                name=f"Models ({i+1}-{min(i+chunk_size, len(symbols))})",
                value=", ".join(chunk),
                inline=False
            )
        
        await interaction.response.send_message(embed=embed)
    
    except Exception as e:
        logger.error(f"Models command error: {e}")
        await interaction.response.send_message(f"✗ Error: {str(e)[:100]}")


@bot.tree.command(name="info", description="Get model information for a symbol")
async def info(interaction: discord.Interaction, symbol: str):
    """
    Get model info for a symbol
    """
    try:
        symbol = symbol.upper()
        model_info = predictor.get_model_info().get(symbol)
        
        if not model_info:
            await interaction.response.send_message(f"⚠️  No model found for {symbol}")
            return
        
        embed = discord.Embed(
            title=f"📄 Model Information - {symbol}",
            color=discord.Color.blue()
        )
        
        for key, value in model_info.items():
            embed.add_field(
                name=key.replace('_', ' ').title(),
                value=str(value),
                inline=True
            )
        
        await interaction.response.send_message(embed=embed)
    
    except Exception as e:
        logger.error(f"Info command error: {e}")
        await interaction.response.send_message(f"✗ Error: {str(e)[:100]}")


@bot.tree.command(name="price", description="Get current price of a cryptocurrency")
async def price(interaction: discord.Interaction, symbol: str):
    """
    Get current price
    """
    await interaction.response.defer()
    
    try:
        symbol = symbol.upper()
        trading_pair = f"{symbol}/USDT"
        
        current_price = predictor.data_fetcher.get_latest_price(trading_pair)
        
        if current_price is None:
            await interaction.followup.send(f"⚠️  Failed to fetch price for {symbol}")
            return
        
        embed = discord.Embed(
            title=f"💰 {symbol} Price",
            color=discord.Color.green(),
            timestamp=datetime.now()
        )
        
        embed.add_field(
            name="Current Price",
            value=f"${current_price:,.2f}",
            inline=False
        )
        
        await interaction.followup.send(embed=embed)
    
    except Exception as e:
        logger.error(f"Price command error: {e}")
        await interaction.followup.send(f"✗ Error: {str(e)[:100]}")


@bot.tree.command(name="help_crypto", description="Show all available commands")
async def help_crypto(interaction: discord.Interaction):
    """
    Show help
    """
    embed = discord.Embed(
        title="👁 Crypto Prediction Bot Commands",
        color=discord.Color.gold(),
        description="Use `/` to access these commands:"
    )
    
    embed.add_field(
        name="/predict <symbol> [timeframe]",
        value="Get detailed price prediction for a cryptocurrency",
        inline=False
    )
    
    embed.add_field(
        name="/predict_all",
        value="Get predictions for all available cryptocurrencies",
        inline=False
    )
    
    embed.add_field(
        name="/models",
        value="List all available cryptocurrency models",
        inline=False
    )
    
    embed.add_field(
        name="/info <symbol>",
        value="Get detailed model information for a symbol",
        inline=False
    )
    
    embed.add_field(
        name="/price <symbol>",
        value="Get current price of a cryptocurrency",
        inline=False
    )
    
    embed.add_field(
        name="/help_crypto",
        value="Show this help message",
        inline=False
    )
    
    embed.set_footer(text="Use `/` to type commands")
    
    await interaction.response.send_message(embed=embed)


# ============ BACKGROUND TASKS ============

@tasks.loop(hours=1)
async def auto_predict():
    """
    Automatically generate predictions every hour
    """
    try:
        logger.info("🔄 Running auto-predictions...")
        results = predictor.predict_all()
        logger.info(f"✓ Auto-predictions completed ({len(results)} symbols)")
    except Exception as e:
        logger.error(f"Auto-predict error: {e}")


# ============ INITIALIZATION ============

async def initialize_bot():
    """
    Initialize bot and predictor
    """
    global predictor
    
    logger.info("\n" + "="*60)
    logger.info("🚀 Initializing Crypto Prediction Discord Bot")
    logger.info("="*60)
    
    # Initialize predictor with correct folder name
    predictor = CryptoPredictor(
        hf_repo="zongowo111/crypto_model",
        hf_folder="models"  # Correct folder name
    )
    
    # Load models
    predictor.initialize()
    
    logger.info(f"🚀 Bot ready to use | Loaded {len(predictor.model_manager.models)} models")
    logger.info("="*60 + "\n")


async def main():
    """
    Main function
    """
    await initialize_bot()
    
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        logger.error("⚠️  DISCORD_BOT_TOKEN not found in .env")
        return
    
    await bot.start(token)


if __name__ == '__main__':
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛸 Bot shutdown")
