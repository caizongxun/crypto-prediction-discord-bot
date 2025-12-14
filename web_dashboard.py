#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Dashboard for Cryptocurrency Price Predictions

Visualize predictions and trading signals in real-time
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import logging
import asyncio
from predictor import CryptoPredictor
import json
from datetime import datetime
import threading

app = Flask(__name__)
CORS(app)

logger = logging.getLogger(__name__)

# Global predictor
predictor = None
update_interval = 3600  # 1 hour


@app.route('/')
def index():
    """
    Main dashboard page
    """
    return render_template('dashboard.html')


@app.route('/api/predictions')
def get_predictions():
    """
    Get all latest predictions
    """
    try:
        predictions = predictor.get_latest_predictions()
        return jsonify(predictions)
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/<symbol>')
def predict_symbol(symbol):
    """
    Get prediction for a specific symbol
    """
    try:
        result = predictor.predict(symbol.upper())
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': f'Failed to predict {symbol}'}), 400
    except Exception as e:
        logger.error(f"Error predicting {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/models')
def get_models():
    """
    Get all model information
    """
    try:
        models = predictor.get_model_info()
        return jsonify(models)
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/<symbol>')
def get_model_info(symbol):
    """
    Get model info for specific symbol
    """
    try:
        info = predictor.model_manager.get_model_info(symbol.upper())
        if info:
            return jsonify(info)
        else:
            return jsonify({'error': f'No model found for {symbol}'}), 404
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def get_status():
    """
    Get bot status
    """
    try:
        return jsonify({
            'status': 'online',
            'models_loaded': len(predictor.model_manager.models),
            'predictions_cached': len(predictor.predictions_cache),
            'last_update': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/update-all')
def update_all():
    """
    Manually trigger prediction update for all symbols
    """
    try:
        results = predictor.predict_all()
        return jsonify({
            'status': 'success',
            'updated': len(results),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error updating predictions: {e}")
        return jsonify({'error': str(e)}), 500


def background_update():
    """
    Background task to update predictions
    """
    while True:
        try:
            logger.info("🔄 Background: Updating all predictions...")
            predictor.predict_all()
            logger.info("✓ Background: Update complete")
        except Exception as e:
            logger.error(f"Background update error: {e}")
        
        # Wait for next update
        for _ in range(update_interval):
            if not app.config.get('running', True):
                return
            threading.Event().wait(1)


def initialize():
    """
    Initialize predictor on startup
    """
    global predictor
    
    logger.info("\n" + "="*60)
    logger.info("🚀 Initializing Web Dashboard")
    logger.info("="*60)
    
    predictor = CryptoPredictor(
        hf_repo="zongowo111/crypto_model",
        hf_folder="models"  # Correct folder name
    )
    
    predictor.initialize()
    logger.info(f"✓ Web dashboard ready | {len(predictor.model_manager.models)} models loaded")
    logger.info("="*60 + "\n")
    
    app.config['running'] = True


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    initialize()
    
    # Start background update thread
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
