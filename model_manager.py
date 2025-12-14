#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Manager - Download and manage LSTM models from Hugging Face
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from huggingface_hub import hf_hub_download, list_repo_files
import re
import json

logger = logging.getLogger(__name__)


class CryptoLSTMModel(torch.nn.Module):
    """Adaptive LSTM model that handles variable hidden_size"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, output_size: int = 1):
        super(CryptoLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Bidirectional output size
        lstm_output_size = hidden_size * 2
        
        # Adaptive regressor
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(lstm_output_size, hidden_size * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.regressor(last_output)
        
        return output


class ModelManager:
    """
    Manage LSTM models from Hugging Face
    
    Features:
    - Auto-download models from HuggingFace
    - Detect model architecture from weights
    - Cache local models
    - Display model info for debugging
    """
    
    def __init__(self, hf_repo: str = "zongowo111/crypto_model", hf_folder: str = "model", cache_dir: str = "./models"):
        self.hf_repo = hf_repo
        self.hf_folder = hf_folder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cpu')
        self.models = {}  # symbol -> model
        self.model_info = {}  # symbol -> {input_size, hidden_size, ...}
        self.available_symbols = []
        
        logger.info(f"🤖 ModelManager initialized (HF repo: {hf_repo}/{hf_folder})")
    
    def list_available_models(self) -> List[str]:
        """
        List all available model files from Hugging Face
        
        Returns:
            List of symbol names (e.g., ['BTC', 'ETH', 'SOL', ...])
        """
        try:
            logger.info(f"📋 Fetching model list from {self.hf_repo}/{self.hf_folder}...")
            
            files = list_repo_files(repo_id=self.hf_repo, repo_type="model")
            
            symbols = []
            for file in files:
                # Match pattern: <SYMBOL>_model_v8.pth
                match = re.match(rf"^{self.hf_folder}/([A-Za-z]+)_model_v\d+\.pth$", file)
                if match:
                    symbol = match.group(1).upper()
                    symbols.append(symbol)
            
            self.available_symbols = sorted(list(set(symbols)))
            logger.info(f"✓ Found {len(self.available_symbols)} models: {', '.join(self.available_symbols)}")
            return self.available_symbols
        
        except Exception as e:
            logger.error(f"✗ Failed to list models: {e}")
            return []
    
    def detect_model_architecture(self, checkpoint: Dict) -> Tuple[int, int]:
        """
        Detect input_size and hidden_size from model weights
        
        For bidirectional LSTM:
        lstm.weight_ih_l0 shape = [hidden_size * 8, input_size]
        """
        try:
            if 'lstm.weight_ih_l0' in checkpoint:
                weight_shape = checkpoint['lstm.weight_ih_l0'].shape
                first_dim = weight_shape[0]
                input_size = weight_shape[1]
                
                # hidden_size = first_dim / 8 for bidirectional LSTM
                hidden_size = first_dim // 8
                
                return input_size, hidden_size
            
            logger.warning("  Could not find lstm.weight_ih_l0 in checkpoint")
            return 44, 32  # Default
        except Exception as e:
            logger.error(f"  Error detecting architecture: {e}")
            return 44, 32
    
    def download_and_load_model(self, symbol: str, force_download: bool = False) -> Optional[torch.nn.Module]:
        """
        Download model from Hugging Face and load it
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            force_download: Force re-download even if cached
        
        Returns:
            Loaded PyTorch model or None if failed
        """
        try:
            symbol = symbol.upper()
            
            # Check cache first
            cache_file = self.cache_dir / f"{symbol}_model_v8.pth"
            
            if cache_file.exists() and not force_download:
                logger.info(f"  Using cached model for {symbol}")
                return self._load_checkpoint(symbol, cache_file)
            
            # Download from Hugging Face
            logger.info(f"  Downloading {symbol} model from Hugging Face...")
            
            hf_filename = f"{self.hf_folder}/{symbol}_model_v8.pth"
            
            model_path = hf_hub_download(
                repo_id=self.hf_repo,
                filename=hf_filename,
                repo_type="model",
                local_dir=str(self.cache_dir),
                local_dir_use_symlinks=False
            )
            
            logger.info(f"  ✓ Downloaded {symbol} model")
            return self._load_checkpoint(symbol, model_path)
        
        except Exception as e:
            logger.error(f"  ✗ Failed to load {symbol}: {e}")
            return None
    
    def _load_checkpoint(self, symbol: str, model_path: Path) -> Optional[torch.nn.Module]:
        """
        Load checkpoint and create model
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Detect architecture
            input_size, hidden_size = self.detect_model_architecture(checkpoint)
            
            # Create model
            model = CryptoLSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                output_size=1
            )
            
            # Load weights
            model.load_state_dict(checkpoint)
            model.eval()
            
            # Store info
            self.models[symbol] = model
            self.model_info[symbol] = {
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': 2,
                'output_size': 1,
                'path': str(model_path)
            }
            
            logger.info(f"✓ {symbol}: input_size={input_size}, hidden_size={hidden_size}")
            return model
        
        except Exception as e:
            logger.error(f"✗ Failed to load checkpoint {symbol}: {e}")
            return None
    
    def load_all_models(self, symbols: Optional[List[str]] = None) -> Dict[str, torch.nn.Module]:
        """
        Load multiple models at once
        
        Args:
            symbols: List of symbols to load. If None, load all available.
        
        Returns:
            Dictionary of loaded models
        """
        if not symbols:
            symbols = self.list_available_models()
        
        logger.info(f"\n📦 Loading {len(symbols)} models...")
        
        for symbol in symbols:
            if symbol not in self.models:
                self.download_and_load_model(symbol)
        
        logger.info(f"✓ Total loaded: {len(self.models)} models\n")
        return self.models
    
    def get_model_info(self, symbol: str) -> Optional[Dict]:
        """
        Get model information for debugging
        """
        return self.model_info.get(symbol.upper())
    
    def get_all_model_info(self) -> Dict:
        """
        Get info for all loaded models
        """
        return self.model_info
    
    def predict(self, symbol: str, features: torch.Tensor) -> Optional[float]:
        """
        Make prediction using model
        
        Args:
            symbol: Cryptocurrency symbol
            features: Input features tensor (44-dimensional)
        
        Returns:
            Predicted price or None
        """
        symbol = symbol.upper()
        
        if symbol not in self.models:
            logger.warning(f"No model found for {symbol}")
            return None
        
        try:
            model = self.models[symbol]
            
            # Ensure correct shape
            if features.dim() == 1:
                features = features.unsqueeze(0).unsqueeze(0)
            elif features.dim() == 2:
                features = features.unsqueeze(0)
            
            with torch.no_grad():
                prediction = model(features).item()
            
            return prediction
        
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return None


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test
    manager = ModelManager()
    symbols = manager.list_available_models()
    
    if symbols:
        manager.load_all_models(symbols[:3])  # Load first 3 as test
        
        # Print model info
        for symbol, info in manager.get_all_model_info().items():
            print(f"\n{symbol}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
