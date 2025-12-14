#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型架構診斷和修復工具

問題: 保存的模型 (checkpoint) 使用不同的隱藏層大小
  - 大多數模型: LSTM hidden_size=256 (雙向 = 512)
  - 當前架構: LSTM hidden_size=128 (雙向 = 256)

解決方案:
  1. 動態檢測模型尺寸並重新初始化
  2. 使用州字典映射載入
  3. 或重新訓練模型
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoLSTMModel(nn.Module):
    """動態 LSTM 模型架構"""
    
    def __init__(self, input_size: int = 44, hidden_size: int = 128, 
                 num_layers: int = 2, output_size: int = 1, 
                 dropout: float = 0.3, bidirectional: bool = True):
        super(CryptoLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM 層
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 回歸層
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最後一步輸出
        last_output = lstm_out[:, -1, :]
        output = self.regressor(last_output)
        return output


class ModelDiagnostic:
    """模型診斷工具"""
    
    @staticmethod
    def analyze_checkpoint(checkpoint_path: str) -> Dict:
        """
        分析檢查點中的模型大小
        
        Args:
            checkpoint_path: .pth 文件路徑
            
        Returns:
            包含模型信息的字典
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint else checkpoint.get('state_dict', checkpoint)
            
            info = {
                'path': checkpoint_path,
                'lstm_weights': {},
                'regressor_shapes': {}
            }
            
            # 分析 LSTM 層
            for key, param in state_dict.items():
                if 'lstm' in key:
                    info['lstm_weights'][key] = tuple(param.shape)
                elif 'regressor' in key:
                    info['regressor_shapes'][key] = tuple(param.shape)
            
            return info
        except Exception as e:
            logger.error(f"❌ 無法分析檢查點: {e}")
            return {}
    
    @staticmethod
    def get_hidden_size_from_checkpoint(checkpoint_path: str) -> Optional[int]:
        """
        從檢查點推斷隱藏層大小
        
        Args:
            checkpoint_path: .pth 文件路徑
            
        Returns:
            隱藏層大小，如果無法確定則返回 None
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint else checkpoint.get('state_dict', checkpoint)
            
            # 查找 LSTM 權重來推斷隱藏層大小
            for key, param in state_dict.items():
                if 'lstm.weight_ih_l0' in key:
                    # LSTM 輸入到隱藏層的權重大小是 (4*hidden_size, input_size)
                    hidden_size = param.shape[0] // 4
                    logger.info(f"📊 推斷隱藏層大小: {hidden_size} (從 {key} 權重 {param.shape})")
                    return hidden_size
            
            return None
        except Exception as e:
            logger.error(f"❌ 推斷隱藏層大小失敗: {e}")
            return None


class ModelLoader:
    """智能模型載入器"""
    
    @staticmethod
    def load_model_flexible(checkpoint_path: str, target_hidden_size: int = 128,
                           map_location: str = 'cpu') -> Optional[torch.nn.Module]:
        """
        靈活載入模型，自動適配不同的隱藏層大小
        
        Args:
            checkpoint_path: 模型檢查點路徑
            target_hidden_size: 目標隱藏層大小
            map_location: PyTorch 設備位置
            
        Returns:
            載入的模型或 None
        """
        try:
            # 分析檢查點
            diagnostic = ModelDiagnostic()
            checkpoint_hidden_size = diagnostic.get_hidden_size_from_checkpoint(checkpoint_path)
            
            if checkpoint_hidden_size is None:
                logger.warning(f"⚠️  無法推斷隱藏層大小，使用目標大小: {target_hidden_size}")
                checkpoint_hidden_size = target_hidden_size
            
            logger.info(f"\n📋 載入模型信息:")
            logger.info(f"   檢查點隱藏層: {checkpoint_hidden_size}")
            logger.info(f"   目標隱藏層: {target_hidden_size}")
            
            # 載入檢查點
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            state_dict = checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint else checkpoint.get('state_dict', checkpoint)
            
            # 如果大小匹配，直接載入
            if checkpoint_hidden_size == target_hidden_size:
                logger.info(f"✅ 隱藏層大小匹配，直接載入")
                model = CryptoLSTMModel(hidden_size=target_hidden_size)
                model.load_state_dict(state_dict, strict=False)
                return model
            
            # 如果大小不匹配，需要大型模型
            logger.info(f"🔄 隱藏層大小不匹配，使用 {checkpoint_hidden_size} 載入")
            model = CryptoLSTMModel(hidden_size=checkpoint_hidden_size)
            model.load_state_dict(state_dict, strict=False)
            
            # 可選: 量化到較小的模型
            if checkpoint_hidden_size > target_hidden_size:
                logger.info(f"📉 正在將模型從 {checkpoint_hidden_size} 量化到 {target_hidden_size}...")
                model = ModelLoader._quantize_model(model, target_hidden_size)
            
            return model
            
        except Exception as e:
            logger.error(f"❌ 載入模型失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def _quantize_model(model: torch.nn.Module, target_hidden_size: int) -> torch.nn.Module:
        """
        將模型從大隱藏層量化到小隱藏層
        
        Args:
            model: 原始模型
            target_hidden_size: 目標隱藏層大小
            
        Returns:
            量化後的模型
        """
        # 新建較小的模型
        small_model = CryptoLSTMModel(hidden_size=target_hidden_size)
        
        # 複製可以直接映射的層
        try:
            # 複製 LSTM 的部分權重 (簡單的方法是平均)
            for (name_src, param_src), (name_tgt, param_tgt) in zip(
                model.named_parameters(), small_model.named_parameters()
            ):
                if name_src == name_tgt:
                    if param_src.shape == param_tgt.shape:
                        param_tgt.data.copy_(param_src.data)
                    else:
                        # 簡單的尺寸調整 (可以改進)
                        if param_src.dim() >= 2:
                            param_tgt.data.copy_(param_src.data[:param_tgt.shape[0], :param_tgt.shape[1]] 
                                               if param_src.shape[0] >= param_tgt.shape[0] else param_src.data)
            
            logger.info(f"✅ 量化完成: {model.hidden_size} -> {target_hidden_size}")
        except Exception as e:
            logger.warning(f"⚠️  量化過程中出錯: {e}")
        
        return small_model


def main():
    """
    主診斷函數
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="🔍 模型架構診斷工具")
    parser.add_argument('-a', '--analyze', type=str, help='分析檢查點文件')
    parser.add_argument('-l', '--load', type=str, help='載入並測試模型')
    parser.add_argument('-d', '--directory', type=str, default='./models', help='模型目錄')
    parser.add_argument('-hs', '--hidden-size', type=int, default=128, help='目標隱藏層大小')
    
    args = parser.parse_args()
    
    diagnostic = ModelDiagnostic()
    loader = ModelLoader()
    
    print("\n" + "="*80)
    print("🔍 模型架構診斷工具")
    print("="*80)
    
    # 分析單個檢查點
    if args.analyze:
        print(f"\n📊 分析: {args.analyze}")
        print("-" * 80)
        info = diagnostic.analyze_checkpoint(args.analyze)
        print(f"\n📋 LSTM 權重:")
        for key, shape in info.get('lstm_weights', {}).items():
            print(f"  {key}: {shape}")
        print(f"\n📋 回歸層:")
        for key, shape in info.get('regressor_shapes', {}).items():
            print(f"  {key}: {shape}")
        
        # 推斷隱藏層大小
        hidden_size = diagnostic.get_hidden_size_from_checkpoint(args.analyze)
        print(f"\n✅ 推斷的隱藏層大小: {hidden_size}")
    
    # 載入並測試模型
    elif args.load:
        print(f"\n🔄 載入: {args.load}")
        print("-" * 80)
        model = loader.load_model_flexible(args.load, target_hidden_size=args.hidden_size)
        if model:
            print(f"✅ 模型載入成功")
            print(f"\n📊 模型架構:")
            print(model)
    
    # 掃描目錄中的所有模型
    else:
        print(f"\n📁 掃描目錄: {args.directory}")
        print("-" * 80)
        
        model_dir = Path(args.directory)
        model_files = list(model_dir.glob('*_model_*.pth')) + list(model_dir.glob('*.pth'))
        
        print(f"\n找到 {len(model_files)} 個模型文件\n")
        
        results = []
        for model_file in sorted(model_files):
            hidden_size = diagnostic.get_hidden_size_from_checkpoint(str(model_file))
            results.append({
                'name': model_file.name,
                'hidden_size': hidden_size
            })
        
        print(f"{'模型':<30} {'隱藏層':<12} {'狀態'}")
        print("-" * 60)
        
        for result in results:
            status = "✅" if result['hidden_size'] == args.hidden_size else "❌"
            print(f"{result['name']:<30} {result['hidden_size']:<12} {status}")
        
        # 統計
        print("\n" + "-" * 60)
        print(f"\n📊 統計:")
        size_groups = {}
        for result in results:
            size = result['hidden_size']
            size_groups[size] = size_groups.get(size, 0) + 1
        
        for size, count in sorted(size_groups.items()):
            match = "✅ 匹配" if size == args.hidden_size else "❌ 不匹配"
            print(f"  隱藏層 {size}: {count} 個模型 {match}")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
