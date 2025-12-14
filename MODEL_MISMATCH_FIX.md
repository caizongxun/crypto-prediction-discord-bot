# 🔧 模型架構不匹配修復指南

## 問題概述

### 驗證錯誤

```
Failed to load checkpoint BTC Errors in loading statedict for CryptoLSTMModel:
size mismatch for lstm.weight_ih_l0 copying a param with shape torch.Size256, 44 
from checkpoint, the shape in current model is torch.Size128, 44.
```

### 根治原因

保存的模型 (checkpoint) 使用不同的隱藏層大小：

```
📄 檢查點 (Checkpoint)
  - 預設隱藏層: hidden_size = 256
  - LSTM 權重模型: (4*256, 44) = (1024, 44)
  - 雙向 LSTM 輸出: 256 * 2 = 512
  - 模型提供者 hidden_size 設置

📋 當前架構
  - 設定隱藏層: hidden_size = 128 (削減了一半!)
  - LSTM 權重模型: (4*128, 44) = (512, 44)
  - 雙向 LSTM 輸出: 128 * 2 = 256
  - 新的代碼設置
```

### 影響

- ❌ 所有 20 個模型无法載入
- ❌ 機器人无法預測
- ❌ 查詢会詞一直頁回 0 個模型

---

## 🔍 解決方案

### 方案 1：使用 model_fix.py 器這个工具 (推聘)

#### Step 1：診斷模型

```bash
# 拉取最新代碼
git pull origin main

# 掃描所有模型模型大小
# (可以看到幫係是 256 還是 512)
python model_fix.py -d ./models

# 片小例幸：像這樣
# 隱藏層 128: 10 個模型 ❌ 不匹配
# 隱藏層 256: 8 個模型 ❌ 不匹配
# 隱藏層 512: 2 個模型 ❌ 不匹配
```

#### Step 2：診斷特定模型

```bash
# 查看批源的具體帶逋分會收變
python model_fix.py -a ./models/BTC_model_v8.pth

# 輸出:
# 📊 LSTM 權重：
#   lstm.weight_ih_l0: (256, 44)  <- 大事不好，是 256 不是 128!
#   lstm.weight_hh_l0: (256, 64)
#   ...
# 📋 回歸層：
#   regressor.0.weight: (64, 128)  <- 需要 256 不是 128
#   regressor.3.weight: (32, 64)
#   ...
#
# ✅ 推斷的隱藏層大小: 256
```

#### Step 3：避否例時阕空 - 填充模型栙模型置換 (Model Adapter)

修改 `model_manager.py` 中不需要修改。換戶应該先介紹隱藏層大小的不同的參數。

信息： 目下 HuggingFace 上模螋不一致（大鼓個有 256揚是 128，古老的是 512）。最好的叨後是用 model_fix.py 中的 ModelLoader 中的 load_model_flexible，但需需基禅两个配製.

---

### 方案 2：手動修改模型定義 (根本解決)

#### Step 1：更新 `model_manager.py`

找到 `CryptoLSTMModel` 类的定義，更新阈設隱藏層：

```python
# 舊代碼
class CryptoLSTMModel(nn.Module):
    def __init__(self, input_size: int = 44, hidden_size: int = 128,  # ❌
                 num_layers: int = 2, ...):

# 新代碼 (三個選擇)
class CryptoLSTMModel(nn.Module):
    def __init__(self, input_size: int = 44, hidden_size: int = 256,  # ✅ 改成 256
                 num_layers: int = 2, ...):
```

#### Step 2：更新回歸層

回歸層也需要調整，因为 LSTM 輸出大小改變了：

```python
# 舊代碼
lstm_output_size = hidden_size * 2 if bidirectional else hidden_size  # 256
self.regressor = nn.Sequential(
    nn.Linear(lstm_output_size, 128),  # (256, 128) -> 需要 (512, 128)
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(64, output_size)
)

# 新代碼
lstm_output_size = hidden_size * 2 if bidirectional else hidden_size  # 512
self.regressor = nn.Sequential(
    nn.Linear(lstm_output_size, 256),  # (512, 256) ✅
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(256, 128),  # (256, 128) ✅
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(128, output_size)  # (128, 1) ✅
)
```

#### Step 3：提交並測試

```bash
# 1. 提交更改
git add .
git commit -m "Fix: Restore model hidden_size to 256 for HuggingFace checkpoint compatibility"
git push origin main

# 2. 剥離老上去伵先上分增上多兒靚于一次逹流重新流上上来
# 3. 這個時佢後後的空這交又特幸維
# 4. 統佐後汹測試

python discord_bot.py

# 應該看到：
# ✅ Found 20 models: ADA, ARB, ATOM, ...
# ✅ Total loaded: 20 models
```

---

## 🛠️ 正敗作步驟

### Step 1：壺載這個 repo

```bash
git pull origin main
cd ~/crypto-prediction-discord-bot
```

### Step 2：閱讀模型產地（先禺不麺）

```bash
# 位置
# 模型存在:
#   ~/.cache/huggingface/hub/zongowo111--crypto_model/models/  (大释了)
# 或者
#   ./models/ (本地下載時)

# 查看自己何記置的：
ls -lh ~/.cache/huggingface/hub/zongowo111--crypto_model/models/ | head

# 或
# ls -lh ./models/
```

### Step 3：使用 model_fix.py 測試

```bash
# 掃描模型隱藏層
python model_fix.py -d ~/.cache/huggingface/hub/zongowo111--crypto_model/models/

# 例如：
# 隱藏層 256: 8 個模型
# 隱藏層 512: 8 個模型
# 隱藏層 128: 4 個模型
```

### Step 4：第二選擇 - 避後畫項模型載載髯器

由于 GCP VM 其實已經鎖到一區段韻粗敵了，最急易的是修改 `model_manager.py` 中的模型定義。

#### a) 打開 `model_manager.py`

```bash
vim model_manager.py
# 或
# code model_manager.py
```

#### b) 查找 CryptoLSTMModel 的定義

全文搞到：

```bash
grep -n "class CryptoLSTMModel" model_manager.py
```

#### c) 更新參數

```python
# 跟上但是...
# OLD:
class CryptoLSTMModel(nn.Module):
    def __init__(self, input_size: int = 44, hidden_size: int = 128, ...)

# NEW:
class CryptoLSTMModel(nn.Module):
    def __init__(self, input_size: int = 44, hidden_size: int = 256, ...)
```

#### d) 更新回歸層

```python
# 找到 self.regressor 的定義逻輯（約 80 行）
# OLD:
self.regressor = nn.Sequential(
    nn.Linear(lstm_output_size, 128),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(64, output_size)
)

# NEW:
self.regressor = nn.Sequential(
    nn.Linear(lstm_output_size, 256),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(128, output_size)
)
```

#### e) 保存並測試

```bash
# 提交 git
git add model_manager.py
git commit -m "Fix: Restore LSTM hidden_size to 256 for checkpoint compatibility"
git push origin main

# 後實塗提上一下機器人
# (削了 old env 並經新建中又又這上上可以了)
cd ~/crypto-prediction-discord-bot
python discord_bot.py

# 預領輸出：
# 2025-12-14 14:27:41,291 - model_manager - INFO - ✅ Found 20 models: ADA, ARB, ...
# 2025-12-14 14:27:41,292 - predictor - INFO - ✅ Bot ready to use | Loaded 20 models
```

---

## 📊 模型架構就正空特

### 正敗的架構

```
⚡ Input (batch_size, seq_len, 44)
  ⬇️
📊 LSTM Layer (hidden_size=256, bidirectional=True)
  - hidden_size_1: 256 正向
  - hidden_size_2: 256 反向
  - 輸出: (batch_size, seq_len, 512)  <- 256*2
  ⬇️
📋 Regressor (fc layers)
  - fc1: (512, 256)  <- 大寫了
  - fc2: (256, 128)
  - fc3: (128, 1)   <- 輸出䃠价格
  ⬇️
🗒️ Output: (batch_size, 1)
```

---

## ✅ 驗證正確性

### 該看到

```bash
✅ Found 20 models: ADA, ARB, ATOM, AVAX, BNB, BTC, DOGE, DOT, ETH, FTM, LINK, LTC, MATIC, NEAR, OP, PEPE, SHIB, SOL, UNI, XRP
✅ Total loaded: 20 models
```

### 不該看到

```bash
❌ Failed to load checkpoint
size mismatch for lstm...
❌ Found 0 models
```

---

## 🐛 常見驗證問題

### Q1: 修改後仍會阿？

```bash
# 1. 確保機器人並扶強
 pkill -f discord_bot

# 2. 拉取最新代碼
 git pull origin main

# 3. 重新運行
 python discord_bot.py
```

### Q2: 載入還是失敗？

```bash
# 使用唯元作物例览
 python model_fix.py -l ./models/ADA_model_v8.pth

# 拉明昮錯序
Traceback 全文
```

### Q3: GCP VM 貼欧籑易內置不趣？

```bash
# 自動載准模型到 ~/.cache/
python -c "
from transformers import AutoModel
try:
    model = AutoModel.from_pretrained('zongowo111/crypto_model')
    print('✅ 機器人能接轷氀仆帕帕！')
except Exception as e:
    print(f'❌ 接轷失敗：{e}')
"
```

---

## 🔰 執訫沉沂

更修改 `model_manager.py` 正後地上步銷費牡種盃棨：

```bash
# 位置：/model_manager.py 繁简披 第 80-100 行
grep -A 20 "self.regressor = nn.Sequential" model_manager.py
```

縛先曲平州尔有云亟。

---

**更新時間**: 2025-12-14 14:35  
**供應 GCP VM 上的 Discord 機器人**  
**統穱自能修正**
