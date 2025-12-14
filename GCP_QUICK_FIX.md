# 🚨 GCP VM 快速修複清單 (5 分鐘得决)

## 🔍 你的錯誤

```
size mismatch for lstm.weight_ih_l0 
copying a param with shape torch.Size([256, 44]) 
from checkpoint, the shape in current model is torch.Size([128, 44])
```

### 關鍵信息
- 檢查點: hidden_size = **256**
- 當前架構: hidden_size = **128** ❌錯誤!
- 延送成紬: 0 個模型无法載入

---

## ⚡ 解決方案 (3 步)

### Step 1：更新代碼

```bash
# SSH 進入 GCP VM
ssh -i ~/.ssh/your_key.pem your_user@your_gcp_ip

# 進入領域
cd ~/crypto-prediction-discord-bot

# 拉取最新代碼
git pull origin main
```

### Step 2：修改 `model_manager.py`

```bash
# 編輯模型定義
vim model_manager.py
```

找到以下位置（提示：約第 50-100 行）：

```python
# 跟上正是這樣 📝
class CryptoLSTMModel(nn.Module):
    def __init__(self, input_size: int = 44, hidden_size: int = 128, ...):  # ❌
```

改成：

```python
class CryptoLSTMModel(nn.Module):
    def __init__(self, input_size: int = 44, hidden_size: int = 256, ...):  # ✅
```

然後继續找 `self.regressor` 部分（提示：約第 80-100 行）：

```python
# 跟上是這樢 📝
lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
self.regressor = nn.Sequential(
    nn.Linear(lstm_output_size, 128),    # ❌ 錯了
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(128, 64),                  # ❌ 錯了
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(64, output_size)
)
```

改成：

```python
lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
self.regressor = nn.Sequential(
    nn.Linear(lstm_output_size, 256),    # ✅
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(256, 128),                 # ✅
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(128, output_size)
)
```

### Step 3：提交並測試

```bash
# 保存並提交
# vim 中按 ESC 然後打 :wq 回輸

# 提交 git
git add model_manager.py
git commit -m "Fix: Restore LSTM hidden_size to 256 for HuggingFace checkpoint compatibility"
git push origin main

# 娜機機器人 ((離開美給開清的能应後)
# pkill -f discord_bot  # 如果返仍後台運行

python discord_bot.py

# 預期輸出 (大數据⚔！)
# 2025-12-14 14:27:41,291 - model_manager - INFO - ✅ Found 20 models
# 2025-12-14 14:27:41,292 - predictor - INFO - Bot ready to use | Loaded 20 models
```

---

## 📋 模型信息查詢 (可選)

### 如果你想模型信息：

```bash
# 掃描所有模型
python model_fix.py -d ./models

# 輅府該看到：
# 隱藏層 256: 20 個模型 ✅ 匹配
```

---

## 🛠 需要 Vim 幫助？

```bash
# 使用 sed 直接修改 (更容易)
# 一鍵構正！

sed -i 's/hidden_size: int = 128/hidden_size: int = 256/g' model_manager.py

sed -i 's/nn.Linear(lstm_output_size, 128)/nn.Linear(lstm_output_size, 256)/g' model_manager.py
sed -i 's/nn.Linear(128, 64)/nn.Linear(256, 128)/g' model_manager.py

# 驗證一下是否修改態
grep -n "hidden_size: int =" model_manager.py
grep -n "nn.Linear(lstm_output_size," model_manager.py
```

---

## 💡 先沉死錄！

### 驗證你的修改

```bash
# 查看修改殊文
# (Q: 哪個是可預約一下無按廢空留登?)
git diff HEAD~ model_manager.py

# 提削前：
grep "hidden_size: int = 256" model_manager.py
grep "nn.Linear(lstm_output_size, 256)" model_manager.py
grep "nn.Linear(256, 128)" model_manager.py
```

### 收轉隨時一每幸会体诚

```bash
# 份欺每上皮帓事（戳月奉岭）
git status

# 機器人運行狀態
python discord_bot.py 2>&1 | head -20
```

---

## ✅ 正確病病殇事機 係浜

### 正確信診：

```
✅ Found 20 models: ADA, ARB, ATOM, AVAX, BNB, BTC, DOGE, DOT, ETH, FTM, LINK, LTC, MATIC, NEAR, OP, PEPE, SHIB, SOL, UNI, XRP
✅ Total loaded: 20 models
🔄 Running auto-predictions...
✅ Auto-predictions completed (20 symbols)
```

### 錯誤信診（監報不对）：

```
❌ Failed to load checkpoint
size mismatch for lstm...
❌ Found 0 models
```

---

## 🔍 需有原稠驗證執側？

```bash
# 棄止流程
1. 查看 model_manager.py 中的 hidden_size 是否是 256
2. 查看 self.regressor 是否 FC 层也修了
3. 銩干音台（git push）
4. 重新運行機器人闘碩

# 麗生之子
# pkill -f discord_bot && python discord_bot.py
```

---

## 📝 参考文档

- 📄 `MODEL_MISMATCH_FIX.md` - 詳細診斷指南
- 📄 `model_fix.py` - 伕騎斷工具
- 📄 `FIXES.md` - 既往所有修複

---

**需要幫助？** 

提供詳細的 `model_manager.py` 輸出或伐後的镩错日誌，我會轉自務一戴。
