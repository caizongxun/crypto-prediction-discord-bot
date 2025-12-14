# 🔍 線程監控完整指南

## 概述

查看後台執行線程的 4 種方法，適用於調試 Discord 機器人和 Web 儀表板。

---

## 📋 方法 1：Python 內置 threading 模塊

### 最簡單的方式

```python
import threading

# 獲取所有活躍線程
all_threads = threading.enumerate()
print(f"線程數: {threading.active_count()}")

for thread in all_threads:
    print(f"{thread.name}: {thread.ident}")
```

### 在 Discord 機器人中使用

在機器人運行時獲取線程信息：

```python
from predictor import CryptoPredictor
import threading

predictor = CryptoPredictor()
predictor.initialize()

# 查看線程
print(f"\n活躍線程: {threading.active_count()}")
for thread in threading.enumerate():
    print(f"  - {thread.name} (ID: {thread.ident})")
```

---

## 📊 方法 2：系統級線程監控（psutil）

### 獲取詳細系統信息

```python
import psutil
import os

# 獲取當前進程
current_pid = os.getpid()
p = psutil.Process(current_pid)

print(f"進程 ID: {current_pid}")
print(f"進程名稱: {p.name()}")
print(f"線程總數: {p.num_threads()}")
print(f"進程狀態: {p.status()}")
print(f"記憶體使用: {p.memory_info().rss / 1024 / 1024:.2f} MB")

# 查看每個線程的 CPU 時間
for thread in p.threads():
    print(f"\n線程 #{thread.id}")
    print(f"  用戶 CPU: {thread.user_time:.3f}s")
    print(f"  系統 CPU: {thread.system_time:.3f}s")
```

---

## 🔧 方法 3：使用 thread_monitor.py 工具

### 基本用法

```bash
# 查看所有線程
python thread_monitor.py

# 詳細模式
python thread_monitor.py -v

# 只顯示後台線程
python thread_monitor.py -d

# 顯示堆棧跟蹤
python thread_monitor.py -s

# 實時監控（每 2 秒更新，監控 10 秒）
python thread_monitor.py -l

# 自定義監控間隔
python thread_monitor.py -l -i 1 -t 30
```

### 輸出示例

```
================================================================================
🔍 PYTHON 線程信息 (threading 模塊)
================================================================================

總線程數: 6

線程名稱                  線程 ID         Daemon  活躍  狀態           
--------------------------------------------------------------------------------
MainThread               140206779    ✗       ✓       🟢 運行中      
Discord bot client       140207456    ✓       ✓       🟢 運行中      
auto_predict             140207789    ✓       ✓       🟢 運行中      
Thread-1                 140207234    ✓       ✓       🟢 運行中      
Thread-2                 140207567    ✓       ✓       🟢 運行中      

================================================================================
🔍 系統線程信息 (psutil 模塊)
================================================================================

進程 ID: 12345
進程名稱: python3
線程狀態: running
線程總數: 6

線程 ID    用戶 CPU(s)    系統 CPU(s)    總 CPU(s)  
-------------------------------------------------------
12345      1.250          0.350          1.600
12346      0.050          0.030          0.080
12347      0.020          0.015          0.035
```

---

## 🚨 方法 4：Linux 命令行工具

### 查看進程線程

```bash
# 查看 Python 進程的所有線程
ps aux | grep python

# 獲取進程 ID（假設為 12345）
pid=12345

# 查看該進程的線程數
cat /proc/$pid/status | grep Threads

# 列出所有線程
ls -la /proc/$pid/task/

# 查看線程 CPU 使用
ps -eLf | grep $pid
```

### 使用 htop 實時監控

```bash
# 安裝 htop
sudo apt-get install htop

# 運行 htop 並按 'H' 顯示線程
htop

# 或直接查看線程
htop -H
```

### 使用 top 命令

```bash
# 進入 top，按 'H' 切換線程視圖
top

# 或直接查看線程
top -H

# 查看特定進程的線程
top -H -p 12345
```

---

## 🎯 Discord 機器人線程分析

### 預期線程

當運行 `python discord_bot.py` 時，應該看到：

```
線程名稱                描述
─────────────────────────────────────────────────────────────
MainThread              主線程，運行機器人
Discord.py client       Discord 客戶端事件循環
auto_predict            自動預測後台任務（每小時運行）
Websocket client        Discord WebSocket 連接
IO threads              I/O 操作線程（可能有多個）
```

### 監控後台任務

```python
# 檢查自動預測任務
import threading

for thread in threading.enumerate():
    if 'auto_predict' in thread.name or 'task' in thread.name:
        print(f"後台任務: {thread.name}")
        print(f"  活躍: {thread.is_alive()}")
        print(f"  Daemon: {thread.daemon}")
```

---

## 🐛 調試常見線程問題

### 1. 線程數不斷增加

**症狀**: 線程數逐漸增加，佔用記憶體

**診斷**:
```python
import threading
import time

for i in range(5):
    print(f"線程數 ({i}): {threading.active_count()}")
    time.sleep(1)
```

**解決方案**:
- 檢查是否有未清理的線程
- 確保所有後台任務正確終止
- 使用 `thread.daemon = True` 設置守護線程

### 2. 線程掛起

**症狀**: 機器人停止響應

**診斷**:
```bash
# 查看線程狀態
python thread_monitor.py -s

# 查看堆棧跟蹤，確定卡在哪裡
```

**解決方案**:
- 添加超時機制
- 使用 asyncio 代替多線程
- 檢查死鎖

### 3. CPU 使用過高

**症狀**: CPU 使用率高達 100%

**診斷**:
```bash
# 查看線程 CPU 使用
python thread_monitor.py

# 查看哪個線程佔用最多 CPU
top -H -p $(pgrep -f 'python discord_bot.py')
```

**解決方案**:
- 優化計算密集型任務
- 添加睡眠時間（`time.sleep()`）
- 使用連接池減少重複連接

---

## 📈 性能監控指標

### 健康的線程配置

```
✅ 正常狀態
- 線程數: 4-8 個
- CPU 使用: < 10% (閒置時)
- 記憶體使用: < 200 MB
- 後台任務: 定期執行

⚠️ 警告狀態
- 線程數: 10-20 個
- CPU 使用: 10-50%
- 記憶體使用: 200-500 MB
- 線程卡頓: > 1 分鐘

🔴 緊急狀態
- 線程數: > 20 個
- CPU 使用: > 80%
- 記憶體使用: > 1 GB
- 多個線程掛起
```

---

## 🛠️ 集成到機器人

### 添加監控命令

```python
@bot.tree.command(name="threads", description="查看當前線程")
async def threads(interaction: discord.Interaction):
    """
    查看後台線程信息
    """
    import threading
    
    threads_list = threading.enumerate()
    
    embed = discord.Embed(
        title="🔍 線程監控",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name=f"活躍線程: {len(threads_list)}",
        value="\n".join([f"• {t.name}" for t in threads_list[:10]]),
        inline=False
    )
    
    await interaction.response.send_message(embed=embed)
```

---

## 📚 完整腳本

```python
#!/usr/bin/env python3
"""
完整線程監控腳本
"""

import threading
import psutil
import os
from datetime import datetime

def show_all_info():
    print("\n" + "="*80)
    print(f"🔍 線程監控報告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Python 線程
    print("\n📋 PYTHON 線程:")
    print(f"  活躍線程: {threading.active_count()}")
    for thread in threading.enumerate():
        daemon = "✓" if thread.daemon else "✗"
        alive = "✓" if thread.is_alive() else "✗"
        print(f"    • {thread.name:<20} (Daemon: {daemon}, 活躍: {alive})")
    
    # 系統線程
    print("\n📊 系統信息:")
    p = psutil.Process(os.getpid())
    print(f"  進程 ID: {p.pid}")
    print(f"  線程總數: {p.num_threads()}")
    print(f"  記憶體使用: {p.memory_info().rss / 1024 / 1024:.2f} MB")
    print(f"  CPU 使用: {p.cpu_percent():.1f}%")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    show_all_info()
```

---

## 📞 故障排除

### 無法導入 psutil

```bash
# 安裝 psutil
pip install psutil
```

### 權限不足（Linux）

```bash
# 某些信息可能需要 sudo
sudo python thread_monitor.py
```

### 在 Docker 容器中使用

```bash
# 在容器中運行監控
docker exec <container_id> python thread_monitor.py
```

---

**更新時間**: 2025-12-14  
**最後修改**: Discord 機器人線程監控指南
