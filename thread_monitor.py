#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
線程監控工具 - 查看所有後台執行線程

Usage:
    python thread_monitor.py
"""

import threading
import psutil
import os
import sys
import time
from datetime import datetime
import traceback


class ThreadMonitor:
    """線程監控類"""
    
    @staticmethod
    def get_python_threads():
        """
        獲取 Python 所有線程（使用 threading 模塊）
        
        Returns:
            List of thread info dictionaries
        """
        threads_info = []
        for thread in threading.enumerate():
            threads_info.append({
                'name': thread.name,
                'id': thread.ident,
                'daemon': thread.daemon,
                'alive': thread.is_alive(),
                'source': 'threading'
            })
        return threads_info
    
    @staticmethod
    def get_system_threads():
        """
        獲取系統級別線程信息（使用 psutil 模塊）
        
        Returns:
            List of system thread info
        """
        try:
            current_pid = os.getpid()
            p = psutil.Process(current_pid)
            
            threads_info = []
            for thread in p.threads():
                threads_info.append({
                    'id': thread.id,
                    'user_time': thread.user_time,
                    'system_time': thread.system_time,
                    'source': 'psutil'
                })
            return threads_info
        except Exception as e:
            print(f"❌ Error getting system threads: {e}")
            return []
    
    @staticmethod
    def get_thread_stacks():
        """
        獲取所有線程的堆棧跟蹤
        
        Returns:
            Dictionary with thread name -> stack trace
        """
        stacks = {}
        for thread_id, stack in sys._current_frames().items():
            # 匹配線程
            for thread in threading.enumerate():
                if thread.ident == thread_id:
                    stacks[thread.name] = traceback.format_stack(stack)
                    break
        return stacks
    
    @classmethod
    def print_python_threads(cls, verbose=False):
        """
        打印 Python 線程信息
        """
        print("\n" + "="*80)
        print("🔍 PYTHON 線程信息 (threading 模塊)")
        print("="*80)
        
        threads = cls.get_python_threads()
        print(f"\n總線程數: {len(threads)}\n")
        
        # 表頭
        print(f"{'線程名稱':<25} {'線程 ID':<15} {'Daemon':<8} {'活躍':<8} {'狀態':<15}")
        print("-" * 80)
        
        # 線程列表
        for i, thread in enumerate(threads, 1):
            status = "🟢 運行中" if thread['alive'] else "🔴 已停止"
            daemon_str = "✓" if thread['daemon'] else "✗"
            alive_str = "✓" if thread['alive'] else "✗"
            
            print(f"{thread['name']:<25} {str(thread['id']):<15} {daemon_str:<8} {alive_str:<8} {status:<15}")
        
        if verbose:
            print("\n" + "-"*80)
            print("📊 詳細信息:\n")
            for thread in threads:
                print(f"  線程: {thread['name']}")
                print(f"    - ID: {thread['id']}")
                print(f"    - Daemon 線程: {'是' if thread['daemon'] else '否'}")
                print(f"    - 活躍: {'是' if thread['alive'] else '否'}")
                print()
    
    @classmethod
    def print_system_threads(cls):
        """
        打印系統級別線程信息
        """
        print("\n" + "="*80)
        print("🔍 系統線程信息 (psutil 模塊)")
        print("="*80)
        
        current_pid = os.getpid()
        p = psutil.Process(current_pid)
        
        print(f"\n進程 ID: {current_pid}")
        print(f"進程名稱: {p.name()}")
        print(f"進程狀態: {p.status()}")
        print(f"線程總數: {p.num_threads()}")
        
        threads = cls.get_system_threads()
        print(f"\n{'線程 ID':<10} {'用戶 CPU(s)':<15} {'系統 CPU(s)':<15} {'總 CPU(s)':<15}")
        print("-" * 55)
        
        total_user = 0
        total_sys = 0
        for thread in threads:
            user_time = thread['user_time']
            sys_time = thread['system_time']
            total_time = user_time + sys_time
            
            total_user += user_time
            total_sys += sys_time
            
            print(f"{thread['id']:<10} {user_time:<15.3f} {sys_time:<15.3f} {total_time:<15.3f}")
        
        print("-" * 55)
        print(f"{'總計':<10} {total_user:<15.3f} {total_sys:<15.3f} {total_user+total_sys:<15.3f}")
    
    @classmethod
    def print_thread_stacks(cls):
        """
        打印線程堆棧跟蹤
        """
        print("\n" + "="*80)
        print("🔍 線程堆棧跟蹤")
        print("="*80)
        
        stacks = cls.get_thread_stacks()
        
        for thread_name, stack in stacks.items():
            print(f"\n📌 線程: {thread_name}")
            print("-" * 80)
            for frame in stack[-3:]:  # 只顯示最後 3 幀
                print(frame.strip())
    
    @classmethod
    def print_daemon_threads(cls):
        """
        打印所有後台線程（Daemon 線程）
        """
        print("\n" + "="*80)
        print("🔴 後台線程 (Daemon Threads)")
        print("="*80)
        
        threads = cls.get_python_threads()
        daemon_threads = [t for t in threads if t['daemon']]
        
        print(f"\n後台線程數: {len(daemon_threads)}\n")
        
        for i, thread in enumerate(daemon_threads, 1):
            status = "🟢 運行中" if thread['alive'] else "🔴 已停止"
            print(f"  {i}. {thread['name']:<30} {status}")
        
        if len(daemon_threads) == 0:
            print("  (無後台線程)")
    
    @classmethod
    def monitor_live(cls, interval=2, duration=10):
        """
        實時監控線程（持續監控幾秒鐘）
        
        Args:
            interval: 更新間隔（秒）
            duration: 監控時長（秒）
        """
        print("\n" + "="*80)
        print("📊 實時線程監控（按 Ctrl+C 停止）")
        print("="*80)
        
        elapsed = 0
        try:
            while elapsed < duration:
                # 清屏
                os.system('clear' if os.name != 'nt' else 'cls')
                
                print(f"\n⏱️  實時監控 - {datetime.now().strftime('%H:%M:%S')}")
                print("="*80)
                
                threads = cls.get_python_threads()
                print(f"\n活躍線程: {len(threads)} 個\n")
                
                print(f"{'#':<3} {'線程名稱':<25} {'ID':<15} {'Daemon':<8} {'狀態':<10}")
                print("-" * 80)
                
                for i, thread in enumerate(threads, 1):
                    status = "🟢 活躍" if thread['alive'] else "🔴 停止"
                    daemon_str = "✓" if thread['daemon'] else "✗"
                    print(f"{i:<3} {thread['name']:<25} {str(thread['id']):<15} {daemon_str:<8} {status:<10}")
                
                print("\n按 Ctrl+C 停止監控...")
                time.sleep(interval)
                elapsed += interval
        
        except KeyboardInterrupt:
            print("\n\n✅ 監控已停止")


def main():
    """
    主函數
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="🔍 線程監控工具")
    parser.add_argument('-v', '--verbose', action='store_true', help='詳細模式')
    parser.add_argument('-s', '--stacks', action='store_true', help='顯示堆棧跟蹤')
    parser.add_argument('-d', '--daemon', action='store_true', help='只顯示後台線程')
    parser.add_argument('-l', '--live', action='store_true', help='實時監控')
    parser.add_argument('-i', '--interval', type=int, default=2, help='實時監控更新間隔（秒）')
    parser.add_argument('-t', '--time', type=int, default=10, help='監控時長（秒）')
    
    args = parser.parse_args()
    
    monitor = ThreadMonitor()
    
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*20 + "🔍 Discord Bot 線程監控工具" + " "*32 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    if args.live:
        monitor.monitor_live(interval=args.interval, duration=args.time)
    else:
        monitor.print_python_threads(verbose=args.verbose)
        monitor.print_system_threads()
        
        if args.daemon:
            monitor.print_daemon_threads()
        
        if args.stacks:
            monitor.print_thread_stacks()
    
    print("\n" + "#"*80 + "\n")


if __name__ == '__main__':
    main()
