#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理超参数进化产生的临时文件夹脚本
用法：python cleanup_evolve_dirs.py [--dry-run]
"""

import os
import shutil
import argparse
from datetime import datetime, timedelta

# 设置中文显示
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def cleanup_evolve_dirs(root_dir, dry_run=False):
    """清理指定目录下的临时训练文件夹"""
    # 要清理的临时文件夹名称模式
    temp_dir_patterns = ['train', 'train2', 'train3', 'train4', 'train5']
    
    # 遍历根目录
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        
        # 检查是否为目录且名称符合临时文件夹模式
        if os.path.isdir(dir_path) and dir_name in temp_dir_patterns:
            # 检查目录中是否包含weights文件夹（YOLO训练的特征）
            weights_dir = os.path.join(dir_path, 'weights')
            if os.path.exists(weights_dir):
                print(f"发现临时训练文件夹: {dir_path}")
                
                # 检查是否为dry-run模式
                if dry_run:
                    print(f"  模拟删除: {dir_path}")
                else:
                    try:
                        shutil.rmtree(dir_path)
                        print(f"  已删除: {dir_path}")
                    except Exception as e:
                        print(f"  错误: 无法删除 {dir_path}，错误: {e}")

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='清理超参数进化产生的临时文件夹')
    parser.add_argument('--dry-run', action='store_true', help='仅显示要删除的文件夹，不实际删除')
    parser.add_argument('--root-dir', default='.', help='要清理的根目录路径')
    args = parser.parse_args()
    
    print(f"开始清理临时训练文件夹 (根目录: {args.root_dir})")
    if args.dry_run:
        print("运行模式: 模拟删除 (dry-run)")
    else:
        print("运行模式: 实际删除")
    
    cleanup_evolve_dirs(args.root_dir, args.dry_run)
    print("清理完成")