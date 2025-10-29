#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集文件移动工具

此脚本用于在训练集和验证集之间移动数据文件（图片和标签）。
支持将指定文件从验证集移动到训练集，或从训练集移动到验证集。

使用方法：
    python utils/move_dataset_files.py --files file1.jpg file2.png --direction val2train
    python utils/move_dataset_files.py --files file1.jpg file2.png --direction train2val
    python utils/move_dataset_files.py --file-list file_list.txt --direction val2train
"""

import os
import sys
import argparse
import shutil
from typing import List

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='在训练集和验证集之间移动数据文件')
    
    # 文件指定方式（互斥组）
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument('--files', nargs='+', help='要移动的文件名列表')
    file_group.add_argument('--file-list', help='包含要移动的文件名的文本文件路径')
    
    # 移动方向
    parser.add_argument('--direction','--d', choices=['val2train', 'train2val'], 
                        required=True, help='移动方向: val2train(从验证集到训练集) 或 train2val(从训练集到验证集)')
    
    # 数据集根目录
    parser.add_argument('--dataset-root', 
                        default='/root/hyg/projects/hushang_fridge_detect/data/hyg_dlg_split_90-10',
                        help='数据集根目录路径')
    
    # 可选的强制移动参数
    parser.add_argument('--force', action='store_true', help='强制覆盖目标位置已存在的文件')
    
    # 测试模式（不实际移动文件）
    parser.add_argument('--dry-run', action='store_true', help='测试模式，不实际移动文件')
    
    return parser.parse_args()

def read_file_list(file_list_path: str) -> List[str]:
    """从文件中读取文件名列表"""
    try:
        with open(file_list_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"错误: 无法读取文件列表 '{file_list_path}': {e}")
        sys.exit(1)

def get_files_to_move(args) -> List[str]:
    """获取要移动的文件列表"""
    if args.files:
        return args.files
    else:
        return read_file_list(args.file_list)

def ensure_directory_exists(directory: str):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def move_file(src_path: str, dst_path: str, force: bool, dry_run: bool):
    """移动文件从源路径到目标路径"""
    # 检查源文件是否存在
    if not os.path.exists(src_path):
        print(f"警告: 源文件不存在: {src_path}")
        return False
    
    # 检查目标文件是否已存在
    if os.path.exists(dst_path):
        if force:
            print(f"警告: 目标文件已存在，将被覆盖: {dst_path}")
        else:
            print(f"错误: 目标文件已存在且未指定 --force 参数: {dst_path}")
            return False
    
    # 确保目标目录存在
    ensure_directory_exists(os.path.dirname(dst_path))
    
    if dry_run:
        print(f"[测试模式] 将移动: {src_path} -> {dst_path}")
    else:
        try:
            shutil.move(src_path, dst_path)
            print(f"已移动: {src_path} -> {dst_path}")
            return True
        except Exception as e:
            print(f"错误: 移动文件时出错: {e}")
            return False

def move_data_files(files: List[str], dataset_root: str, direction: str, force: bool, dry_run: bool):
    """移动数据文件（图片和标签）"""
    # 根据方向确定源目录和目标目录
    if direction == 'val2train':
        src_img_dir = os.path.join(dataset_root, 'val', 'images')
        src_label_dir = os.path.join(dataset_root, 'val', 'labels')
        dst_img_dir = os.path.join(dataset_root, 'train', 'images')
        dst_label_dir = os.path.join(dataset_root, 'train', 'labels')
    else:  # train2val
        src_img_dir = os.path.join(dataset_root, 'train', 'images')
        src_label_dir = os.path.join(dataset_root, 'train', 'labels')
        dst_img_dir = os.path.join(dataset_root, 'val', 'images')
        dst_label_dir = os.path.join(dataset_root, 'val', 'labels')
    
    # 确保目录存在
    ensure_directory_exists(src_img_dir)
    ensure_directory_exists(src_label_dir)
    ensure_directory_exists(dst_img_dir)
    ensure_directory_exists(dst_label_dir)
    
    # 统计信息
    total_files = len(files)
    moved_files = 0
    skipped_files = 0
    
    print(f"开始移动 {total_files} 个文件，方向: {direction}")
    print(f"源图片目录: {src_img_dir}")
    print(f"源标签目录: {src_label_dir}")
    print(f"目标图片目录: {dst_img_dir}")
    print(f"目标标签目录: {dst_label_dir}")
    print("=" * 80)
    
    # 处理每个文件
    for filename in files:
        print(f"处理文件: {filename}")
        
        # 获取图片文件的完整路径
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        img_path = None
        
        # 检查是否带有扩展名
        if any(filename.lower().endswith(ext) for ext in img_extensions):
            # 已经有扩展名
            base_name = filename
            img_path = os.path.join(src_img_dir, filename)
        else:
            # 尝试不同的扩展名
            for ext in img_extensions:
                temp_path = os.path.join(src_img_dir, filename + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    base_name = filename + ext
                    break
        
        if not img_path:
            print(f"警告: 找不到图片文件: {filename} (尝试了多种扩展名)")
            skipped_files += 1
            continue
        
        # 构建标签文件路径（将图片扩展名替换为.txt）
        label_base_name = os.path.splitext(base_name)[0] + '.txt'
        label_path = os.path.join(src_label_dir, label_base_name)
        
        # 移动图片文件
        dst_img_path = os.path.join(dst_img_dir, base_name)
        img_moved = move_file(img_path, dst_img_path, force, dry_run)
        
        # 移动标签文件（如果存在）
        dst_label_path = os.path.join(dst_label_dir, label_base_name)
        label_moved = True
        
        if os.path.exists(label_path):
            label_moved = move_file(label_path, dst_label_path, force, dry_run)
        else:
            print(f"警告: 标签文件不存在: {label_path}")
        
        # 更新统计
        if img_moved and label_moved:
            moved_files += 1
        else:
            skipped_files += 1
        
        print("-" * 50)
    
    # 打印总结
    print("=" * 80)
    print(f"移动完成！")
    print(f"总计文件: {total_files}")
    print(f"成功移动: {moved_files}")
    print(f"跳过: {skipped_files}")
    if dry_run:
        print("注意: 这是测试模式，没有实际移动文件。使用 --dry-run 可以执行实际移动。")

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 获取要移动的文件列表
    files = get_files_to_move(args)
    
    # 执行移动操作
    move_data_files(files, args.dataset_root, args.direction, args.force, args.dry_run)

if __name__ == '__main__':
    main()