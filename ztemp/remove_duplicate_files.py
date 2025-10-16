#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除指定文件夹中的重复文件脚本
功能：遍历指定目录及其子目录，计算文件哈希值，删除内容相同的重复文件
用法：python remove_duplicate_files.py <文件夹路径> [--dry-run] [--follow-links]
"""

import os
import hashlib
import argparse
from collections import defaultdict
import time


def calculate_file_hash(file_path, block_size=65536):
    """计算文件的MD5哈希值"""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as file:
            buf = file.read(block_size)
            while buf:
                hasher.update(buf)
                buf = file.read(block_size)
        return hasher.hexdigest()
    except Exception as e:
        print(f"警告: 无法读取文件 {file_path}，错误: {e}")
        return None


def get_all_files(directory, follow_links=False):
    """获取目录下所有文件的路径"""
    file_paths = []
    try:
        for root, _, files in os.walk(directory, followlinks=follow_links):
            for file in files:
                file_path = os.path.join(root, file)
                # 确保是文件而不是符号链接（除非指定跟随链接）
                if follow_links or not os.path.islink(file_path):
                    file_paths.append(file_path)
    except Exception as e:
        print(f"错误: 遍历目录时出错: {e}")
    return file_paths


def remove_duplicate_files(directory, dry_run=False, follow_links=False):
    """删除目录中的重复文件"""
    if not os.path.exists(directory):
        print(f"错误: 目录 {directory} 不存在")
        return

    if not os.path.isdir(directory):
        print(f"错误: {directory} 不是一个目录")
        return

    print(f"开始扫描目录: {directory}")
    start_time = time.time()

    # 获取所有文件
    all_files = get_all_files(directory, follow_links)
    total_files = len(all_files)
    print(f"找到 {total_files} 个文件")

    if total_files == 0:
        print("没有找到文件，退出")
        return

    # 按文件大小分组，减少需要计算哈希值的文件数量
    files_by_size = defaultdict(list)
    for i, file_path in enumerate(all_files, 1):
        try:
            file_size = os.path.getsize(file_path)
            files_by_size[file_size].append(file_path)
            if i % 1000 == 0 or i == total_files:
                print(f"已扫描 {i}/{total_files} 个文件")
        except Exception as e:
            print(f"警告: 无法获取文件大小 {file_path}，错误: {e}")

    print(f"按文件大小分组完成，共有 {len(files_by_size)} 个不同的文件大小组")

    # 计算每个文件大小组中文件的哈希值
    files_by_hash = defaultdict(list)
    duplicate_count = 0
    processed_files = 0
    for size, files in files_by_size.items():
        # 只有一个文件的组不可能有重复
        if len(files) <= 1:
            processed_files += len(files)
            continue

        for file_path in files:
            file_hash = calculate_file_hash(file_path)
            if file_hash:
                files_by_hash[file_hash].append(file_path)
            processed_files += 1
            
            if processed_files % 1000 == 0 or processed_files == total_files:
                print(f"已处理 {processed_files}/{total_files} 个文件")

    print(f"哈希值计算完成，找到 {len(files_by_hash)} 个不同的文件哈希值")

    # 找出并删除重复文件
    deleted_count = 0
    saved_space = 0
    for file_hash, file_paths in files_by_hash.items():
        # 如果有多个文件具有相同的哈希值，则它们是重复的
        if len(file_paths) > 1:
            # 保留第一个文件，删除其余的
            # 按文件路径排序，确保结果可预测
            file_paths.sort()
            keep_file = file_paths[0]
            duplicate_files = file_paths[1:]
            
            duplicate_count += len(duplicate_files)
            
            print(f"\n找到 {len(duplicate_files)} 个重复文件 (哈希值: {file_hash}):")
            print(f"  保留: {keep_file}")
            
            for dup_file in duplicate_files:
                try:
                    file_size = os.path.getsize(dup_file)
                    saved_space += file_size
                    
                    if dry_run:
                        print(f"  模拟删除: {dup_file} ({file_size/1024/1024:.2f} MB)")
                    else:
                        os.remove(dup_file)
                        print(f"  删除: {dup_file} ({file_size/1024/1024:.2f} MB)")
                        deleted_count += 1
                except Exception as e:
                    print(f"  错误: 无法删除 {dup_file}，错误: {e}")

    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n扫描完成，耗时: {duration:.2f} 秒")
    print(f"找到 {duplicate_count} 个重复文件")
    
    if dry_run:
        print(f"模拟操作: 未实际删除任何文件")
        print(f"如果执行实际删除，预计将释放 {saved_space/1024/1024/1024:.2f} GB 空间")
    else:
        print(f"已成功删除 {deleted_count} 个重复文件")
        print(f"已释放 {saved_space/1024/1024/1024:.2f} GB 空间")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="删除指定文件夹中的重复文件")
    parser.add_argument("directory", help="要扫描的文件夹路径")
    parser.add_argument("--dry-run", action="store_true", help="仅显示要删除的文件，不实际删除")
    parser.add_argument("--follow-links", action="store_true", help="跟随符号链接")
    
    args = parser.parse_args()
    remove_duplicate_files(args.directory, args.dry_run, args.follow_links)