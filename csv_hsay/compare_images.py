#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计两个目录中相同的图片文件，并找出只存在于第一个目录而不在第二个目录中的文件
相同条件：文件名相互包含或相等，文件内容大小一致，或MD5值相同
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple


def calculate_md5(file_path: str) -> str:
    """
    计算文件的MD5值
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件的MD5值
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"计算文件MD5时出错 {file_path}: {e}")
        return None


def get_file_info(directory: str, calculate_md5_flag: bool = True) -> Dict[str, Dict[str, any]]:
    """
    获取目录中所有图片文件的信息
    
    Args:
        directory: 目录路径
        calculate_md5_flag: 是否计算MD5值
        
    Returns:
        字典，键为文件名，值为包含文件信息的字典
    """
    file_info = {}
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"警告: 目录不存在: {directory}")
        return file_info
    
    if not directory_path.is_dir():
        print(f"警告: 不是一个目录: {directory}")
        return file_info
    
    total_files = len(list(directory_path.iterdir()))
    processed = 0
    
    # 遍历目录中的所有文件
    for file_path in directory_path.iterdir():
        if file_path.is_file():
            processed += 1
            # 对于大量文件，显示进度
            if total_files > 100 and processed % 20 == 0:
                print(f"处理中... {processed}/{total_files} 文件")
            
            # 获取文件名和扩展名
            filename = file_path.name
            # 获取文件大小
            file_size = file_path.stat().st_size
            
            # 存储文件信息
            file_info[filename] = {
                'size': file_size,
                'path': str(file_path)
            }
            
            # 计算MD5值
            if calculate_md5_flag:
                md5 = calculate_md5(str(file_path))
                file_info[filename]['md5'] = md5
    
    return file_info


def compare_files(dir1_files: Dict[str, Dict[str, any]], dir2_files: Dict[str, Dict[str, any]], use_md5: bool = True) -> Tuple[List[Tuple[str, str]], Dict[str, List[str]], List[str]]:
    """
    比较两个目录中的文件，找出相同的文件和只存在于第一个目录的文件
    
    Args:
        dir1_files: 第一个目录的文件信息
        dir2_files: 第二个目录的文件信息
        use_md5: 是否使用MD5进行比对
        
    Returns:
        元组：(完全匹配的文件对列表, 基于条件匹配的文件字典, 只存在于第一个目录的文件列表)
    """
    # 存储匹配的文件
    matches = []
    # 存储匹配规则
    match_rules = {
        'name_equal': [],      # 文件名完全相等
        'name_contains_1to2': [],  # dir1文件名包含在dir2中
        'name_contains_2to1': [],  # dir2文件名包含在dir1中
        'size_equal': [],      # 文件大小相等
        'md5_equal': []        # MD5值相等
    }
    
    # 存储已匹配的dir1文件
    matched_dir1_files = set()
    
    # 如果使用MD5，先建立MD5到文件的映射
    dir2_md5_map = {}
    if use_md5:
        for filename, info in dir2_files.items():
            if 'md5' in info and info['md5']:
                dir2_md5_map[info['md5']] = filename
    
    # 遍历第一个目录的所有文件
    for filename1, info1 in dir1_files.items():
        # 检查文件名完全相等的情况
        if filename1 in dir2_files:
            info2 = dir2_files[filename1]
            matches.append((filename1, filename1))
            match_rules['name_equal'].append(filename1)
            matched_dir1_files.add(filename1)
            continue
        
        # 检查文件名相互包含的情况
        found_match = False
        for filename2, info2 in dir2_files.items():
            # 情况1: dir1文件名包含在dir2中
            if filename1 in filename2:
                matches.append((filename1, filename2))
                match_rules['name_contains_1to2'].append(f"{filename1} -> {filename2}")
                matched_dir1_files.add(filename1)
                found_match = True
                break
            # 情况2: dir2文件名包含在dir1中
            elif filename2 in filename1:
                matches.append((filename1, filename2))
                match_rules['name_contains_2to1'].append(f"{filename1} <- {filename2}")
                matched_dir1_files.add(filename1)
                found_match = True
                break
        
        if found_match:
            continue
        
        # 检查MD5值相等的情况
        if use_md5 and 'md5' in info1 and info1['md5'] and info1['md5'] in dir2_md5_map:
            filename2 = dir2_md5_map[info1['md5']]
            matches.append((filename1, filename2))
            match_rules['md5_equal'].append(f"{filename1} <-> {filename2} (MD5: {info1['md5'][:8]}...)")
            matched_dir1_files.add(filename1)
            continue
        
        # 检查文件大小相等的情况（仅在MD5不匹配或未启用时使用）
        for filename2, info2 in dir2_files.items():
            if info1['size'] == info2['size']:
                matches.append((filename1, filename2))
                match_rules['size_equal'].append(f"{filename1} <-> {filename2} (大小: {info1['size']} bytes)")
                matched_dir1_files.add(filename1)
                break
    
    # 找出只存在于第一个目录的文件
    only_in_dir1 = [f for f in dir1_files.keys() if f not in matched_dir1_files]
    
    return matches, match_rules, only_in_dir1


def main():
    """主函数"""
    # 定义两个目录路径
    dir1 = "/root/hyg/projects/hushang_fridge_detect/csv_hsay/txzbj_dif"
    dir2 = "/root/hyg/projects/hushang_fridge_detect/data/hyg_txzbj/hyg_txzbj_1027/images"
    
    print("开始统计两个目录中相同的图片文件...")
    print(f"目录1: {dir1}")
    print(f"目录2: {dir2}")
    print("=" * 80)
    
    # 获取两个目录的文件信息
    print("正在获取目录1的文件信息...")
    dir1_files = get_file_info(dir1)
    print(f"目录1包含 {len(dir1_files)} 个文件")
    
    print("正在获取目录2的文件信息...")
    dir2_files = get_file_info(dir2)
    print(f"目录2包含 {len(dir2_files)} 个文件")
    print("=" * 80)
    
    # 比较文件
    print("正在比较文件...")
    matches, match_rules, only_in_dir1 = compare_files(dir1_files, dir2_files, use_md5=True)
    
    # 输出结果
    print(f"找到 {len(matches)} 个相同的文件")
    print("=" * 80)
    
    # 按匹配规则输出详细信息
    print("匹配规则统计:")
    print(f"1. 文件名完全相等: {len(match_rules['name_equal'])} 个")
    for i, match in enumerate(match_rules['name_equal'], 1):
        print(f"   - {i:3d}. {match}")
        if i >= 10 and len(match_rules['name_equal']) > 10:
            print(f"   - ... 还有 {len(match_rules['name_equal']) - 10} 个")
            break
    
    print(f"\n2. 目录1文件名包含在目录2中: {len(match_rules['name_contains_1to2'])} 个")
    for i, match in enumerate(match_rules['name_contains_1to2'], 1):
        print(f"   - {i:3d}. {match}")
        if i >= 10 and len(match_rules['name_contains_1to2']) > 10:
            print(f"   - ... 还有 {len(match_rules['name_contains_1to2']) - 10} 个")
            break
    
    print(f"\n3. 目录2文件名包含在目录1中: {len(match_rules['name_contains_2to1'])} 个")
    for i, match in enumerate(match_rules['name_contains_2to1'], 1):
        print(f"   - {i:3d}. {match}")
        if i >= 10 and len(match_rules['name_contains_2to1']) > 10:
            print(f"   - ... 还有 {len(match_rules['name_contains_2to1']) - 10} 个")
            break
    
    print(f"\n4. MD5值相同: {len(match_rules['md5_equal'])} 个")
    for i, match in enumerate(match_rules['md5_equal'], 1):
        print(f"   - {i:3d}. {match}")
        if i >= 10 and len(match_rules['md5_equal']) > 10:
            print(f"   - ... 还有 {len(match_rules['md5_equal']) - 10} 个")
            break
    
    print(f"\n5. 文件大小相等: {len(match_rules['size_equal'])} 个")
    for i, match in enumerate(match_rules['size_equal'], 1):
        print(f"   - {i:3d}. {match}")
        if i >= 10 and len(match_rules['size_equal']) > 10:
            print(f"   - ... 还有 {len(match_rules['size_equal']) - 10} 个")
            break
    
    print("=" * 80)
    # 输出只存在于第一个目录的文件
    print(f"只存在于目录1而不在目录2中的文件: {len(only_in_dir1)} 个")
    for i, filename in enumerate(only_in_dir1, 1):
        print(f"   - {i:3d}. {filename}")
        if i >= 50 and len(only_in_dir1) > 50:
            print(f"   - ... 还有 {len(only_in_dir1) - 50} 个")
            break
    
    print("=" * 80)
    print("统计完成！")


if __name__ == "__main__":
    main()