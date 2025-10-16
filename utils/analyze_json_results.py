#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON结果分析工具
统计JSON文件中result字段是否为空，并复制非空文件到指定文件夹

支持功能：
1. 统计result字段是否为空
2. 复制非空文件到指定文件夹
3. 查找并统计特定字段内容的出现次数

支持两种JSON格式：
1. predictions.result - 预测结果格式（YOLO预测输出）
2. annotations.result - 标注格式（Label Studio标注文件）

使用案例：
1. 基本分析：python analyze_json_results.py rag_json_1010/
2. 复制非空文件：python analyze_json_results.py rag_json_1010/ -o results/non_empty/
3. 显示详细信息：python analyze_json_results.py rag_json_1010/ --verbose
4. 查找特定字段：python analyze_json_results.py rag_json_1010/ --field "predictions.result.rectanglelabels"

作者: AI助手
版本: 1.2
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import argparse


def get_nested_value(data: Any, path: str) -> List[Any]:
    """
    获取嵌套字典或列表中指定路径的值，支持复杂的嵌套列表和字典结构
    
    Args:
        data: JSON数据（字典或列表）
        path: 字段路径，如 "predictions.result.value.rectanglelabels"
    
    Returns:
        找到的值的列表
    """
    keys = path.split('.')
    results = []
    
    def traverse(obj: Any, keys: List[str], index: int):
        if index >= len(keys):
            if obj is not None:
                results.append(obj)
            return
        
        key = keys[index]
        if isinstance(obj, dict):
            if key in obj:
                traverse(obj[key], keys, index + 1)
        elif isinstance(obj, list):
            for item in obj:
                if item is not None:
                    traverse(item, keys, index)
        
    traverse(data, keys, 0)
    
    # 如果按照原始路径没有找到结果，且路径包含'rectanglelabels'，尝试其他可能的路径
    if not results and 'rectanglelabels' in path:
        # 尝试完整路径 predictions.result.value.rectanglelabels
        if path == 'predictions.result.rectanglelabels':
            results = get_nested_value(data, 'predictions.result.value.rectanglelabels')
        
        # 如果还没找到，尝试更复杂的遍历，查找所有可能的rectanglelabels
        if not results:
            def find_rectanglelabels(obj):
                if isinstance(obj, dict):
                    if 'rectanglelabels' in obj:
                        results.append(obj['rectanglelabels'])
                    for v in obj.values():
                        find_rectanglelabels(v)
                elif isinstance(obj, list):
                    for item in obj:
                        find_rectanglelabels(item)
            find_rectanglelabels(data)
    
    return results


def is_result_empty(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    检查result字段是否为空，并返回检查结果和格式类型
    
    Args:
        data: JSON数据字典
    
    Returns:
        Tuple[是否为空, 格式类型]
    """
    # 检查predictions.result格式（YOLO预测格式）
    if 'predictions' in data and isinstance(data['predictions'], list):
        for prediction in data['predictions']:
            if isinstance(prediction, dict) and 'result' in prediction:
                result = prediction['result']
                if isinstance(result, list) and len(result) > 0:
                    # 检查result列表中的元素是否为空
                    for item in result:
                        if item and isinstance(item, dict):
                            # 检查是否有实际内容
                            if item.get('value') or item.get('confidence') or item.get('rectanglelabels'):
                                return False, "predictions.result"
                elif result and len(result) > 0:
                    return False, "predictions.result"
        return True, "predictions.result"
    
    # 检查annotations.result格式（Label Studio格式）
    if 'annotations' in data and isinstance(data['annotations'], list):
        for annotation in data['annotations']:
            if isinstance(annotation, dict) and 'result' in annotation:
                result = annotation['result']
                if isinstance(result, list) and len(result) > 0:
                    # 检查result列表中的元素是否为空
                    for item in result:
                        if item and isinstance(item, dict):
                            # 检查是否有实际内容
                            if item.get('value') or item.get('id') or item.get('type'):
                                return False, "annotations.result"
                elif result and len(result) > 0:
                    return False, "annotations.result"
        return True, "annotations.result"
    
    # 检查直接result字段
    if 'result' in data:
        result = data['result']
        if isinstance(result, list) and len(result) > 0:
            return False, "direct.result"
        elif result:
            return False, "direct.result"
    
    return True, "unknown"


def analyze_json_folder(json_folder: str, output_folder: str = None, 
                       verbose: bool = False, show_progress: bool = True, 
                       field_path: str = None) -> Dict[str, Any]:
    """
    分析JSON文件夹，统计result字段为空和非空的文件，或查找特定字段内容
    
    Args:
        json_folder: JSON文件所在的文件夹路径
        output_folder: 非空文件复制到的目标文件夹（可选）
        verbose: 是否显示详细信息
        show_progress: 是否显示进度条
        field_path: 要查找的字段路径，如 "predictions.result.rectanglelabels"（可选）
    
    Returns:
        包含统计信息的字典
    """
    json_folder_path = Path(json_folder)
    
    if not json_folder_path.exists():
        raise FileNotFoundError(f"文件夹 {json_folder} 不存在")
    
    # 获取所有JSON文件，包括子文件夹中的文件
    json_files = list(json_folder_path.rglob("*.json"))
    
    if not json_files:
        raise ValueError(f"文件夹 {json_folder} 中没有找到JSON文件")
    
    print(f"📁 找到 {len(json_files)} 个JSON文件")
    
    # 统计信息
    stats = {
        'total_files': len(json_files),
        'empty_count': 0,
        'non_empty_count': 0,
        'error_count': 0,
        'non_empty_files': [],
        'empty_files': [],
        'error_files': [],
        'format_stats': {},
        'file_sizes': {}
    }
    
    # 如果指定了字段路径，初始化字段统计
    if field_path:
        stats['field_stats'] = {}
        stats['field_path'] = field_path
        print(f"🔍 正在查找字段: {field_path}")
    
    # 创建输出文件夹（如果需要）
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📂 输出文件夹: {output_path.absolute()}")
    
    # 分析每个JSON文件
    for i, json_file in enumerate(json_files, 1):
        if show_progress:
            progress = f"[{i}/{len(json_files)}]"
        else:
            progress = ""
        
        try:
            # 获取文件大小
            file_size = json_file.stat().st_size
            stats['file_sizes'][str(json_file)] = file_size
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 如果指定了字段路径，查找并统计字段内容
            if field_path:
                field_values = get_nested_value(data, field_path)
                
                # 处理找到的值并更新统计
                if field_values:
                    for value in field_values:
                        # 处理列表类型的值（如rectanglelabels通常是列表）
                        if isinstance(value, list):
                            for item in value:
                                if item not in stats['field_stats']:
                                    stats['field_stats'][item] = 0
                                stats['field_stats'][item] += 1
                        # 处理字符串类型的值
                        elif isinstance(value, str):
                            if value not in stats['field_stats']:
                                stats['field_stats'][value] = 0
                            stats['field_stats'][value] += 1
                        # 处理其他类型的值
                        else:
                            value_str = str(value)
                            if value_str not in stats['field_stats']:
                                stats['field_stats'][value_str] = 0
                            stats['field_stats'][value_str] += 1
                
                if verbose:
                    if field_values:
                        print(f"{progress} ✅ {json_file.name}: 找到字段值 {field_values}")
                    else:
                        print(f"{progress} ❌ {json_file.name}: 未找到字段值")
                else:
                    print(f"{progress} {'✅' if field_values else '❌'} {json_file.name}")
                
            else:
                # 原始功能：检查result字段是否为空
                is_empty, format_type = is_result_empty(data)
                
                # 更新格式统计
                if format_type not in stats['format_stats']:
                    stats['format_stats'][format_type] = 0
                stats['format_stats'][format_type] += 1
                
                if is_empty:
                    stats['empty_count'] += 1
                    stats['empty_files'].append(str(json_file))
                    if verbose:
                        print(f"{progress} ❌ {json_file.name}: result字段为空 ({format_type})")
                    else:
                        print(f"{progress} ❌ {json_file.name}")
                else:
                    stats['non_empty_count'] += 1
                    stats['non_empty_files'].append(str(json_file))
                    if verbose:
                        print(f"{progress} ✅ {json_file.name}: result字段不为空 ({format_type})")
                    else:
                        print(f"{progress} ✅ {json_file.name}")
                    
                    # 复制非空文件到输出文件夹
                    if output_folder:
                        dest_file = output_path / json_file.name
                        shutil.copy2(json_file, dest_file)
                        if verbose:
                            print(f"    📋 已复制到: {dest_file}")
        
        except json.JSONDecodeError as e:
            stats['error_count'] += 1
            stats['error_files'].append(str(json_file))
            print(f"{progress} ⚠️  {json_file.name}: JSON解析错误 - {e}")
        except Exception as e:
            stats['error_count'] += 1
            stats['error_files'].append(str(json_file))
            print(f"{progress} ⚠️  {json_file.name}: 读取错误 - {e}")
    
    return stats


def print_stats(stats: Dict[str, Any], verbose: bool = False):
    """打印统计结果"""
    print("\n" + "=" * 80)
    print("📊 统计结果")
    print("=" * 80)
    
    total = stats['total_files']
    errors = stats['error_count']
    
    print(f"📁 总文件数: {total}")
    
    # 如果有字段统计
    if 'field_stats' in stats:
        field_path = stats['field_path']
        field_stats = stats['field_stats']
        
        print(f"🔍 字段 '{field_path}' 内容统计:")
        if field_stats:
            # 按出现次数排序
            sorted_stats = sorted(field_stats.items(), key=lambda x: x[1], reverse=True)
            
            for value, count in sorted_stats:
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"   '{value}': {count} 次 ({percentage:.1f}%)")
        else:
            print("   未找到匹配的字段内容")
    else:
        # 原始统计信息
        empty = stats['empty_count']
        non_empty = stats['non_empty_count']
        
        print(f"✅ result字段不为空: {non_empty} ({non_empty/total*100:.1f}%)")
        print(f"❌ result字段为空: {empty} ({empty/total*100:.1f}%)")
    
    print(f"⚠️  解析错误: {errors} ({errors/total*100:.1f}%)")
    
    # 文件格式统计
    if 'format_stats' in stats and stats['format_stats']:
        print("\n📋 文件格式统计:")
        for format_type, count in stats['format_stats'].items():
            print(f"   {format_type}: {count} 个文件")
    
    # 文件大小统计
    if stats['file_sizes']:
        sizes = list(stats['file_sizes'].values())
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        min_size = min(sizes) if sizes else 0
        
        print(f"\n📏 文件大小统计:")
        print(f"   平均大小: {avg_size/1024:.1f} KB")
        print(f"   最大大小: {max_size/1024:.1f} KB")
        print(f"   最小大小: {min_size/1024:.1f} KB")
    
    # 详细文件列表
    if verbose:
        if 'non_empty_files' in stats and stats['non_empty_files']:
            print(f"\n✅ 非空文件列表 ({len(stats['non_empty_files'])} 个):")
            for file_path in stats['non_empty_files'][:10]:  # 只显示前10个
                print(f"   📄 {Path(file_path).name}")
            if len(stats['non_empty_files']) > 10:
                print(f"   ... 还有 {len(stats['non_empty_files']) - 10} 个文件")
        
        if stats['error_files']:
            print(f"\n⚠️  错误文件列表 ({len(stats['error_files'])} 个):")
            for file_path in stats['error_files'][:5]:  # 只显示前5个
                print(f"   ❗ {Path(file_path).name}")
    
    print("=" * 80)


def main():
    """主函数"""
    # 自定义ArgumentParser错误处理
    class CustomArgumentParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help()
            sys.exit(1)
    
    parser = CustomArgumentParser(
        description='JSON结果分析工具 - 统计result字段是否为空并复制非空文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用案例:
  # 基本分析
  python analyze_json_results.py rag_json_1010/
  
  # 复制非空文件到指定文件夹
  python analyze_json_results.py rag_json_1010/ -o results/non_empty/
  
  # 显示详细信息
  python analyze_json_results.py rag_json_1010/ --verbose
  
  # 不显示进度条（处理大量文件时）
  python analyze_json_results.py rag_json_1010/ --no-progress
  
  # 查找特定字段内容并统计
  python analyze_json_results.py rag_json_1010/ --field "predictions.result.rectanglelabels"
  
  # 查找特定字段并显示详细信息
  python analyze_json_results.py rag_json_1010/ --field "predictions.result.rectanglelabels" --verbose
        """)
    
    parser.add_argument('json_folder', help='JSON文件所在的文件夹路径')
    parser.add_argument('-o', '--output', help='非空文件复制到的目标文件夹')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    parser.add_argument('--no-progress', action='store_true', help='不显示进度条')
    parser.add_argument('--field', help='要查找的字段路径，如 "predictions.result.rectanglelabels"')
    parser.add_argument('--version', action='version', version='JSON结果分析工具 v1.2')
    
    args = parser.parse_args()
    
    print("🚀 JSON结果分析工具")
    print("=" * 80)
    print(f"📁 分析文件夹: {args.json_folder}")
    if args.output:
        print(f"📂 输出文件夹: {args.output}")
    print("=" * 80)
    
    try:
        stats = analyze_json_folder(
            json_folder=args.json_folder,
            output_folder=args.output,
            verbose=args.verbose,
            show_progress=not args.no_progress,
            field_path=args.field
        )
        
        print_stats(stats, args.verbose)
        
        # 总结信息
        if args.output and 'non_empty_count' in stats and stats['non_empty_count'] > 0:
            print(f"\n🎉 成功复制 {stats['non_empty_count']} 个非空文件到: {args.output}")
        
        if stats['error_count'] > 0:
            print(f"\n⚠️  注意: 有 {stats['error_count']} 个文件解析失败，请检查这些文件格式")
    
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()