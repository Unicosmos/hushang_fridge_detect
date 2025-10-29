#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理简单格式的CSV文件并下载所有图片
此脚本合并了原bash和Python脚本的功能
适用于格式：ID,JSON图片数据
"""

import argparse
import csv
import json
import os
import sys
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_all_images.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SimpleImageDownloader:
    """简单图片下载器"""
    
    def __init__(self, output_image_dir: str, use_original_filename: bool = False):
        """
        初始化图片下载器
        
        Args:
            output_image_dir: 输出图片目录
            use_original_filename: 是否使用URL中的原始文件名
        """
        self.output_image_dir = Path(output_image_dir)
        self.output_image_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.use_original_filename = use_original_filename
    
    def get_filename_from_url(self, url: str, row_id: str, index: int) -> str:
        """
        从URL生成唯一文件名
        
        Args:
            url: 图片URL
            row_id: 行ID
            index: 图片在该行的索引
            
        Returns:
            生成的文件名
        """
        # 如果设置使用原始文件名
        if self.use_original_filename:
            # 解析URL，提取原始文件名
            parsed_url = urllib.parse.urlparse(url)
            path = parsed_url.path
            original_filename = os.path.basename(path)
            
            # 如果URL中包含有效的文件名
            if original_filename and '.' in original_filename:
                # 添加ID前缀以避免文件名冲突
                name_without_ext, ext = os.path.splitext(original_filename)
                # 对原始文件名进行清理，移除可能的特殊字符
                safe_filename = f"id_{row_id}_{name_without_ext}{ext}"
                logger.debug(f"使用清理后的原始文件名: {safe_filename}")
                return safe_filename
        
        # 默认命名方式
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        _, ext = os.path.splitext(path)
        
        # 如果没有扩展名，默认使用.jpg
        if not ext:
            ext = '.jpg'
        
        # 生成基于ID的唯一文件名
            filename = f"id_{row_id}{ext}"
        return filename
    
    def download_image(self, url: str, row_id: str, index: int) -> Optional[str]:
        """
        下载图片到输出目录
        
        Args:
            url: 图片URL
            row_id: 行ID
            index: 图片在该行的索引
            
        Returns:
            本地文件路径，如果下载失败返回None
        """
        try:
            filename = self.get_filename_from_url(url, row_id, index)
            local_path = self.output_image_dir / filename
            
            # 如果文件已存在，跳过下载
            if local_path.exists():
                logger.debug(f"图片已存在，跳过下载: {filename}")
                self.skipped_count += 1
                return str(local_path.absolute())
            
            # 下载图片
            logger.info(f"正在下载图片: {url}")
            
            # 设置请求头，模拟浏览器访问
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(local_path, 'wb') as f:
                    f.write(response.read())
            
            self.downloaded_count += 1
            logger.info(f"图片下载成功: {filename}")
            return str(local_path.absolute())
            
        except Exception as e:
            logger.error(f"下载图片失败 {url}: {str(e)}")
            self.error_count += 1
            return None
    
    def process_row_images(self, json_data: str, row_id: str) -> List[str]:
        """
        处理单行的图片数据
        
        Args:
            json_data: JSON字符串（包含图片URLs）
            row_id: 行ID
            
        Returns:
            本地图片路径列表
        """
        try:
            # 解析JSON数据
            if not json_data or json_data.strip() == '':
                return []
                
            data = json.loads(json_data)
            if not isinstance(data, list):
                logger.warning(f"ID {row_id}: JSON数据不是数组格式")
                return []
            
            # 提取所有图片URL
            image_urls = []
            for item in data:
                if isinstance(item, dict) and 'handle' in item:
                    handle = item['handle']
                    if handle and handle.strip():
                        image_urls.append(handle.strip())
            
            if not image_urls:
                logger.warning(f"ID {row_id}: 未找到有效的图片URL")
                return []
            
            # 处理所有图片
            processed_paths = []
            for index, url in enumerate(image_urls):
                local_path = self.download_image(url, row_id, index)
                if local_path:
                    processed_paths.append(local_path)
            
            return processed_paths
            
        except json.JSONDecodeError as e:
            logger.error(f"ID {row_id}: JSON解析失败: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"ID {row_id}: 处理图片失败: {str(e)}")
            return []


def process_csv(input_csv: str, output_image_dir: str, use_original_filename: bool = False):
    """
    处理简单格式的CSV文件并下载所有图片
    
    Args:
        input_csv: 输入CSV文件路径
        output_image_dir: 输出图片目录
        use_original_filename: 是否使用URL中的原始文件名
    """
    logger.info(f"开始处理CSV文件: {input_csv}")
    logger.info(f"输出图片目录: {output_image_dir}")
    logger.info(f"使用原始文件名: {'是' if use_original_filename else '否'}")
    
    downloader = SimpleImageDownloader(output_image_dir, use_original_filename)
    
    try:
        with open(input_csv, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            
            # 处理每一行数据（没有表头）
            row_count = 0
            processed_count = 0
            
            for row in reader:
                row_count += 1
                
                if row_count % 10 == 0:
                    logger.info(f"已处理 {row_count} 行数据")
                
                # 确保行至少有两列
                if len(row) < 2:
                    logger.warning(f"第{row_count}行: 数据格式不正确，跳过")
                    continue
                
                # 获取ID和JSON数据
                row_id = row[0].strip()
                json_data = row[1].strip()
                
                # 处理图片
                processed_paths = downloader.process_row_images(json_data, row_id)
                if processed_paths:
                    processed_count += 1
                    logger.info(f"ID {row_id}: 成功处理 {len(processed_paths)} 张图片")
    
    except FileNotFoundError:
        logger.error(f"输入文件不存在: {input_csv}")
        return
    except Exception as e:
        logger.error(f"处理CSV文件时发生错误: {str(e)}")
        return
    
    # 输出统计信息
    logger.info("=" * 50)
    logger.info("处理完成！统计信息:")
    logger.info(f"总处理行数: {row_count}")
    logger.info(f"成功处理行: {processed_count}")
    logger.info(f"成功下载图片: {downloader.downloaded_count}")
    logger.info(f"跳过已存在图片: {downloader.skipped_count}")
    logger.info(f"处理失败图片: {downloader.error_count}")
    logger.info("=" * 50)


def show_help():
    """显示帮助信息"""
    help_text = """
用法: python3 download_all_images.py [CSV文件路径] [选项]

参数:
  CSV文件路径             可选参数，CSV文件路径（默认: 台下制冰机不一致.csv）

选项:
  -h, --help             显示此帮助信息
  -v, --verbose          显示详细日志信息
  -o, --output-dir       指定输出目录路径
  -n, --original-filename    使用URL中的原始文件名（带ID前缀避免冲突）

示例:
  python3 download_all_images.py
  python3 download_all_images.py 台下制冰机不一致.csv
  python3 download_all_images.py /path/to/your/file.csv --verbose
  python3 download_all_images.py -o /custom/output/directory
  python3 download_all_images.py --original-filename
  python3 download_all_images.py -o /custom/output --original-filename
    """
    print(help_text)


def main():
    """
    主函数 - 合并了原bash脚本的功能，提供命令行参数处理和默认值设置
    """
    # 获取脚本所在目录和当前工作目录
    script_dir = Path(__file__).parent.resolve()
    invocation_dir = Path.cwd()
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='处理简单格式的CSV文件并下载所有图片',
        add_help=False  # 禁用默认的帮助信息，使用自定义的
    )
    
    parser.add_argument(
        'csv_file',  # 位置参数
        nargs='?',   # 可选的位置参数
        help='输入CSV文件路径'
    )
    
    parser.add_argument(
        '-h', '--help',
        action='store_true',
        help='显示帮助信息'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细日志信息'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        help='指定输出目录路径'
    )
    
    parser.add_argument(
        '--original-filename','-n',
        action='store_true',
        help='使用URL中的原始文件名（带ID前缀避免冲突）'
    )
    
    args = parser.parse_args()
    
    # 检查是否显示帮助
    if args.help:
        show_help()
        return
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 设置CSV文件路径（默认值或用户提供）
    if args.csv_file:
        input_csv = args.csv_file
        # 如果用户提供的是相对路径，转换为绝对路径
        if not Path(input_csv).is_absolute():
            input_csv = (invocation_dir / input_csv).resolve()
    else:
        input_csv = script_dir / '台下制冰机不一致.csv'
    
    # 转换为绝对路径并规范化
    input_csv = Path(input_csv).resolve()
    
    # 检查输入文件是否存在
    if not input_csv.exists():
        logger.error(f"错误: 输入CSV文件不存在: {input_csv}")
        sys.exit(1)
    
    # 设置输出目录
    if args.output_dir:
        # 用户指定的输出目录
        output_image_dir = args.output_dir
        # 如果用户提供的是相对路径，转换为绝对路径
        if not Path(output_image_dir).is_absolute():
            output_image_dir = (invocation_dir / output_image_dir).resolve()
    else:
        # 默认输出目录（CSV文件名+_images后缀）
        csv_base_name = input_csv.stem
        output_image_dir = invocation_dir / f"{csv_base_name}_images"
    
    # 转换为绝对路径并规范化
    output_image_dir = Path(output_image_dir).resolve()
    
    # 打印开始信息
    print("=" * 50)
    print("开始下载所有图片")
    print("=" * 50)
    print(f"输入CSV: {input_csv}")
    print(f"输出图片目录: {output_image_dir}")
    print(f"使用原始文件名: {'是' if args.original_filename else '否'}")
    print("=" * 50)
    
    # 处理CSV文件
    process_csv(str(input_csv), str(output_image_dir), args.original_filename)
    
    # 打印完成信息
    print("=" * 50)
    print("处理完成！")
    print(f"所有图片已下载到: {output_image_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()