#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理简单格式的CSV文件并下载所有图片
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
        logging.FileHandler('process_simple_csv.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SimpleImageDownloader:
    """简单图片下载器"""
    
    def __init__(self, output_image_dir: str):
        """
        初始化图片下载器
        
        Args:
            output_image_dir: 输出图片目录
        """
        self.output_image_dir = Path(output_image_dir)
        self.output_image_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_count = 0
        self.skipped_count = 0
        self.error_count = 0
    
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
        # 解析URL，提取文件扩展名
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        _, ext = os.path.splitext(path)
        
        # 如果没有扩展名，默认使用.jpg
        if not ext:
            ext = '.jpg'
        
        # 生成基于ID和索引的唯一文件名
        filename = f"id_{row_id}_img_{index}{ext}"
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


def process_csv(input_csv: str, output_image_dir: str):
    """
    处理简单格式的CSV文件并下载所有图片
    
    Args:
        input_csv: 输入CSV文件路径
        output_image_dir: 输出图片目录
    """
    logger.info(f"开始处理CSV文件: {input_csv}")
    logger.info(f"输出图片目录: {output_image_dir}")
    
    downloader = SimpleImageDownloader(output_image_dir)
    
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='处理简单格式的CSV文件并下载所有图片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python process_simple_csv.py -i 台下制冰机不一致.csv -o 输出图片目录
        """
    )
    
    parser.add_argument(
        '-i', '--input-csv',
        required=True,
        help='输入CSV文件路径'
    )
    
    parser.add_argument(
        '-o', '--output-image-dir',
        required=True,
        help='输出图片目录'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细日志信息'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证输入文件是否存在
    if not os.path.exists(args.input_csv):
        logger.error(f"输入文件不存在: {args.input_csv}")
        sys.exit(1)
    
    # 处理CSV文件
    process_csv(args.input_csv, args.output_image_dir)


if __name__ == '__main__':
    main()