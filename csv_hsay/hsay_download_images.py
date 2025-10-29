#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片下载和处理脚本
从CSV文件中提取图片URL，下载图片并合并多图，生成包含本地图片路径的新CSV文件
"""

import argparse
import csv
import json
import hashlib
import os
import sys
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_images.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ImageDownloader:
    """图片下载和处理类"""
    
    def __init__(self, image_dir: str):
        """
        初始化图片下载器
        
        Args:
            image_dir: 图片保存目录
        """
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_count = 0
        self.skipped_count = 0
        self.error_count = 0
    
    def get_filename_from_url(self, url: str) -> str:
        """
        从URL中提取文件名
        
        Args:
            url: 图片URL
            
        Returns:
            文件名
        """
        # 解析URL，提取路径部分
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        
        # 提取文件名（去掉查询参数）
        filename = os.path.basename(path)
        
        # 如果没有扩展名，默认添加.jpg
        if not os.path.splitext(filename)[1]:
            filename += '.jpg'
            
        return filename
    
    def download_image(self, url: str) -> Optional[str]:
        """
        下载单张图片
        
        Args:
            url: 图片URL
            
        Returns:
            本地文件路径，如果下载失败返回None
        """
        try:
            filename = self.get_filename_from_url(url)
            local_path = self.image_dir / filename
            
            # 如果文件已存在，跳过下载
            if local_path.exists():
                logger.debug(f"图片已存在，跳过下载: {filename}")
                self.skipped_count += 1
                return str(local_path)
            
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
            return str(local_path)
            
        except Exception as e:
            logger.error(f"下载图片失败 {url}: {str(e)}")
            self.error_count += 1
            return None
    
    def merge_images(self, image_paths: List[str], output_filename: str) -> Optional[str]:
        """
        合并多张图片为一张
        
        Args:
            image_paths: 图片路径列表
            output_filename: 输出文件名
            
        Returns:
            合并后的图片路径，如果失败返回None
        """
        try:
            if len(image_paths) <= 1:
                return image_paths[0] if image_paths else None
            
            output_path = self.image_dir / output_filename
            
            # 如果合并后的图片已存在，直接返回
            if output_path.exists():
                logger.debug(f"合并图片已存在，跳过合并: {output_filename}")
                return str(output_path)
            
            # 打开所有图片
            images = []
            for path in image_paths:
                if os.path.exists(path):
                    try:
                        img = Image.open(path)
                        # 转换为RGB模式，确保兼容性
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        images.append(img)
                    except Exception as e:
                        logger.warning(f"无法打开图片 {path}: {str(e)}")
                        continue
            
            if not images:
                logger.error("没有有效的图片可以合并")
                return None
            
            # 计算合并后的图片尺寸
            # 水平排列图片
            total_width = sum(img.width for img in images)
            max_height = max(img.height for img in images)
            
            # 创建新图片
            merged_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
            
            # 粘贴图片
            x_offset = 0
            for img in images:
                merged_image.paste(img, (x_offset, 0))
                x_offset += img.width
            
            # 保存合并后的图片
            merged_image.save(output_path, 'JPEG', quality=85)
            
            # 关闭图片对象
            for img in images:
                img.close()
            
            logger.info(f"图片合并成功: {output_filename}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"合并图片失败: {str(e)}")
            return None
    
    def process_row_images(self, json_data: str, row_index: int) -> Optional[str]:
        """
        处理单行的图片数据
        
        Args:
            json_data: 现场结果JSON字符串
            row_index: 行索引
            
        Returns:
            本地图片路径
        """
        try:
            # 解析JSON数据
            if not json_data or json_data.strip() == '':
                return None
                
            data = json.loads(json_data)
            if not isinstance(data, list):
                logger.warning(f"第{row_index}行: JSON数据不是数组格式")
                return None
            
            # 提取所有图片URL
            image_urls = []
            for item in data:
                if isinstance(item, dict) and 'handle' in item:
                    handle = item['handle']
                    if handle and handle.strip():
                        image_urls.append(handle.strip())
            
            if not image_urls:
                logger.warning(f"第{row_index}行: 未找到有效的图片URL")
                return None
            
            # 下载所有图片
            downloaded_paths = []
            for url in image_urls:
                local_path = self.download_image(url)
                if local_path:
                    downloaded_paths.append(local_path)
            
            if not downloaded_paths:
                logger.error(f"第{row_index}行: 所有图片下载失败")
                return None
            
            # 如果只有一张图片，直接返回
            if len(downloaded_paths) == 1:
                return downloaded_paths[0]
            
            # 多张图片需要合并
            # 使用行数据的MD5作为合并图片的文件名
            content_for_hash = json_data + str(row_index)
            hash_md5 = hashlib.md5(content_for_hash.encode('utf-8')).hexdigest()
            merged_filename = f"merged_{hash_md5}.jpg"
            
            merged_path = self.merge_images(downloaded_paths, merged_filename)
            return merged_path
            
        except json.JSONDecodeError as e:
            logger.error(f"第{row_index}行: JSON解析失败: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"第{row_index}行: 处理图片失败: {str(e)}")
            return None


def process_csv(input_csv: str, output_csv: str, image_dir: str):
    """
    处理CSV文件，下载图片并生成新的CSV文件
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        image_dir: 图片保存目录
    """
    logger.info(f"开始处理CSV文件: {input_csv}")
    logger.info(f"图片保存目录: {image_dir}")
    logger.info(f"输出CSV文件: {output_csv}")
    
    downloader = ImageDownloader(image_dir)
    
    try:
        with open(input_csv, 'r', encoding='utf-8') as infile, \
             open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # 读取并写入表头
            headers = next(reader)
            headers.append('本地图片路径')
            writer.writerow(headers)
            
            # 找到"现场结果"列的索引
            try:
                result_column_index = headers.index('现场结果')
            except ValueError:
                logger.error("未找到'现场结果'列")
                return
            
            # 处理每一行数据
            row_count = 0
            for row_index, row in enumerate(reader, start=2):  # 从第2行开始（第1行是表头）
                row_count += 1
                
                if row_count % 100 == 0:
                    logger.info(f"已处理 {row_count} 行数据")
                
                # 获取现场结果数据
                if len(row) > result_column_index:
                    json_data = row[result_column_index]
                    local_image_path = downloader.process_row_images(json_data, row_index)
                else:
                    local_image_path = None
                
                # 添加本地图片路径列
                row.append(local_image_path or '')
                writer.writerow(row)
    
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
    logger.info(f"成功下载图片: {downloader.downloaded_count}")
    logger.info(f"跳过已存在图片: {downloader.skipped_count}")
    logger.info(f"下载失败图片: {downloader.error_count}")
    logger.info(f"输出文件: {output_csv}")
    logger.info("=" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='从CSV文件下载图片并生成包含本地路径的新CSV文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python download_images.py
  python download_images.py -i input.csv -o output.csv -d ./images
  python download_images.py --input-csv data.csv --output-csv result.csv --image-dir /path/to/images
        """
    )
    
    parser.add_argument(
        '-i', '--input-csv',
        default='/root/zhifang/projects/snowman_review/data/处理后的沪上阿姨数据.csv',
        help='输入CSV文件路径 (默认: /root/zhifang/projects/snowman_review/data/处理后的沪上阿姨数据.csv)'
    )
    
    parser.add_argument(
        '-o', '--output-csv',
        required=True,
        help='输出CSV文件路径'
    )
    
    parser.add_argument(
        '-d', '--image-dir',
        required=True,
        help='图片保存目录'
    )
    
    parser.add_argument(
        '-v', '--verbose',
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
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理CSV文件
    process_csv(args.input_csv, args.output_csv, args.image_dir)


if __name__ == '__main__':
    main()