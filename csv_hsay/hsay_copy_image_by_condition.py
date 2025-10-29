#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据条件过滤CSV数据并处理图片的脚本
从CSV文件中根据指定条件过滤数据，处理现场结果中的图片URL，复制或下载图片到指定目录
"""

import argparse
import csv
import json
import hashlib
import os
import sys
import urllib.request
import urllib.parse
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hsay_copy_image_by_condition.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """图片处理类"""
    
    def __init__(self, source_image_dir: str, output_image_dir: str):
        """
        初始化图片处理器
        
        Args:
            source_image_dir: 源图片目录（可为空，表示从URL下载）
            output_image_dir: 输出图片目录
        """
        self.source_image_dir = Path(source_image_dir) if source_image_dir else None
        self.output_image_dir = Path(output_image_dir)
        self.output_image_dir.mkdir(parents=True, exist_ok=True)
        self.copied_count = 0
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
    
    def find_local_image(self, url: str) -> Optional[str]:
        """
        在本地图片目录中查找对应的图片文件
        
        Args:
            url: 图片URL
            
        Returns:
            本地图片路径，如果未找到返回None
        """
        if not self.source_image_dir or not self.source_image_dir.exists():
            return None
        
        filename = self.get_filename_from_url(url)
        local_path = self.source_image_dir / filename
        
        if local_path.exists():
            return str(local_path)
        
        # 尝试不同的扩展名
        base_name = os.path.splitext(filename)[0]
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            test_path = self.source_image_dir / f"{base_name}{ext}"
            if test_path.exists():
                return str(test_path)
        
        return None
    
    def copy_image(self, source_path: str, url: str) -> Optional[str]:
        """
        复制图片到输出目录
        
        Args:
            source_path: 源图片路径
            url: 原始URL（用于生成目标文件名）
            
        Returns:
            目标图片路径，如果复制失败返回None
        """
        try:
            filename = self.get_filename_from_url(url)
            target_path = self.output_image_dir / filename
            
            # 如果目标文件已存在，跳过复制
            if target_path.exists():
                logger.debug(f"图片已存在，跳过复制: {filename}")
                self.skipped_count += 1
                return str(target_path.absolute())
            
            # 复制图片
            logger.info(f"正在复制图片: {source_path} -> {target_path}")
            shutil.copy2(source_path, target_path)
            
            self.copied_count += 1
            logger.info(f"图片复制成功: {filename}")
            return str(target_path.absolute())
            
        except Exception as e:
            logger.error(f"复制图片失败 {source_path}: {str(e)}")
            self.error_count += 1
            return None
    
    def download_image(self, url: str) -> Optional[str]:
        """
        下载图片到输出目录
        
        Args:
            url: 图片URL
            
        Returns:
            本地文件路径，如果下载失败返回None
        """
        try:
            filename = self.get_filename_from_url(url)
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
    
    def process_image_url(self, url: str) -> Optional[str]:
        """
        处理单个图片URL
        
        Args:
            url: 图片URL
            
        Returns:
            本地图片路径
        """
        if not url or not url.strip():
            return None
        
        url = url.strip()
        
        # 如果有源图片目录，先尝试从本地复制
        if self.source_image_dir:
            local_path = self.find_local_image(url)
            if local_path:
                return self.copy_image(local_path, url)
        
        # 如果本地没有找到，则下载
        return self.download_image(url)
    
    def process_row_images(self, json_data: str, row_index: int) -> List[str]:
        """
        处理单行的图片数据
        
        Args:
            json_data: 现场结果JSON字符串
            row_index: 行索引
            
        Returns:
            本地图片路径列表
        """
        try:
            # 解析JSON数据
            if not json_data or json_data.strip() == '':
                return []
                
            data = json.loads(json_data)
            if not isinstance(data, list):
                logger.warning(f"第{row_index}行: JSON数据不是数组格式")
                return []
            
            # 提取所有图片URL
            image_urls = []
            for item in data:
                if isinstance(item, dict) and 'handle' in item:
                    handle = item['handle']
                    if handle and handle.strip():
                        image_urls.append(handle.strip())
            
            if not image_urls:
                logger.warning(f"第{row_index}行: 未找到有效的图片URL")
                return []
            
            # 处理所有图片
            processed_paths = []
            for url in image_urls:
                local_path = self.process_image_url(url)
                if local_path:
                    processed_paths.append(local_path)
            
            return processed_paths
            
        except json.JSONDecodeError as e:
            logger.error(f"第{row_index}行: JSON解析失败: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"第{row_index}行: 处理图片失败: {str(e)}")
            return []


class CSVFilter:
    """CSV数据过滤器"""
    
    @staticmethod
    def filter_eq(value: Any, filter_value: str) -> bool:
        """
        相等过滤器
        
        Args:
            value: 要比较的值
            filter_value: 过滤值
            
        Returns:
            是否匹配
        """
        return str(value).strip() == filter_value.strip()
    
    @staticmethod
    def apply_filter(row: List[str], headers: List[str], filter_column: str, 
                    filter_op: str, filter_value: str) -> bool:
        """
        应用过滤条件
        
        Args:
            row: 数据行
            headers: 表头
            filter_column: 过滤列名
            filter_op: 过滤操作符
            filter_value: 过滤值
            
        Returns:
            是否通过过滤
        """
        try:
            # 找到过滤列的索引
            if filter_column not in headers:
                logger.error(f"未找到过滤列: {filter_column}")
                return False
            
            column_index = headers.index(filter_column)
            
            # 获取该行对应列的值
            if len(row) <= column_index:
                return False
            
            cell_value = row[column_index]
            
            # 应用过滤操作
            if filter_op == 'eq':
                return CSVFilter.filter_eq(cell_value, filter_value)
            else:
                logger.error(f"不支持的过滤操作符: {filter_op}")
                return False
                
        except Exception as e:
            logger.error(f"应用过滤条件时发生错误: {str(e)}")
            return False


def process_csv(input_csv: str, output_csv: str, source_image_dir: str, 
                output_image_dir: str, filter_column: str, filter_op: str, 
                filter_value: str):
    """
    处理CSV文件，根据条件过滤数据并处理图片
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        source_image_dir: 源图片目录
        output_image_dir: 输出图片目录
        filter_column: 过滤列名
        filter_op: 过滤操作符
        filter_value: 过滤值
    """
    logger.info(f"开始处理CSV文件: {input_csv}")
    logger.info(f"过滤条件: {filter_column} {filter_op} {filter_value}")
    logger.info(f"源图片目录: {source_image_dir or '无（将从URL下载）'}")
    logger.info(f"输出图片目录: {output_image_dir}")
    logger.info(f"输出CSV文件: {output_csv}")
    
    processor = ImageProcessor(source_image_dir, output_image_dir)
    
    try:
        with open(input_csv, 'r', encoding='utf-8') as infile, \
             open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # 读取并写入表头
            headers = next(reader)
            headers.append('本地图片地址')
            writer.writerow(headers)
            
            # 找到"现场结果"列的索引
            try:
                result_column_index = headers.index('现场结果')
            except ValueError:
                logger.error("未找到'现场结果'列")
                return
            
            # 处理每一行数据
            row_count = 0
            filtered_count = 0
            
            for row_index, row in enumerate(reader, start=2):  # 从第2行开始（第1行是表头）
                row_count += 1
                
                if row_count % 100 == 0:
                    logger.info(f"已处理 {row_count} 行数据，过滤后保留 {filtered_count} 行")
                
                # 应用过滤条件
                if not CSVFilter.apply_filter(row, headers[:-1], filter_column, filter_op, filter_value):
                    continue
                
                filtered_count += 1
                
                # 获取现场结果数据并处理图片
                local_image_paths = []
                if len(row) > result_column_index:
                    json_data = row[result_column_index]
                    processed_paths = processor.process_row_images(json_data, row_index)
                    local_image_paths = processed_paths
                
                # 添加本地图片地址列（多个路径用分号分隔）
                image_paths_str = ';'.join(local_image_paths) if local_image_paths else ''
                row.append(image_paths_str)
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
    logger.info(f"过滤后保留行数: {filtered_count}")
    logger.info(f"成功复制图片: {processor.copied_count}")
    logger.info(f"成功下载图片: {processor.downloaded_count}")
    logger.info(f"跳过已存在图片: {processor.skipped_count}")
    logger.info(f"处理失败图片: {processor.error_count}")
    logger.info(f"输出文件: {output_csv}")
    logger.info("=" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='根据条件过滤CSV数据并处理图片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 从本地图片目录复制图片
  python hsay_copy_image_by_condition.py -i input.csv -o output.csv \\
    -s ./source_images -d ./output_images \\
    -c "店铺名称" -op eq -v "某某店"
  
  # 从URL下载图片（不指定源图片目录）
  python hsay_copy_image_by_condition.py -i input.csv -o output.csv \\
    -d ./output_images -c "状态" -op eq -v "已完成"
        """
    )
    
    parser.add_argument(
        '-i', '--input-csv',
        required=True,
        help='输入CSV文件路径'
    )
    
    parser.add_argument(
        '-o', '--output-csv',
        required=True,
        help='输出CSV文件路径'
    )
    
    parser.add_argument(
        '-s', '--source-image-dir',
        default='',
        help='源图片目录（为空时从URL下载图片）'
    )
    
    parser.add_argument(
        '-d', '--output-image-dir',
        required=True,
        help='输出图片目录'
    )
    
    parser.add_argument(
        '-c', '--filter-column',
        required=True,
        help='过滤列名'
    )
    
    parser.add_argument(
        '-op', '--filter-op',
        default='eq',
        choices=['eq'],
        help='过滤操作符（目前支持: eq）'
    )
    
    parser.add_argument(
        '-v', '--filter-value',
        required=True,
        help='过滤值'
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
    
    # 验证源图片目录（如果指定了的话）
    if args.source_image_dir and not os.path.exists(args.source_image_dir):
        logger.error(f"源图片目录不存在: {args.source_image_dir}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理CSV文件
    process_csv(
        args.input_csv, 
        args.output_csv, 
        args.source_image_dir, 
        args.output_image_dir,
        args.filter_column, 
        args.filter_op, 
        args.filter_value
    )


if __name__ == '__main__':
    main()


# # 从本地图片目录复制图片
# python hsay_copy_image_by_condition.py -i input.csv -o output.csv \
#   -s ./source_images -d ./output_images \
#   -c "店铺名称" -op eq -v "某某店"

# # 从URL下载图片（不指定源图片目录）
# python hsay_copy_image_by_condition.py -i input.csv -o output.csv \
#   -d ./output_images -c "状态" -op eq -v "已完成"