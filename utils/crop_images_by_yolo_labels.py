#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据YOLO格式的标签文件裁剪图像
"""

import os
import argparse
from PIL import Image
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局统计变量
stats = {
    'total_images': 0,
    'processed_images': 0,
    'successfully_cropped': 0,
    'total_cropped_objects': 0,
    'no_objects_found': 0,
    'missing_labels': 0,
    'missing_images': 0,
    'format_errors': 0,
    'other_errors': 0
}

def crop_image_by_label(image_path, label_path, output_dir, confidence_threshold=None):
    """
    根据YOLO格式的标签裁剪图像
    
    Args:
        image_path (str): 图像文件路径
        label_path (str): YOLO标签文件路径
        output_dir (str): 输出目录
        confidence_threshold (float, optional): 置信度阈值，None表示不使用
    
    Returns:
        int: 成功裁剪的目标数量
    """
    stats['total_images'] += 1
    
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            logger.warning(f"图像文件不存在: {image_path}")
            stats['missing_images'] += 1
            return 0
        
        if not os.path.exists(label_path):
            logger.warning(f"标签文件不存在: {label_path}")
            stats['missing_labels'] += 1
            return 0
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取图像
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        # 获取图像名称（不含扩展名）
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 读取标签文件
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = f.readlines()
        
        # 裁剪每个目标
        crop_count = 0
        for i, label_line in enumerate(labels):
            # 解析YOLO格式：class x_center y_center width height [confidence]
            parts = label_line.strip().split()
            if len(parts) < 5:
                logger.warning(f"标签格式错误，行 {i+1}: {label_line.strip()}")
                stats['format_errors'] += 1
                continue
            
            # 检查是否有置信度值
            has_confidence = len(parts) >= 6
            
            # 如果设置了置信度阈值且标签包含置信度
            if confidence_threshold is not None and has_confidence:
                confidence = float(parts[5])
                if confidence < confidence_threshold:
                    logger.debug(f"跳过置信度低于阈值的目标: {confidence}")
                    continue
            
            # 解析坐标
            class_id = parts[0]
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            # 计算裁剪区域（左上和右下坐标）
            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(img_width, int(x_center + width / 2))
            y2 = min(img_height, int(y_center + height / 2))
            
            # 裁剪图像
            cropped_image = image.crop((x1, y1, x2, y2))
            
            # 保存裁剪后的图像
            output_filename = f"{image_name}_crop_{i:03d}_class{class_id}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cropped_image.save(output_path)
            crop_count += 1
        
        stats['processed_images'] += 1
        
        if crop_count > 0:
            stats['successfully_cropped'] += 1
            stats['total_cropped_objects'] += crop_count
            logger.info(f"成功裁剪 {crop_count} 个目标到 {output_dir}")
        else:
            stats['no_objects_found'] += 1
            logger.info(f"没有找到符合条件的目标进行裁剪: {image_path}")
            
        return crop_count
            
    except Exception as e:
        stats['other_errors'] += 1
        logger.error(f"处理文件时出错: {image_path}, 错误: {str(e)}")
        return 0

def print_summary_statistics():
    """
    打印裁剪操作的汇总统计信息
    """
    logger.info("\n=== 裁剪操作汇总统计 ===")
    logger.info(f"总图像数: {stats['total_images']}")
    logger.info(f"成功处理的图像数: {stats['processed_images']}")
    logger.info(f"成功裁剪出目标的图像数: {stats['successfully_cropped']}")
    logger.info(f"总共裁剪出的目标数: {stats['total_cropped_objects']}")
    logger.info("\n失败原因统计:")
    logger.info(f"  未找到符合条件目标: {stats['no_objects_found']}")
    logger.info(f"  标签文件不存在: {stats['missing_labels']}")
    logger.info(f"  图像文件不存在: {stats['missing_images']}")
    logger.info(f"  标签格式错误: {stats['format_errors']}")
    logger.info(f"  其他错误: {stats['other_errors']}")
    logger.info("=======================\n")

def main():
    parser = argparse.ArgumentParser(description='根据YOLO标签裁剪图像')
    parser.add_argument('--image', '--i', type=str, help='单个图像文件路径')
    parser.add_argument('--label', '--l', type=str, help='单个标签文件路径')
    parser.add_argument('--image-dir', '--i', type=str, help='图像目录路径')
    parser.add_argument('--label-dir', '--l', type=str, help='标签目录路径')
    parser.add_argument('--output-dir', '--o', type=str, required=True, help='输出目录')
    parser.add_argument('--conf-threshold', '--conf', type=float, default=None, 
                        help='置信度阈值，低于此值的目标将被跳过')
    
    args = parser.parse_args()
    
    # 处理单个文件
    if args.image and args.label:
        crop_count = crop_image_by_label(args.image, args.label, args.output_dir, args.conf_threshold)
        logger.info(f"\n单个图像处理完成，成功裁剪出 {crop_count} 个目标")
    # 处理目录
    elif args.image_dir and args.label_dir:
        # 获取所有图像文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_files = [f for f in os.listdir(args.image_dir) 
                      if f.lower().endswith(image_extensions)]
        
        total_images = len(image_files)
        processed = 0
        total_cropped = 0
        
        logger.info(f"开始处理 {total_images} 个图像文件...")
        
        for image_file in image_files:
            # 构建文件路径
            image_path = os.path.join(args.image_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(args.label_dir, label_file)
            
            # 裁剪图像
            crop_count = crop_image_by_label(image_path, label_path, args.output_dir, args.conf_threshold)
            total_cropped += crop_count
            
            processed += 1
            if processed % 10 == 0 or processed == total_images:
                logger.info(f"进度: {processed}/{total_images} 图像已处理，累计裁剪 {total_cropped} 个目标")
        
        # 打印汇总统计
        print_summary_statistics()
        
        # 验证输出目录中的实际文件数量
        if os.path.exists(args.output_dir):
            actual_files = len([f for f in os.listdir(args.output_dir) 
                              if os.path.isfile(os.path.join(args.output_dir, f))])
            logger.info(f"输出目录中实际存在的裁剪文件数量: {actual_files}")
            
            # 验证计数是否一致
            if actual_files == stats['total_cropped_objects']:
                logger.info("文件计数验证成功: 裁剪统计与实际文件数量一致")
            else:
                logger.warning(f"文件计数不一致: 统计 {stats['total_cropped_objects']} 个，实际 {actual_files} 个")
    else:
        parser.error("必须指定 --image 和 --label 或者 --image-dir 和 --label-dir")

if __name__ == "__main__":
    main()