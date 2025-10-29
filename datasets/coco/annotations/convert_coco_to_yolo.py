import json
import os
import argparse

def convert_coco_to_yolo(coco_json_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载COCO JSON文件
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    
    # 类别ID映射（COCO类别ID从1开始，YOLO从0开始）
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    class_id_map = {cat['id']: i for i, cat in enumerate(data['categories'])}
    
    # 保存类别名称到classes.txt
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for i, cat in enumerate(data['categories']):
            f.write(f"{i} {cat['name']}\n")
    
    # 处理每个图像的标注
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # 获取图像信息映射
    images = {img['id']: img for img in data['images']}
    
    # 生成YOLO标注文件
    for img_id, anns in annotations_by_image.items():
        img = images[img_id]
        img_width = img['width']
        img_height = img['height']
        img_filename = os.path.splitext(img['file_name'])[0]  # 去掉扩展名
        output_file = os.path.join(output_dir, f"{img_filename}.txt")
        
        with open(output_file, 'w') as f:
            for ann in anns:
                category_id = ann['category_id']
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                
                # 归一化坐标
                x_center = (x + w/2) / img_width
                y_center = (y + h/2) / img_height
                width = w / img_width
                height = h / img_height
                
                # 转换类别ID（COCO ID从1开始，YOLO从0开始）
                yolo_class_id = class_id_map[category_id]
                
                # 写入一行：class_id x_center y_center width height
                f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"成功转换 {len(annotations_by_image)} 张图像的标注到YOLO格式")
    print(f"类别文件已保存到: {os.path.join(output_dir, 'classes.txt')}")
    print(f"YOLO标注文件已保存到: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将COCO格式标注转换为YOLO格式')
    parser.add_argument('--coco-json', required=True, help='COCO JSON标注文件路径（如instances_val2017.json）')
    parser.add_argument('--output-dir', required=True, help='YOLO标注文件输出目录')
    args = parser.parse_args()
    
    convert_coco_to_yolo(args.coco_json, args.output_dir)