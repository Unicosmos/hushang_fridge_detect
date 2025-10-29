# prompt:
# test_dsg.log 列出label为1和label为0的图片名，统计各自图片个数。
# 对于label标记为1的图片，找到它们中和 洁净 中文件名相同的个数




import os
import re
import json

# 读取test_dsg.log文件并分析
log_file = '/root/hyg/projects/hushang_fridge_detect/test_dsg.log'
clean_dir = '/root/hyg/projects/hushang_fridge_detect/data/cls/hyg_dlg/洁净'

label_0_images = []
label_1_images = []

# 解析日志文件
with open(log_file, 'r', encoding='utf-8') as f:
    content = f.read()
    # 使用正则表达式匹配图片路径和对应的label
    pattern = r'\[(\d+)/95\] (.+?)\n\{\s*"label": (\d+),'  
    matches = re.findall(pattern, content)
    
    for match in matches:
        img_path = match[1]
        label = int(match[2])
        # 提取文件名（不含扩展名）
        img_name = os.path.basename(img_path)
        # 移除扩展名，因为洁净目录中的文件名可能有不同的扩展名（_0_0.jpg）
        img_name_without_ext = os.path.splitext(img_name)[0]
        
        if label == 0:
            label_0_images.append((img_name, img_name_without_ext))
        else:
            label_1_images.append((img_name, img_name_without_ext))

# 统计数量
print(f"Label为0的图片数量: {len(label_0_images)}")
print(f"Label为0的图片列表:")
for img_name, _ in label_0_images:
    print(f"  - {img_name}")

print(f"\nLabel为1的图片数量: {len(label_1_images)}")
print(f"Label为1的图片列表:")
for img_name, _ in label_1_images:
    print(f"  - {img_name}")

# 获取洁净目录中的文件名（不含扩展名部分）
clean_files = set()
if os.path.exists(clean_dir):
    for file in os.listdir(clean_dir):
        # 移除扩展名和可能的_0_0部分
        base_name = os.path.splitext(file)[0]
        # 处理如 "xxx_0_0" 这样的文件名
        if '_0_0' in base_name:
            base_name = base_name.replace('_0_0', '')
        clean_files.add(base_name)

# 查找label为1且在洁净目录中存在的图片
matched_count = 0
matched_images = []

for img_name, img_name_without_ext in label_1_images:
    # 检查两种情况：完全匹配或带_0_0后缀的匹配
    if img_name_without_ext in clean_files or f"{img_name_without_ext}_0_0" in [f for f in os.listdir(clean_dir) if os.path.isfile(os.path.join(clean_dir, f))]:
        matched_count += 1
        matched_images.append(img_name)
    else:
        # 直接检查洁净目录中的文件名
        for clean_file in os.listdir(clean_dir):
            clean_base = os.path.splitext(clean_file)[0]
            if '_0_0' in clean_base:
                clean_base = clean_base.replace('_0_0', '')
            if clean_base == img_name_without_ext:
                matched_count += 1
                matched_images.append(img_name)
                break

print(f"\n在Label为1的图片中，与洁净目录文件名相同的数量: {matched_count}")
print(f"匹配的图片列表:")
for img_name in matched_images:
    print(f"  - {img_name}")