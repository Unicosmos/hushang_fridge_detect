import os
import re

# 定义要搜索的目录
labels_dir = '/root/hyg/projects/hushang_fridge_detect/data/hyg_rag_1011/labels'

# 存储符合条件的文件路径
matched_files = []

# 遍历目录中的所有.txt文件
try:
    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(labels_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 检查文件的每一行
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # 获取第一个数字
                        first_number_match = re.match(r'^\s*(\d+)', line)
                        if first_number_match:
                            first_number = first_number_match.group(1)
                            # 检查是否包含数字'4'
                            if '0' in first_number:
                                matched_files.append(file_path)
                                break  # 找到符合条件的行后，跳出循环检查下一个文件
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")

    # 打印结果
    if matched_files:
        print(f"找到 {len(matched_files)} 个符合条件的文件：")
        for file_path in matched_files:
            print(file_path)
    else:
        print("没有找到符合条件的文件。")
except Exception as e:
    print(f"遍历目录时出错: {e}")