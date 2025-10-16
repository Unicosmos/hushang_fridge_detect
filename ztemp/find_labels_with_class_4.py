import os
import re

# 定义要搜索的标签目录
labels_dir = '/root/hyg/projects/hushang_fridge_detect/data/hyg_rag_1011/labels'

# 存储包含类别编号4的文件路径
files_with_class_4 = []

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
                        
                        # 使用正则表达式匹配行首的数字
                        match = re.match(r'^(\d+)', line)
                        if match:
                            class_id = match.group(1)
                            # 检查是否是类别4
                            if class_id == '4':
                                files_with_class_4.append(file_path)
                                print(f"在文件 {file_path} 中找到类别4")
                                break  # 找到后跳出循环，检查下一个文件
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")

    # 打印结果统计
    if files_with_class_4:
        print(f"\n总共找到 {len(files_with_class_4)} 个包含类别4的文件。")
        print("建议清理这些文件中的类别4标注或更新数据集配置。")
    else:
        print("\n没有找到包含类别4的文件。")
        print("CUDA错误可能由其他原因引起，建议检查数据集配置和模型设置。")
except Exception as e:
    print(f"遍历目录时出错: {e}")