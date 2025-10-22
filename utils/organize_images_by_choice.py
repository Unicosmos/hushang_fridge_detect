#!/usr/bin/env python3
"""
按 CSV 中的分类(choice)重新组织图片：
- 读取 CSV 的 `image` 列，仅使用文件名进行匹配（忽略路径）
- 将输入目录中的图片复制到以分类名为子目录的输出目录

用法示例：
    python3 scripts/organize_images_by_choice.py \
        --csv-path /root/zhifang/cruise/data/classification/gm-led_num_orientation_classify-251015/dataset.csv \
        --images-dir /root/zhifang/cruise/data/classification/gm-led_num_orientation_classify-251015/images \
        --output-dir /root/zhifang/cruise/data/classification/gm-led_num_orientation_classify-251015/organized_by_choice

默认参数已指向上述路径，可直接：
    python3 scripts/organize_images_by_choice.py

参数：
- --dry-run       仅打印统计信息，不执行复制
- --overwrite     允许覆盖已存在的目标文件
- --include-unknown  将未标注(choice 为空)的图片复制到“未标注”目录
- --unknown-name  未标注目录名称，默认“未标注”
"""

import argparse
import csv
import os
from pathlib import Path
import shutil
from collections import Counter, defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 CSV 的分类将图片复制到对应子目录，仅按文件名匹配。"
    )
    parser.add_argument("--csv-path", help="CSV 文件路径")
    parser.add_argument("--images-dir", help="输入图片目录")
    parser.add_argument("--output-dir", help="输出根目录")
    parser.add_argument("--dry-run", action="store_true", help="仅打印统计信息，不执行复制")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的目标文件")
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="将未标注(choice 为空)的图片复制到未标注目录",
    )
    parser.add_argument(
        "--unknown-name",
        default="未标注",
        help="未标注目录名称（choice 为空时使用）",
    )
    return parser.parse_args()


def read_csv_mapping(csv_path: Path):
    """读取 CSV，返回 (filename -> choice) 映射，以及统计信息。"""
    mapping = {}
    conflicts = defaultdict(set)
    duplicate_rows = 0

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image" not in reader.fieldnames or "choice" not in reader.fieldnames:
            raise ValueError("CSV 缺少必要列：image 或 choice")

        for row in reader:
            image_path = (row.get("image") or "").strip()
            choice = (row.get("choice") or "").strip()
            if not image_path:
                continue

            # 只使用文件名进行匹配
            filename = os.path.basename(image_path)
            if not filename:
                continue

            if filename in mapping:
                # 已存在且分类一致则忽略；不一致则记录冲突
                if mapping[filename] != choice:
                    conflicts[filename].update([mapping[filename], choice])
                duplicate_rows += 1
                continue

            mapping[filename] = choice

    return mapping, conflicts, duplicate_rows


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()

    csv_path = Path(args.csv_path)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在：{csv_path}")
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"输入图片目录不存在或非目录：{images_dir}")

    mapping, conflicts, duplicate_rows = read_csv_mapping(csv_path)

    # 统计与复制
    category_counter = Counter()
    missing_in_dir = []
    copied_count = 0

    # 构建输入目录的文件集合（仅文件名）
    available_filenames = {p.name for p in images_dir.iterdir() if p.is_file()}

    # 为了避免重复复制，同一文件只复制一次
    for filename, choice in mapping.items():
        if filename not in available_filenames:
            missing_in_dir.append(filename)
            continue

        src = images_dir / filename
        # 未标注分类处理
        target_choice = choice if choice else (args.unknown_name if args.include_unknown else None)
        if target_choice is None:
            # 跳过未标注
            continue

        dest_dir = output_dir / target_choice
        dest = dest_dir / filename

        category_counter[target_choice] += 1

        if args.dry_run:
            # 仅统计，不复制
            continue

        ensure_dir(dest_dir)
        if dest.exists() and not args.overwrite:
            # 不覆盖已存在文件
            continue

        shutil.copy2(src, dest)
        copied_count += 1

    # 打印总结
    print("=== 按分类复制图片 总结 ===")
    print(f"CSV 路径: {csv_path}")
    print(f"输入目录: {images_dir}")
    print(f"输出目录: {output_dir}")
    print(f"CSV 行中重复同名文件计数: {duplicate_rows}")
    print(f"CSV 提供的唯一文件名数: {len(mapping)}")
    print(f"输入目录可用文件数: {len(available_filenames)}")
    print(f"输入目录缺失的 CSV 文件数: {len(missing_in_dir)}")
    if missing_in_dir:
        print(f"示例缺失(前10): {missing_in_dir[:10]}")

    if conflicts:
        print(f"存在分类冲突的文件数: {len(conflicts)}")
        sample_conflict = next(iter(conflicts.items()))
        print(f"示例冲突: {sample_conflict[0]} -> {sorted(sample_conflict[1])}")
    else:
        print("未检测到分类冲突。")

    if args.dry_run:
        print("模式: dry-run（未执行复制）")
    else:
        print(f"已复制文件数: {copied_count}")

    if category_counter:
        print("各分类计数:")
        for cat, cnt in category_counter.most_common():
            print(f"  {cat}: {cnt}")
    else:
        print("无待复制的分类或文件。")


if __name__ == "__main__":
    main()