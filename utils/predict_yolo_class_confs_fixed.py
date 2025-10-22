import os
import psutil
import time
import argparse
from datetime import datetime
import yaml
from pathlib import Path
from ultralytics import YOLO
from zoneinfo import ZoneInfo
import json
import numpy as np
import gc
import torch

# 设置北京时区
BEIJING_TZ = ZoneInfo("Asia/Shanghai")

# 配置日志
from logging import getLogger, StreamHandler, Formatter
logger = getLogger(__name__)
handler = StreamHandler()
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel('INFO')


def cleanup_memory():
    """清理内存的辅助函数"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_single_result(result, config, task_type, save_dir, result_index, filter_mask=None):
    """处理单个预测结果并保存JSON文件
    
    Args:
        result: YOLO预测结果对象
        config: 配置字典
        task_type: 任务类型
        save_dir: 保存目录
        result_index: 结果索引
        filter_mask: 过滤掩码，指定哪些检测应该保留
    """
    try:
        # 从过滤后的结果中直接构建数据，而不是使用to_json()
        yolo_data = []
        
        # 对于检测和分割任务
        if task_type in ["detect", "segment"] and hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes
            
            # 确保所有需要的属性都存在
            if hasattr(boxes, "xyxy") and hasattr(boxes, "conf") and hasattr(boxes, "cls"):
                num_boxes = len(boxes)
                names = getattr(result, "names", {})
                
                # 处理每个边界框，应用过滤掩码
                for i in range(num_boxes):
                    # 如果提供了过滤掩码，并且当前检测不应该保留，则跳过
                    if filter_mask is not None and i < len(filter_mask) and not filter_mask[i]:
                        continue
                    
                    # 获取类别信息
                    cls_id = int(boxes.cls[i].item())
                    cls_name = names.get(cls_id, f"class_{cls_id}")
                    confidence = float(boxes.conf[i].item())
                    
                    # 获取边界框坐标
                    x1 = float(boxes.xyxy[i][0].item())
                    y1 = float(boxes.xyxy[i][1].item())
                    x2 = float(boxes.xyxy[i][2].item())
                    y2 = float(boxes.xyxy[i][3].item())
                    
                    box_data = {
                        "name": cls_name,
                        "confidence": confidence,
                        "box": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        }
                    }
                    
                    # 如果是分割任务且有mask信息，并且提供了过滤掩码
                    if task_type == "segment" and hasattr(result, "masks") and result.masks is not None:
                        masks = result.masks
                        if hasattr(masks, "xy") and i < len(masks.xy):
                            # 如果提供了过滤掩码，确保只保留对应的掩码
                            if filter_mask is None or filter_mask[i]:
                                polygon = masks.xy[i]
                                box_data["segments"] = {
                                    "x": [float(point[0]) for point in polygon],
                                    "y": [float(point[1]) for point in polygon]
                                }
                    
                    yolo_data.append(box_data)
        
        # 对于分类任务
        elif task_type == "classify" and hasattr(result, "probs") and result.probs is not None:
            names = getattr(result, "names", {})
            top5_indices = result.probs.top5
            top5_confs = result.probs.top5conf
            
            for idx, conf in zip(top5_indices, top5_confs):
                yolo_data.append({
                    "name": names.get(int(idx.item()), f"class_{int(idx.item())}"),
                    "confidence": float(conf.item())
                })
        
        # 如果没有成功构建数据，尝试使用原始JSON（作为后备方案）
        if not yolo_data:
            try:
                json_str = result.to_json()
                yolo_data = json.loads(json_str)
            except Exception as e:
                logger.warning(f"无法从结果对象构建数据，ID: {result_index}, 错误: {e}")
        
        # 确定图片路径和文件名
        if hasattr(result, "path") and result.path:
            source_name = Path(result.path).stem
            file_name = Path(result.path).name
            json_file = save_dir / f"{source_name}.json"

            # 根据是否有data_prefix参数来设置image路径
            if config.get("data_prefix"):
                prefix = config["data_prefix"].rstrip("/")
                image_path = f"{prefix}/{file_name}"
            else:
                image_path = str(result.path)
        else:
            source_name = f"result_{result_index}"
            file_name = f"result_{result_index}.jpg"
            json_file = save_dir / f"result_{result_index}.json"

            # 根据是否有data_prefix参数来设置image路径
            if config.get("data_prefix"):
                prefix = config["data_prefix"].rstrip("/")
                image_path = f"{prefix}/{file_name}"
            else:
                image_path = file_name

        # 获取图片尺寸信息
        original_width = result.orig_shape[1] if hasattr(result, "orig_shape") else 640
        original_height = result.orig_shape[0] if hasattr(result, "orig_shape") else 640

        # 转换为Label Studio格式
        label_studio_data = {
            "data": {"image": image_path},
            "predictions": [
                {
                    "model_version": "yolo",
                    "score": 0.0,  # 整体预测分数，可以设置为平均置信度
                    "result": [],
                }
            ],
        }

        # 计算平均置信度作为整体分数
        if yolo_data:
            avg_confidence = sum(item.get("confidence", 0) for item in yolo_data) / len(yolo_data)
            label_studio_data["predictions"][0]["score"] = round(avg_confidence, 5)

        # 转换每个检测结果
        for idx, detection in enumerate(yolo_data):
            # 根据任务类型创建不同格式的结果项
            if task_type == "segment" and "segments" in detection:
                # 分割任务：使用polygon格式
                segments = detection.get("segments", {})
                x_coords = segments.get("x", [])
                y_coords = segments.get("y", [])

                if x_coords and y_coords and len(x_coords) == len(y_coords):
                    # 将像素坐标转换为百分比坐标
                    points = []
                    for x, y in zip(x_coords, y_coords):
                        x_pct = (x / original_width) * 100
                        y_pct = (y / original_height) * 100
                        points.append([round(x_pct, 2), round(y_pct, 2)])

                    result_item = {
                        "id": f"result_{idx}",
                        "type": "polygonlabels",
                        "from_name": "label",
                        "to_name": "image",
                        "original_width": original_width,
                        "original_height": original_height,
                        "image_rotation": 0,
                        "value": {
                            "points": points,
                            "polygonlabels": [detection.get("name", "unknown")],
                        },
                    }

                    # 添加置信度信息
                    if "confidence" in detection:
                        result_item["value"]["confidence"] = round(detection["confidence"], 5)

                    label_studio_data["predictions"][0]["result"].append(result_item)
            elif task_type == "classify":
                # 分类任务：使用choices格式
                result_item = {
                    "id": f"result_{idx}",
                    "type": "choices",
                    "from_name": "choice",
                    "to_name": "image",
                    "value": {"choices": [detection.get("name", "unknown")]},
                }

                # 添加置信度信息
                if "confidence" in detection:
                    result_item["value"]["confidence"] = round(detection["confidence"], 5)

                label_studio_data["predictions"][0]["result"].append(result_item)
            else:
                # 检测任务：使用rectangle格式
                box = detection.get("box", {})

                # 将像素坐标转换为百分比坐标（Label Studio要求）
                x1_pct = (box.get("x1", 0) / original_width) * 100
                y1_pct = (box.get("y1", 0) / original_height) * 100
                x2_pct = (box.get("x2", 0) / original_width) * 100
                y2_pct = (box.get("y2", 0) / original_height) * 100

                width_pct = x2_pct - x1_pct
                height_pct = y2_pct - y1_pct

                # 创建Label Studio格式的结果项
                result_item = {
                    "id": f"result_{idx}",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": round(x1_pct, 2),
                        "y": round(y1_pct, 2),
                        "width": round(width_pct, 2),
                        "height": round(height_pct, 2),
                        "rectanglelabels": [detection.get("name", "unknown")],
                    },
                }

                # 添加置信度信息（可选）
                if "confidence" in detection:
                    result_item["value"]["confidence"] = round(detection["confidence"], 5)

                label_studio_data["predictions"][0]["result"].append(result_item)

        # 保存Label Studio格式的JSON文件
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(label_studio_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Label Studio格式JSON结果已保存到: {json_file}")
        return True

    except Exception as e:
        logger.error(f"保存第{result_index}个结果的Label Studio格式JSON文件时出错: {e}")
        return False


def filter_results_by_class_confidence(result, class_confs, global_conf, task_type):
    """根据类别置信度阈值过滤检测结果
    
    注意：在新版本的Ultralytics YOLO库中，Boxes对象的属性是只读的，
    因此我们返回过滤掩码而不是修改原始对象。
    
    Args:
        result: YOLO预测结果对象
        class_confs: 类别置信度阈值字典，可以是类别名称或类别ID
        global_conf: 全局置信度阈值
        task_type: 任务类型
        
    Returns:
        keep_mask: 保留的检测的布尔掩码数组，None表示不需要过滤
    """
    # 记录日志，确认进入过滤函数
    logger.debug(f"进入类别置信度过滤函数，类别置信度: {class_confs}")
    
    # 对于分类任务，不进行过滤
    if not class_confs or task_type == "classify":
        logger.debug("无类别置信度或分类任务，跳过过滤")
        return None

    # 对于检测和分割任务，计算过滤掩码
    if hasattr(result, "boxes") and result.boxes is not None:
        boxes = result.boxes
        if len(boxes) == 0:
            logger.debug("Boxes为空，跳过过滤")
            return None

        # 获取类别ID和置信度
        cls_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else np.array([])
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.array([])
        
        logger.debug(f"原始检测数量: {len(confs)}")

        # 创建掩码，保留满足条件的检测
        keep_mask = np.ones_like(confs, dtype=bool)

        for i, (cls_id, conf) in enumerate(zip(cls_ids, confs)):
            # 默认使用全局置信度阈值
            threshold = global_conf
            
            # 尝试按类别ID查找阈值
            if int(cls_id) in class_confs:
                threshold = class_confs[int(cls_id)]
                logger.debug(f"类别ID {int(cls_id)} 找到置信度阈值: {threshold}")
            # 尝试按类别名称查找阈值
            elif hasattr(result, "names") and result.names:
                cls_name = result.names.get(int(cls_id), "")
                if cls_name in class_confs:
                    threshold = class_confs[cls_name]
                    logger.debug(f"类别名称 {cls_name} 找到置信度阈值: {threshold}")
            
            # 应用阈值过滤
            if conf < threshold:
                keep_mask[i] = False
                logger.debug(f"过滤掉检测 {i}: 类别={int(cls_id)}, 置信度={conf:.4f} < 阈值={threshold}")
            else:
                logger.debug(f"保留检测 {i}: 类别={int(cls_id)}, 置信度={conf:.4f} >= 阈值={threshold}")

        # 统计过滤结果
        kept_count = np.sum(keep_mask)
        logger.debug(f"过滤后保留的检测数量: {kept_count}/{len(confs)}")

        return keep_mask
    
    return None


def process_results_streaming(model, source, config, predict_args, task_type):
    """流式处理预测结果，避免内存累积"""
    # 使用stream=True启用流式处理
    predict_args_streaming = predict_args.copy()
    predict_args_streaming["stream"] = True
    
    # 获取类别置信度配置
    class_confs = config.get("class_confs", {})
    global_conf = config.get("conf", 0.25)
    
    # 记录类别置信度设置
    if class_confs:
        logger.info(f"应用类别置信度阈值: {class_confs}")
        # 设置一个较低的全局阈值，确保所有可能需要的检测都被保留
        min_class_conf = min(class_confs.values(), default=global_conf)
        predict_args_streaming["conf"] = min(global_conf, min_class_conf * 0.9)
        logger.info(f"设置初始置信度阈值为: {predict_args_streaming['conf']} (为后续类别过滤保留足够检测)")
    else:
        logger.info(f"使用全局置信度阈值: {global_conf}")

    # 开始流式预测
    results_generator = model.predict(source=source, **predict_args_streaming)

    processed_count = 0
    json_saved_count = 0
    detection_summary = {}
    total_detections = 0
    save_dir = None

    # 确定保存目录
    if config.get("save_json", False):
        save_dir = Path(
            predict_args.get("project", f"runs/{task_type}")
        ) / predict_args.get("name", "predict")
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"JSON文件将保存到: {save_dir}")

    logger.info("开始流式处理预测结果...")

    for result in results_generator:
        try:
            processed_count += 1
            
            # 计算过滤掩码（不再修改原始result对象）
            filter_mask = None
            if class_confs:
                filter_mask = filter_results_by_class_confidence(result, class_confs, global_conf, task_type)
                
            # 记录当前结果的检测数量（用于调试）
            if hasattr(result, "boxes") and result.boxes is not None:
                if filter_mask is not None:
                    filtered_count = np.sum(filter_mask)
                    logger.debug(f"处理第{processed_count}张图片，过滤后检测数量: {filtered_count}")
                else:
                    logger.debug(f"处理第{processed_count}张图片，检测数量: {len(result.boxes)}")

            # 统计检测结果 - 应用过滤掩码
            if task_type == "classify" or (
                hasattr(result, "probs") and result.probs is not None
            ):
                # 分类任务
                if hasattr(result, "probs") and result.probs is not None:
                    top1_idx = result.probs.top1
                    top1_name = result.names.get(top1_idx, f"class_{top1_idx}")
                    detection_summary[top1_name] = (
                        detection_summary.get(top1_name, 0) + 1
                    )
                    total_detections += 1
            elif hasattr(result, "boxes") and result.boxes is not None:
                # 检测/分割任务 - 应用过滤掩码统计
                cls_ids = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else np.array([])
                
                if filter_mask is not None:
                    # 应用过滤掩码
                    filtered_cls_ids = cls_ids[filter_mask]
                    total_detections += len(filtered_cls_ids)
                    
                    # 统计过滤后的每个类别的检测数量
                    for cls_id in filtered_cls_ids:
                        cls_id = int(cls_id.item())
                        cls_name = result.names.get(cls_id, f"class_{cls_id}")
                        detection_summary[cls_name] = (
                            detection_summary.get(cls_name, 0) + 1
                        )
                else:
                    # 无过滤，直接统计
                    total_detections += len(cls_ids)
                    for cls_id in cls_ids:
                        cls_id = int(cls_id.item())
                        cls_name = result.names.get(cls_id, f"class_{cls_id}")
                        detection_summary[cls_name] = (
                            detection_summary.get(cls_name, 0) + 1
                        )

            # 处理JSON保存 - 传递过滤掩码
            if config.get("save_json", False) and save_dir:
                if process_single_result(
                    result, config, task_type, save_dir, processed_count, filter_mask
                ):
                    json_saved_count += 1

            # 定期清理内存和报告进度
            if processed_count % 50 == 0:
                cleanup_memory()
                current_memory = psutil.virtual_memory().percent
                logger.info(
                    f"已处理 {processed_count} 张图片，当前内存使用: {current_memory:.1f}%"
                )

                # 如果内存使用超过85%，强制清理
                if current_memory > 85:
                    logger.warning(
                        f"内存使用过高 ({current_memory:.1f}%)，执行强制清理"
                    )
                    cleanup_memory()
                    time.sleep(0.1)  # 短暂暂停让系统回收内存

        except Exception as e:
            logger.error(f"处理第 {processed_count} 个结果时出错: {e}")
            continue

    # 最终清理
    cleanup_memory()

    return {
        "processed_count": processed_count,
        "total_detections": total_detections,
        "detection_summary": detection_summary,
        "json_saved_count": json_saved_count,
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLO目标检测预测脚本")

    # 基础配置
    parser.add_argument("--config", type=str, help="预测配置文件路径（可选）")
    parser.add_argument("--model", type=str, required=False, help="YOLO模型权重文件路径")
    parser.add_argument("--source", type=str, required=False, help="输入源：图片路径、目录、URL、视频等")
    parser.add_argument(
        "--task",
        type=str,
        choices=["detect", "obb", "segment", "classify", "pose"],
        help="YOLO任务类型：detect(检测), obb(旋转框检测), segment(分割), classify(分类), pose(姿态估计)",
    )

    # 预测参数
    parser.add_argument("--name", type=str, default=None, help="实验名称")
    parser.add_argument("--imgsz", type=int, help="输入图像尺寸")
    parser.add_argument("--conf", type=float, help="全局置信度阈值")
    parser.add_argument("--class-confs", type=str, help="不同类别的置信度阈值，格式为'class1:conf1,class2:conf2'或'id1:conf1,id2:conf2'")
    parser.add_argument("--iou", type=float, help="NMS IoU阈值")
    parser.add_argument("--max-det", type=int, help="每张图像最大检测数")
    parser.add_argument("--device", type=str, default="", help="推理设备，如：0,1,2,3 或 cpu")
    parser.add_argument("--half", action="store_true", help="使用FP16半精度推理")
    parser.add_argument("--dnn", action="store_true", help="使用OpenCV DNN进行ONNX推理")

    # 输出配置
    parser.add_argument("--save", action="store_true", help="保存预测结果图像")
    parser.add_argument("--save-json", action="store_true", help="保存结果到JSON文件")
    parser.add_argument(
        "--data-prefix",
        type=str,
        default=None,
        dest="data_prefix",
        help="JSON文件中data.image字段的路径前缀",
    )
    parser.add_argument("--project", type=str, default="runs/detect", help="保存结果的项目目录")
    parser.add_argument("--exist-ok", action="store_true", help="现有项目/名称可以，不递增")

    # 其他配置
    parser.add_argument("--verbose", action="store_true", help="显示详细预测日志")
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="按类别过滤")
    parser.add_argument("--agnostic-nms", action="store_true", help="类别无关的NMS")
    parser.add_argument("--augment", action="store_true", help="增强推理")

    return parser.parse_args()


def load_and_override_config(config_path, args):
    """加载配置文件并用命令行参数覆盖"""
    config = {}

    # 如果提供了配置文件，则加载它
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"加载配置文件: {config_path}")

    # 用命令行参数覆盖配置文件中的设置
    if args.model is not None:
        config["model"] = args.model
    if args.source is not None:
        config["source"] = args.source
    if args.task is not None:
        config["task"] = args.task
    if args.name is not None:
        config["name"] = args.name
    else:
        # 设置默认名称
        timestamp = datetime.now(BEIJING_TZ).strftime("%Y%m%d%H%M%S")
        config["name"] = f"predict_{timestamp}"

    # 预测参数
    if args.imgsz is not None:
        config["imgsz"] = args.imgsz
    if args.conf is not None:
        config["conf"] = args.conf
    if args.class_confs is not None:
        # 解析类别置信度阈值，格式为'class1:conf1,class2:conf2'或'id1:conf1,id2:conf2'
        class_confs = {}
        try:
            for item in args.class_confs.split(","):
                if ":" in item:
                    cls, conf = item.split(":")
                    try:
                        # 尝试将类别解析为整数（ID）
                        cls_id = int(cls.strip())
                        class_confs[cls_id] = float(conf.strip())
                    except ValueError:
                        # 如果不是整数，则作为类别名称
                        class_confs[cls.strip()] = float(conf.strip())
            config["class_confs"] = class_confs
            logger.info(f"设置类别置信度阈值: {class_confs}")
        except Exception as e:
            logger.error(f"解析class-confs参数时出错: {e}")
    if args.iou is not None:
        config["iou"] = args.iou
    if args.max_det is not None:
        config["max_det"] = args.max_det

    # 设备设置
    if args.device:
        if args.device.lower() == "cpu":
            config["device"] = "cpu"
        else:
            # 将逗号分隔的设备ID转换为列表
            device_list = [int(d.strip()) for d in args.device.split(",")]
            config["device"] = device_list

    # 其他布尔参数
    boolean_params = ["half", "dnn", "save", "save_json", "exist_ok", "verbose", "agnostic_nms", "augment"]
    for param in boolean_params:
        if getattr(args, param):
            config[param] = True

    # 其他参数设置
    if args.data_prefix is not None:
        config["data_prefix"] = args.data_prefix
    if args.project != "runs/detect":
        config["project"] = args.project
    if args.classes is not None:
        config["classes"] = args.classes

    return config


def get_source_info(source_path):
    """获取输入源信息"""
    source_path = Path(source_path)

    if source_path.is_file():
        if source_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            return {"type": "image", "path": str(source_path)}
        elif source_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".wmv"]:
            return {"type": "video", "path": str(source_path)}
        else:
            return {"type": "file", "path": str(source_path)}
    elif source_path.is_dir():
        return {"type": "directory", "path": str(source_path)}
    elif str(source_path).startswith(("http://", "https://")):
        return {"type": "url", "path": str(source_path)}
    elif str(source_path).isdigit():
        return {"type": "webcam", "path": str(source_path)}
    else:
        return {"type": "unknown", "path": str(source_path)}


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 记录初始内存使用情况
    initial_memory_usage = psutil.virtual_memory().percent
    logger.info(f"程序启动时内存使用: {initial_memory_usage:.1f}%")

    # 加载并覆盖配置
    config = load_and_override_config(args.config, args)

    # 检查必需参数
    if "model" not in config or not config["model"]:
        logger.error("错误：必须指定模型路径（通过 --model 参数或配置文件）")
        return

    if "source" not in config or not config["source"]:
        logger.error("错误：必须指定输入源（通过 --source 参数或配置文件）")
        return

    # 加载模型
    model = YOLO(config["model"])

    # 确定任务类型
    model_task_type = model.task
    user_task_type = config.get("task")

    if user_task_type:
        task_type = user_task_type
        logger.info(f"使用用户指定的任务类型: {task_type}")
        if task_type != model_task_type:
            logger.warning(f"用户指定任务类型 ({task_type}) 与模型任务类型 ({model_task_type}) 不匹配")
    else:
        task_type = model_task_type
        logger.info(f"自动检测到模型任务类型: {task_type}")

    # 从预测参数中排除model、source、task、data_prefix，因为它们需要单独处理
    predict_args = {
        k: v
        for k, v in config.items()
        if k not in ["model", "source", "task", "data_prefix", "class_confs"]
    }

    # 根据任务类型设置正确的 project 路径
    if any(
        config.get(save_param, False)
        for save_param in ["save", "save_json"]
    ):
        if "project" not in predict_args:
            predict_args["project"] = f"runs/{task_type}"
            logger.info(f"根据任务类型设置保存路径: runs/{task_type}")

    try:
        logger.info("开始预测...")
        logger.info(f"模型: {config['model']}")
        logger.info(f"输入源: {config['source']}")

        # 使用流式处理执行预测
        logger.info("开始流式预测处理...")
        processing_stats = process_results_streaming(
            model, config["source"], config, predict_args, task_type
        )

        # 打印预测结果摘要
        logger.info(f"预测完成，共处理 {processing_stats['processed_count']} 张图片")
        logger.info(f"总检测数: {processing_stats['total_detections']}")

        if processing_stats["detection_summary"]:
            logger.info(f"类别统计: {processing_stats['detection_summary']}")

        if config.get("save_json", False):
            logger.info(f"成功保存 {processing_stats['json_saved_count']} 个Label Studio格式的JSON文件")

        # 最终内存清理和监控
        cleanup_memory()
        final_memory_usage = psutil.virtual_memory().percent
        logger.info(f"预测完成后内存使用: {final_memory_usage:.1f}%")

        logger.info("预测完成！")

    except Exception as e:
        # 异常时也进行内存清理
        cleanup_memory()
        logger.error(f"预测过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()