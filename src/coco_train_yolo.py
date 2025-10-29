# 此文件已修改为注释形式，避免自动下载大型COCO数据集

# 原代码 (已注释)：
# from ultralytics import YOLO
# 
# # Load a model
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# 
# # Train the model - 注意：这行会自动下载COCO数据集(>20GB)
# results = model.train(data="coco.yaml", epochs=100, imgsz=640)

# 说明：
# 1. COCO数据集非常大(>20GB)，会占用大量磁盘空间和下载时间
# 2. 如果需要使用自定义数据集训练，请使用项目根目录的train_yolo.py脚本
# 3. 要正确终止正在进行的COCO数据集下载，请在终端中使用Ctrl+C多次或找到并终止对应的Python进程