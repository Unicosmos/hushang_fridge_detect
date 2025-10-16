import os
import sys

# 添加项目根目录到Python路径，确保可以导入utils模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 在导入任何可能使用matplotlib的库之前，先导入中文显示修复模块
try:
    from utils.simple_fix_chinese import fix_matplotlib_chinese, MATPLOTLIB_CHINESE_FIXED, has_available_chinese_fonts, reapply_chinese_fix
    # 立即应用修复
    fix_matplotlib_chinese()
except ImportError as e:
    # 如果导入失败，提供备用的修复函数和必要的辅助函数
    MATPLOTLIB_CHINESE_FIXED = False
    
    def fix_matplotlib_chinese():
        try:
            import matplotlib
            # 简单的字体设置
            font_config = {
                'font.family': ['sans-serif'],
                'font.sans-serif': ['DejaVu Sans', 'sans-serif'],
                'axes.unicode_minus': False,
                'text.usetex': False
            }
            matplotlib.rcParams.update(font_config)
            # 尝试导入pyplot并设置
            try:
                from matplotlib import pyplot as plt
                plt.rcParams.update(font_config)
            except ImportError:
                pass
            global MATPLOTLIB_CHINESE_FIXED
            MATPLOTLIB_CHINESE_FIXED = True
        except ImportError:
            pass
    
    def has_available_chinese_fonts():
        """检查系统中是否有可用的中文字体"""
        try:
            import matplotlib.font_manager as fm
            # 检查一些基本的中文字体标识符
            available_fonts = [f.name.lower() for f in fm.fontManager.ttflist]
            chinese_identifiers = ['noto', 'droid', 'hei', 'yahei', 'sim', 'chinese']
            for font in available_fonts:
                if any(identifier in font for identifier in chinese_identifiers):
                    return True
        except Exception:
            pass
        return False
    
    def reapply_chinese_fix(force=False):
        """重新应用中文显示修复"""
        global MATPLOTLIB_CHINESE_FIXED
        if force:
            MATPLOTLIB_CHINESE_FIXED = False
        if not MATPLOTLIB_CHINESE_FIXED:
            try:
                font_config = {
                    'font.family': ['sans-serif'],
                    'font.sans-serif': ['WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei', 'sans-serif'],
                    'axes.unicode_minus': False,
                    'text.usetex': False
                }
                matplotlib.rcParams.update(font_config)
                MATPLOTLIB_CHINESE_FIXED = True
                return True
            except:
                return False
        return True
    
    def reapply_chinese_fix(force=False):
        # force参数在此简单实现中被忽略，但保持函数签名一致
        return fix_matplotlib_chinese()
        
    def has_available_chinese_fonts():
        return True
    
    MATPLOTLIB_CHINESE_FIXED = False

import psutil
import threading
import time
import argparse
from datetime import datetime
import yaml
import mlflow
from pathlib import Path
from ultralytics import YOLO, settings
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG

# 设置北京时区
BEIJING_TZ = ZoneInfo('Asia/Shanghai')


load_dotenv()

# 设置日志器
logger = getLogger(__name__)

os.environ['LD_PRELOAD'] = '/lib/x86_64-linux-gnu/libtcmalloc.so.4'


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO目标检测训练脚本')
    
    # 基础配置
    parser.add_argument('--config', type=str, 
                       help='训练配置文件路径')
    parser.add_argument('--override-config', type=str, default=None,
                       help='覆盖配置文件路径，用于覆盖主配置文件中的参数')
    parser.add_argument('--data', type=str, default=None,
                       help='训练数据配置文件路径，覆盖配置文件中的设置')
    # MLflow配置
    parser.add_argument('--mlflow-uri', type=str, 
                       default='http://47.96.219.105:50001/',
                       help='MLflow服务器地址')
    parser.add_argument('--disable-mlflow', action='store_true',
                       help='禁用MLflow集成')
    parser.add_argument('--mlflow-experiment-description', type=str, default=None,
                       help='MLflow实验描述信息')
    parser.add_argument('--mlflow-task-type', type=str, default='detect',
                       help='YOLO任务类型，如：detect, segment, classify, pose, obb')
    parser.add_argument('--mlflow-tags', type=str, nargs='*', default=None,
                       help='MLflow实验标签，格式：key1=value1 key2=value2')
    
    # 超参数进化配置
    parser.add_argument('--evolve', action='store_true',
                       help='启用YOLO超参数进化功能')
    parser.add_argument('--evolve-population', type=int, default=10,
                       help='超参数进化的种群大小')
    parser.add_argument('--evolve-generations', type=int, default=10,
                       help='超参数进化的代数')
    parser.add_argument('--evolve-mutation', type=float, default=0.4,
                       help='超参数进化的突变概率')
    parser.add_argument('--evolve-crossover', type=float, default=0.2,
                       help='超参数进化的交叉概率')
    
    # 运行时可调整的关键参数
    parser.add_argument('--name', type=str, default=None,
                       help='实验名称，覆盖配置文件中的设置')
    parser.add_argument('--model', type=str, default=None,
                       help='YOLO基础模型路径，覆盖配置文件中的设置')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数，覆盖配置文件中的设置')
    parser.add_argument('--batch', type=int, default=None,
                       help='批次大小，覆盖配置文件中的设置')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='输入图像尺寸，覆盖配置文件中的设置')
    parser.add_argument('--device', type=str, default=None,
                       help='训练设备，如：0,1,2,3 或 cpu，覆盖配置文件中的设置')
    parser.add_argument('--workers', type=int, default=None,
                       help='数据加载进程数，覆盖配置文件中的设置')
    parser.add_argument('--lr0', type=float, default=None,
                       help='初始学习率，覆盖配置文件中的设置')
    parser.add_argument('--patience', type=int, default=None,
                       help='早停耐心值，覆盖配置文件中的设置')
    parser.add_argument('--save-period', type=int, default=None,
                       help='模型保存周期，覆盖配置文件中的设置')
    parser.add_argument('--cache', action='store_true',
                       help='启用图像缓存到内存')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练的路径')
    parser.add_argument('--fraction', type=float, default=None,
                       help='使用数据集的比例，用于快速测试')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子，确保结果可复现')
    
    # 数据增强参数
    parser.add_argument('--mosaic', type=float, default=None,
                       help='Mosaic增强概率')
    parser.add_argument('--mixup', type=float, default=None,
                       help='MixUp增强概率')
    parser.add_argument('--copy-paste', type=float, default=None,
                       help='Copy-Paste增强概率')
    parser.add_argument('--degrees', type=float, default=None,
                       help='旋转角度范围')
    parser.add_argument('--translate', type=float, default=None,
                       help='平移范围')
    parser.add_argument('--scale', type=float, default=None,
                       help='缩放范围')
    parser.add_argument('--fliplr', type=float, default=None,
                       help='水平翻转概率')
    
    # 其他配置
    parser.add_argument('--profile', action='store_true',
                       help='启用性能分析')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细训练日志')
    
    return parser.parse_args()


def setup_evolution_config(config, args):
    """设置超参数进化配置"""
    if args.evolve:
        logger.info("启用YOLO超参数进化功能")
        
        # 设置超参数进化相关参数
        config['evolve'] = True
        config['evolve_iterations'] = args.evolve_generations  # 将generations映射为iterations
        config['evolve_population'] = args.evolve_population
        config['evolve_mutation'] = args.evolve_mutation
        config['evolve_crossover'] = args.evolve_crossover
        
        logger.info(f"超参数进化配置: 迭代次数={args.evolve_generations}, \
                    种群大小={args.evolve_population}, \
                    突变概率={args.evolve_mutation}, \
                    交叉概率={args.evolve_crossover}")
    
    return config


def setup_mlflow(args):
    """配置MLflow集成"""
    if args.disable_mlflow:
        logger.info("MLflow集成已禁用")
        return None, None
    
    # 设置实验名称和运行名称
    mlflow_experiment_name = getattr(args, 'name', None)
    mlflow_run = f'{mlflow_experiment_name}_{datetime.now(BEIJING_TZ).strftime("%Y%m%d%H%M%S")}'
    
    # 设置MLflow 环境变量
    os.environ['MLFLOW_TRACKING_URI'] = args.mlflow_uri
    os.environ['MLFLOW_EXPERIMENT_NAME'] = mlflow_experiment_name
    os.environ['MLFLOW_RUN'] = mlflow_run
    
    # 启用MLflow集成
    settings.update({"mlflow": True})
    
    # 设置MLflow实验和标签
    try:
        mlflow.set_tracking_uri(args.mlflow_uri)
        
        # 准备实验标签
        experiment_tags = {}
        
        # 添加实验描述
        if args.mlflow_experiment_description:
            experiment_tags['mlflow.note.content'] = args.mlflow_experiment_description
        
        # 添加任务类型
        experiment_tags['yolo_task_type'] = args.mlflow_task_type
        
        # 添加自定义标签
        if args.mlflow_tags:
            for tag in args.mlflow_tags:
                if '=' in tag:
                    key, value = tag.split('=', 1)
                    experiment_tags[key.strip()] = value.strip()
        
        # 设置或获取实验
        experiment = mlflow.set_experiment(
            experiment_name=mlflow_experiment_name,
            tags=experiment_tags if experiment_tags else None
        )
        
        logger.info(f"MLflow实验已设置: {experiment.name} (ID: {experiment.experiment_id})")
        if experiment_tags:
            logger.info(f"实验标签: {experiment_tags}")
            
    except Exception as e:
        logger.warning(f"设置MLflow实验信息时出错: {e}")
    
    logger.info(f"MLflow集成已启用 - 实验: {mlflow_experiment_name}, 运行: {mlflow_run}")
    logger.info(f"MLflow服务器: {args.mlflow_uri}")
    logger.info("ultralytics将自动记录训练指标、参数和模型工件")

def load_and_override_config(config_path, args):
    """加载配置文件并用命令行参数覆盖"""
    # 读取主训练配置文件
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"加载主配置文件: {config_path}")
    logger.info(f"主配置文件内容: {config}")
    
    # 如果提供了覆盖配置文件，加载并覆盖主配置文件中的参数
    if args.override_config is not None:
        with open(args.override_config, "r") as f:
            override_config = yaml.safe_load(f)
        
        logger.info(f"加载覆盖配置文件: {args.override_config}")
        logger.info(f"覆盖配置文件内容: {override_config}")
        
        # 使用覆盖配置文件中的参数覆盖主配置文件
        for key, value in override_config.items():
            if key in config and value != config[key]:
                logger.info(f"使用覆盖配置文件中的值覆盖 {key}: {config[key]} -> {value}")
                config[key] = value
            elif key not in config:
                logger.info(f"从覆盖配置文件中添加新参数 {key}: {value}")
                config[key] = value
    
    # 用命令行参数覆盖配置文件中的设置
    if args.data is not None:
        config['data'] = args.data
        logger.info(f"覆盖data: {args.data}")
    
    if args.model is not None:
        config['model'] = args.model
        logger.info(f"覆盖model: {args.model}")
    
    if args.epochs is not None:
        config['epochs'] = args.epochs
        logger.info(f"覆盖epochs: {args.epochs}")
    
 
    if args.name is not None:
        config['name'] = args.name
        logger.info(f"覆盖name: {args.name}")
    else:
        if "name" in config:
            args.name = config['name']
        else:
            logger.error("错误: name未设置。请在命令行中使用--name参数或在配置文件中设置name字段。")
            raise ValueError("name参数是必需的，请设置该参数后重新运行。")
    # 设置 name 为实验名称+时间戳, 避免本地命名冲突自动+1
    origin_name = config["name"]
    timestamp = datetime.now(BEIJING_TZ).strftime("%Y%m%d%H%M%S")
    config['name'] = f"{origin_name}_{timestamp}"
    
    if args.batch is not None:
        config['batch'] = args.batch
        logger.info(f"覆盖batch: {args.batch}")
    
    if args.imgsz is not None:
        config['imgsz'] = args.imgsz
        logger.info(f"覆盖imgsz: {args.imgsz}")
    
    if args.device is not None:
        # 处理设备参数
        if args.device.lower() == 'cpu':
            config['device'] = 'cpu'
        else:
            # 将逗号分隔的设备ID转换为列表
            device_list = [int(d.strip()) for d in args.device.split(',')]
            config['device'] = device_list
        logger.info(f"覆盖device: {config['device']}")
    
    if args.workers is not None:
        config['workers'] = args.workers
        logger.info(f"覆盖workers: {args.workers}")
    
    if args.lr0 is not None:
        config['lr0'] = args.lr0
        logger.info(f"覆盖lr0: {args.lr0}")
    
    if args.patience is not None:
        config['patience'] = args.patience
        logger.info(f"覆盖patience: {args.patience}")
    
    if args.save_period is not None:
        config['save_period'] = args.save_period
        logger.info(f"覆盖save_period: {args.save_period}")
    
    if args.cache:
        config['cache'] = True
        logger.info("启用图像缓存")
    
    if args.resume is not None:
        config['resume'] = args.resume
        logger.info(f"从检查点恢复训练: {args.resume}")
    
    if args.fraction is not None:
        config['fraction'] = args.fraction
        logger.info(f"覆盖fraction: {args.fraction}")
    
    if args.seed is not None:
        config['seed'] = args.seed
        logger.info(f"覆盖seed: {args.seed}")
    
    # 数据增强参数覆盖
    if args.mosaic is not None:
        config['mosaic'] = args.mosaic
        logger.info(f"覆盖mosaic: {args.mosaic}")
    
    if args.mixup is not None:
        config['mixup'] = args.mixup
        logger.info(f"覆盖mixup: {args.mixup}")
    
    if args.copy_paste is not None:
        config['copy_paste'] = args.copy_paste
        logger.info(f"覆盖copy_paste: {args.copy_paste}")
    
    if args.degrees is not None:
        config['degrees'] = args.degrees
        logger.info(f"覆盖degrees: {args.degrees}")
    
    if args.translate is not None:
        config['translate'] = args.translate
        logger.info(f"覆盖translate: {args.translate}")
    
    if args.scale is not None:
        config['scale'] = args.scale
        logger.info(f"覆盖scale: {args.scale}")
    
    if args.fliplr is not None:
        config['fliplr'] = args.fliplr
        logger.info(f"覆盖fliplr: {args.fliplr}")
    
    # 其他参数覆盖
    if args.profile:
        config['profile'] = True
        logger.info("启用性能分析")
    
    if args.verbose:
        config['verbose'] = True
        logger.info("启用详细日志")
    
    return config


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载并覆盖配置
    config = load_and_override_config(args.config, args)
    
    # 设置超参数进化配置
    config = setup_evolution_config(config, args)
    
    # 配置MLflow集成
    setup_mlflow(args)
    
    # MLflow集成状态检查
    if not args.disable_mlflow and not settings.get('mlflow', False):
        logger.info("MLflow集成已禁用")
    

    
    # 获取任务名称，用于组织目录结构
    task = config.get('task', 'default')
    
    # 处理模型路径（支持 OSS）
    logger.info(f"处理模型路径: {config['model']}")

    
    # 处理数据集路径（支持 OSS）
    logger.info(f"处理数据集路径: {config['data']}")
    
    # 使用处理后的本地路径加载模型
    print(config['model'])
    logger.info(config['model'])
    model = YOLO(config['model'])
    
    # 从训练参数中排除model、data，因为它们需要单独处理
    train_args = {k: v for k, v in config.items() if k not in ['model', 'data', 'evolve', 'evolve_iterations', 'evolve_population', 'evolve_mutation', 'evolve_crossover']}
    
    try:
        logger.info("开始训练...")
        logger.info(f"使用配置文件: {args.config}")
        logger.info(f"模型: {config['model']}")
        logger.info(f"数据集: {config['data']}")
        
        # 检查是否启用了超参数进化
        if config.get('evolve', False):
            logger.info("正在进行超参数进化...")
            # 为超参数进化设置专用的项目路径和名称
            # 创建一个包含时间戳的唯一父目录
            evolve_parent_dir = os.path.join('runs', 'detect', 'evolve')
            # 注意：config['name']已经包含了时间戳，所以直接使用它作为父目录名
            evolve_parent_name = config.get('name', 'evolve')
            
            # 创建训练结果的专用目录结构：parent_dir/parent_name/train
            # 这样所有的train/train2/train3等文件夹都会在同一个父目录下
            evolve_project = os.path.join(evolve_parent_dir, evolve_parent_name)
            evolve_name = 'train'  # 使用固定的名称'train'，让ultralytics自动处理编号
            
            logger.info(f"超参数进化结果将保存在: {evolve_project}")
            logger.info(f"训练结果文件夹将按顺序创建为: {os.path.join(evolve_project, 'train')}, {os.path.join(evolve_project, 'train2')} 等")
            
            # 复制train_args并添加专用的project和name参数
            evolve_args = train_args.copy()
            evolve_args['project'] = evolve_project
            evolve_args['name'] = evolve_name
            evolve_args['exist_ok'] = False  # 确保每次迭代都创建新目录
            
            # 在调用model.tune()前重新应用matplotlib中文显示配置
            # 这是因为ultralytics在内部可能会重置matplotlib配置
            logger.info("重新应用matplotlib中文显示配置")
            
            # 使用增强版的中文显示修复函数
            fix_matplotlib_chinese()
            
            # 导入matplotlib并再次确认设置
            import matplotlib
            logger.info(f"当前matplotlib字体配置: {matplotlib.rcParams['font.sans-serif']}")
            
            # 使用model.tune()方法进行超参数进化，并指定专用路径
            # 使用space参数自定义超参数搜索空间，包括训练轮次
            # 在ultralytics 8.3.203版本中，这是正确的方式来指定参与进化的参数
            custom_space = {
                # 'epochs': (50, 300),  # 训练轮次范围从50到300
                'lr0': (0.0001, 0.01),  # 学习率范围
                'lrf': (0.01, 0.1),  # 新增：最终学习率比例
                'momentum': (0.7, 0.99),  # 动量范围
                'weight_decay': (0.0001, 0.001),  # 权重衰减范围
                'warmup_epochs': (0.0, 5.0),  # 预热轮次范围
                'warmup_momentum': (0.7, 0.95),  # 预热动量范围
                'warmup_bias_lr': (0.001, 0.1),  # 预热偏置学习率范围
                # 'batch': (4, 32),  # 批次大小 (注意：使用'batch'而不是'batch_size'以兼容ultralytics库)
                'mosaic': (0.0, 1.0),  # 新增：马赛克增强概率
                'mixup': (0.0, 0.8),  # 新增：混合增强概率
                'close_mosaic': (0, 20),  # 新增：最后N个epoch关闭马赛克
            }
            
            # 创建一个自定义转换器，确保epochs参数始终是整数
            # 这是因为ultralytics要求epochs必须是整数类型，但超参数进化过程中可能会生成浮点数
            class EpochsIntConverter:
                def __init__(self):
                    # 在ultralytics 8.3.203中，我们需要直接替换config中的epochs参数
                    # 因为参数验证发生在更早的阶段
                    pass
                    
                def ensure_int_epochs(self, hyperparams):
                    # 在使用超参数前，将epochs转换为整数
                    if 'epochs' in hyperparams and isinstance(hyperparams['epochs'], float):
                        original_epochs = hyperparams['epochs']
                        hyperparams['epochs'] = int(hyperparams['epochs'])
                        logger.info(f"已将浮点epochs值 {original_epochs} 转换为整数 {hyperparams['epochs']}")
                    return hyperparams
            
            # 初始化转换器
            epochs_converter = EpochsIntConverter()
            
            # 增强中文显示设置，确保在ultralytics库中也能正常显示中文
            global MATPLOTLIB_CHINESE_FIXED
            if not MATPLOTLIB_CHINESE_FIXED:
                try:
                    import matplotlib
                    # 强制应用中文字体设置
                    font_config = {
                        'font.family': ['sans-serif'],
                        'font.sans-serif': ['WenQuanYi Micro Hei', 'Heiti TC', 'SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'sans-serif'],
                        'axes.unicode_minus': False,
                        'text.usetex': False
                    }
                    matplotlib.rcParams.update(font_config)
                    # 尝试设置更多模块的字体
                    if hasattr(matplotlib, 'rc'):
                        matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': font_config['font.sans-serif']})
                    # 再次强制设置后端
                    matplotlib.use('Agg', force=True)
                    MATPLOTLIB_CHINESE_FIXED = True
                    logger.info("中文显示设置已增强")
                except Exception as e:
                    logger.warning(f"增强中文显示设置时出错: {e}")
            
            try:
                # 注意：ultralytics 8.3.203版本的model.tune()不支持直接传入population、mutation、crossover参数
                # 这些参数需要在创建模型时或通过其他方式设置
                
                # 由于ultralytics在内部会先验证参数类型，然后才会调用train方法
                # 我们需要修改tune方法的调用方式，在传递参数前确保epochs是整数
                # 这里我们需要创建一个自定义的回调函数来处理参数转换
                import functools
                
                # 保存原始的Tuner类的__call__方法
                from ultralytics.engine.tuner import Tuner
                original_tuner_call = Tuner.__call__
                
                # 包装__call__方法，确保在每次迭代时都将epochs转换为整数并重新配置中文字体
                @functools.wraps(original_tuner_call)
                def wrapped_tuner_call(self, model, iterations=30):
                    try:
                        # 保存原始的样本生成方法
                        if hasattr(self, 'sample'):
                            original_sample = self.sample
                            
                            @functools.wraps(original_sample)
                            def wrapped_sample(self):
                                # 生成样本
                                sample = original_sample()
                                # 确保epochs是整数
                                sample = epochs_converter.ensure_int_epochs(sample)
                                return sample
                            
                            # 替换样本生成方法
                            self.sample = functools.partial(wrapped_sample, self)
                        
                        # 在每次迭代调用前强制重新配置matplotlib的中文字体设置
                        logger.info("在超参数进化迭代前强制重新配置matplotlib中文字体设置")
                        reapply_chinese_fix(force=True)
                        
                        # 检查内存使用情况
                        memory_usage = psutil.virtual_memory().percent
                        if memory_usage > 80:
                            logger.warning(f"内存使用率过高 ({memory_usage}%)，可能导致训练中断")
                        
                        # 调用原始方法
                        return original_tuner_call(self, model, iterations)
                    except Exception as e:
                        logger.error(f"超参数调优过程中出现错误: {e}")
                        # 返回一个包含空结果的字典，避免后续处理时出错
                        return {'best_params': None}
                
                # 替换Tuner类的__call__方法
                Tuner.__call__ = wrapped_tuner_call
                
                # 执行超参数进化
                try:
                    results = model.tune(
                        data=config['data'],
                        iterations=config['evolve_iterations'],
                        space=custom_space,  # 使用space参数指定要进化的参数及其范围
                        **evolve_args
                    )
                except Exception as e:
                    logger.error(f"超参数进化过程中发生异常: {e}")
                    # 创建一个包含基本结果的字典，避免后续处理时出错
                    results = {'best_params': None}
                
            finally:
                # 恢复原始的Tuner.__call__方法
                try:
                    from ultralytics.engine.tuner import Tuner
                    if hasattr(Tuner, '__call__') and Tuner.__call__ != original_tuner_call:
                        Tuner.__call__ = original_tuner_call
                        logger.info("已恢复原始的Tuner.__call__方法")
                except Exception as e:
                    logger.error(f"恢复Tuner.__call__方法时出错: {e}")
            # 保存最优参数 - 增强的结果处理逻辑
            try:
                if results is not None:
                    # 即使results存在，也要确保best_params不是None
                    if 'best_params' in results and results['best_params'] is not None:
                        best_params_path = os.path.join('runs', 'detect', 'evolve', 'best_hyp.yaml')
                        os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
                        with open(best_params_path, 'w') as f:
                            yaml.dump(results['best_params'], f)
                        logger.info(f"最优超参数已保存到: {best_params_path}")
                    else:
                        logger.warning("超参数进化训练完成但未找到有效的最优参数")
                else:
                    logger.warning("超参数进化训练完成但未返回结果数据")
            except Exception as e:
                logger.error(f"处理和保存超参数进化结果时出错: {e}")
        else:
            # 开始常规训练，ultralytics会自动处理MLflow集成
            # 自动记录的内容包括：
            # - 训练参数（学习率、批次大小等）
            # - 每个epoch的指标（loss、mAP等）
            # - 模型工件（权重文件、配置文件等）
            results = model.train(data=config['data'], **train_args)
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        
        # 记录详细的错误信息，包括堆栈跟踪
        import traceback
        logger.error(f"错误堆栈跟踪:\n{traceback.format_exc()}")
        
        # 记录系统资源使用情况
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)
        logger.info(f"Memory usage: {memory_usage}%")
        logger.info(f"CPU usage: {cpu_usage}%")
        
        # 检查是否是由于内存不足导致的错误
        if memory_usage > 90:
            logger.warning("训练中断可能是由于内存不足导致的。建议减小批次大小或使用更轻量的模型。")
        
        # 尝试进行资源清理
        try:
            # 如果使用了GPU，可以尝试释放显存
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("已尝试释放GPU显存")
        except:
            pass
        
        logger.error("训练过程异常终止")
        
        # 不再重新抛出异常，而是以非零状态码退出，避免程序崩溃
        sys.exit(1)


if __name__ == '__main__':
    main()

# 使用示例:
# python train_yolo.py --config /path/to/config.yaml --epochs 100 --batch 32
# python train_yolo.py --config /path/to/config.yaml --model yolo11n.pt --epochs 50 --lr0 0.001 --device 0,1
# python train_yolo.py --config /path/to/config.yaml --model yolo11s.pt --disable-mlflow --cache --verbose
# nohup python train_yolo.py --config /path/to/config.yaml --model yolo11m.pt --epochs 200 >out_box.log 2>&1 &
# 启用超参数进化: python train_yolo.py --config /path/to/config.yaml --evolve --evolve-population 15 --evolve-generations 50
# 注：超参数进化功能在ultralytics 8.3.203版本中通过model.tune()实现
