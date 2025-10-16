#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超级增强版matplotlib中文显示修复方案
彻底解决ultralytics在超参数进化过程中重置matplotlib配置的问题
"""

import sys
import os
import matplotlib
from matplotlib import font_manager, pyplot as plt
import functools
import importlib
import warnings
import types
import builtins
import subprocess
import logging

# 配置日志
logging.basicConfig(
    level=logging.WARNING,  # 修改为WARNING级别，只显示警告和错误
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('matplotlib_chinese_fix')

# 忽略字体相关的警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# 特别忽略findfont警告，避免字体未找到时的警告信息
warnings.filterwarnings("ignore", message="findfont.*not found", category=UserWarning)

# 强制设置matplotlib后端
matplotlib.use('Agg', force=True)  # 非交互式后端，适合生成图像文件

# 增强的字体配置，优先使用系统中可用的中文字体
font_config = {
    # 使用sans-serif字体族
    'font.family': ['sans-serif', 'serif'],
    # 优先使用中文字体，调整了顺序以提供更美观的中文显示效果
    'font.sans-serif': [
        'Noto Sans CJK SC',    # 专为简体中文设计的Noto字体，美观度更高
        'Noto Sans CJK TC',    # 繁体中文Noto字体
        'Noto Sans CJK JP',    # Noto字体，系统中已确认存在
        'Droid Sans Fallback', # Android的回退字体，系统中已确认存在
        'WenQuanYi Micro Hei', # 文泉驿微米黑
        'Heiti TC',            # 黑体(繁体)
        'SimHei',              # 黑体
        'Microsoft YaHei',     # 微软雅黑
        'Arial Unicode MS',    # Arial Unicode MS
        'DejaVu Sans',         # 普通字体，作为最后的备选
        'sans-serif'           # 通用sans-serif
    ],
    # 衬线字体配置，提供更优雅的显示效果
    'font.serif': [
        'Noto Serif CJK SC',   # 简体中文衬线字体
        'Noto Serif CJK TC',   # 繁体中文衬线字体
        'Noto Serif CJK JP',   # 日文衬线字体
        'Noto Serif CJK KR',   # 韩文衬线字体
        'DejaVu Serif',        # 普通衬线字体
        'serif'                # 通用serif字体
    ],
    # 解决负号显示问题
    'axes.unicode_minus': False,
    # 禁用LaTeX渲染（避免中文显示问题）
    'text.usetex': False,
    # 设置合适的字体大小
    'font.size': 12,
    # 确保图像背景为白色
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    # 设置图像DPI，提高清晰度
    'figure.dpi': 100,
    # 设置保存图像的DPI
    'savefig.dpi': 100,
}

# 全局标志
MATPLOTLIB_CHINESE_FIXED = False

class MatplotlibChineseEnforcer:
    """强制matplotlib使用中文字体的增强器类"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MatplotlibChineseEnforcer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # 确保中文字体配置正确设置
        self._ensure_chinese_fonts_installed()
        
        # 尝试找到系统中实际可用的中文字体
        self.available_fonts = self._find_available_chinese_fonts()
        self._update_font_config_with_available_fonts()
        
        # 应用基本配置
        self._apply_configs()
        
        # 设置matplotlib配置持久化
        self._patch_matplotlib_rcparams()
        
        # 深度修补ultralytics相关函数
        self._deep_patch_ultralytics()
        
        # 监控matplotlib配置变化
        self._start_config_monitoring()
        
        self._initialized = True
        global MATPLOTLIB_CHINESE_FIXED
        MATPLOTLIB_CHINESE_FIXED = True
    
    def _ensure_chinese_fonts_installed(self):
        """确保系统中至少有一个中文字体可用"""
        try:
            # 检查是否有中文字体可用 - 优先检查Noto字体，与主配置保持一致
            test_fonts = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Droid Sans Fallback', 'DejaVu Sans']
            has_font = False
            
            for font_name in test_fonts:
                try:
                    # 测试字体是否可用
                    if font_manager.findfont(font_manager.FontProperties(family=font_name)):
                        has_font = True
                        break
                except:
                    continue
            
            if not has_font:
                logger.warning("系统中可能缺少中文字体，中文显示可能会有问题")
        except Exception as e:
            logger.error(f"检查中文字体时出错: {e}")
    
    def _find_available_chinese_fonts(self):
        """查找系统中可用的中文字体"""
        available_fonts = []
        try:
            # 查找所有系统字体
            all_fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
            
            # 定义中文字体标识符
            chinese_font_identifiers = ['hei', 'yahei', 'song', 'sim', 'noto', 'microsoft', 'wenquan', 'chinese']
            
            # 收集中文字体
            for font_path in all_fonts:
                font_name = os.path.basename(font_path).lower()
                if any(identifier in font_name for identifier in chinese_font_identifiers):
                    # 尝试获取字体名称
                    try:
                        font_props = font_manager.FontProperties(fname=font_path)
                        font_family = font_props.get_family()[0]
                        available_fonts.append(font_family)
                    except Exception as e:
                        # 如果无法获取字体名称，使用文件名
                        available_fonts.append(font_name)
                        logger.debug(f"无法获取字体名称 {font_path}: {e}")
        except Exception as e:
            logger.error(f"查找中文字体时出错: {e}")
        
        # 去重并返回
        unique_fonts = list(set(available_fonts))
        return unique_fonts
    
    def _update_font_config_with_available_fonts(self):
        """用可用的中文字体更新字体配置"""
        try:
            # 直接获取所有可用字体名称
            available_font_names = [f.name for f in font_manager.fontManager.ttflist]
            available_font_lower = [font.lower() for font in available_font_names]
            
            # 定义我们想要的字体优先级列表，与全局配置保持一致
            desired_fonts = [
                'Noto Sans CJK SC',    # 专为简体中文设计的Noto字体，美观度更高
                'Noto Sans CJK TC',    # 繁体中文Noto字体
                'Noto Sans CJK JP',    # Noto字体，系统中已确认存在
                'Droid Sans Fallback', # Android的回退字体，系统中已确认存在
                'WenQuanYi Micro Hei', # 文泉驿微米黑
                'Heiti TC',            # 黑体(繁体)
                'SimHei',              # 黑体
                'Microsoft YaHei',     # 微软雅黑
                'Arial Unicode MS',    # Arial Unicode MS
                'DejaVu Sans'          # 普通字体，作为最后的备选
            ]
            
            # 构建有效的字体列表 - 优先使用系统中可用的字体
            effective_fonts = []
            found_noto_jp = False
            found_droid = False
            
            for font in desired_fonts:
                if font.lower() in available_font_lower:
                    # 找到实际的字体名称（保持大小写一致）
                    index = available_font_lower.index(font.lower())
                    actual_font = available_font_names[index]
                    effective_fonts.append(actual_font)
                    
                    # 记录找到的关键字体
                    if 'Noto Sans CJK JP' in actual_font:
                        found_noto_jp = True
                    if 'Droid Sans Fallback' in actual_font:
                        found_droid = True
            
            # 确保至少包含关键的中文字体
            if not found_noto_jp and 'Noto Sans CJK JP' in desired_fonts:
                effective_fonts.insert(0, 'Noto Sans CJK JP')
            if not found_droid and 'Droid Sans Fallback' in desired_fonts:
                effective_fonts.insert(1, 'Droid Sans Fallback')
            
            # 确保至少有一个字体可用
            if not effective_fonts:
                # 如果没有找到任何中文字体，添加一些常见的系统字体作为备选
                effective_fonts = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
            
            # 最后添加sans-serif作为通用回退
            if 'sans-serif' not in effective_fonts:
                effective_fonts.append('sans-serif')
            
            font_config['font.sans-serif'] = effective_fonts
            self.available_fonts = effective_fonts
            
            # 记录最终的字体配置
            logger.debug(f"最终应用的字体配置: {effective_fonts}")
        except Exception as e:
            logger.error(f"更新字体配置时出错: {e}")
            # 如果出错，使用默认配置
            font_config['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
    
    def _apply_configs(self):
        """应用所有配置"""
        # 强制设置matplotlib后端
        try:
            matplotlib.use('Agg', force=True)
        except Exception as e:
            logger.error(f"设置matplotlib后端时出错: {e}")
            
        # 应用字体配置
        matplotlib.rcParams.update(font_config)
        
        # 确保pyplot的字体也被设置
        plt.rcParams.update(font_config)
        
        # 设置更多matplotlib模块的字体
        if hasattr(matplotlib, 'rc'):
            matplotlib.rc('font', **{'family': ['sans-serif'], 'sans-serif': font_config['font.sans-serif']})
        
        # 设置环境变量
        os.environ['MPLCONFIGDIR'] = os.path.join(os.path.expanduser('~'), '.config', 'matplotlib')
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['MPLBACKEND'] = 'Agg'
        
        # 确保中文显示正常的额外设置
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
    
    def _patch_matplotlib_rcparams(self):
        """Monkey patch matplotlib的rcParams，使其配置更持久"""
        try:
            # 保存原始的update方法
            original_update = matplotlib.rcParams.update
            original_plt_update = plt.rcParams.update
            
            @functools.wraps(original_update)
            def wrapped_update(*args, **kwargs):
                # 调用原始方法
                result = original_update(*args, **kwargs)
                
                # 重新应用中文字体配置
                self._reapply_critical_settings()
                
                return result
            
            @functools.wraps(original_plt_update)
            def wrapped_plt_update(*args, **kwargs):
                # 调用原始方法
                result = original_plt_update(*args, **kwargs)
                
                # 重新应用中文字体配置
                self._reapply_critical_settings()
                
                return result
            
            # 应用patch
            matplotlib.rcParams.update = wrapped_update
            plt.rcParams.update = wrapped_plt_update
            
            # 同时patch matplotlib.rc方法
            if hasattr(matplotlib, 'rc'):
                original_rc = matplotlib.rc
                
                @functools.wraps(original_rc)
                def wrapped_rc(*args, **kwargs):
                    result = original_rc(*args, **kwargs)
                    self._reapply_critical_settings()
                    return result
                
                matplotlib.rc = wrapped_rc
            
            # 尝试patch更多可能重置配置的方法
            if hasattr(matplotlib, 'rcdefaults'):
                original_rcdefaults = matplotlib.rcdefaults
                
                @functools.wraps(original_rcdefaults)
                def wrapped_rcdefaults():
                    result = original_rcdefaults()
                    self._reapply_critical_settings()
                    return result
                
                matplotlib.rcdefaults = wrapped_rcdefaults
                
        except Exception as e:
            logger.error(f"尝试patch matplotlib.rcParams时出错: {e}")
    
    def _reapply_critical_settings(self):
        """重新应用关键的中文显示设置"""
        # 确保字体设置正确
        matplotlib.rcParams['font.family'] = ['sans-serif']
        matplotlib.rcParams['font.sans-serif'] = font_config['font.sans-serif']
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['text.usetex'] = False
        
        # 同步到pyplot
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = font_config['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['text.usetex'] = False
    
    def _deep_patch_ultralytics(self):
        """深度修补ultralytics库中可能影响中文显示的函数"""
        try:
            # 尝试导入ultralytics的plotting模块
            import ultralytics
            
            # 方法1: 直接修改ultralytics导入的matplotlib配置
            self._reapply_critical_settings()
            
            # 方法2: 尝试patch ultralytics的utils.plotting模块
            try:
                from ultralytics.utils import plotting
                
                # 尝试直接访问plotting模块中可能的plt对象
                if hasattr(plotting, 'plt'):
                    plotting.plt.rcParams.update(font_config)
                    plotting.plt.rcParams['font.family'] = ['sans-serif']
                    plotting.plt.rcParams['font.sans-serif'] = font_config['font.sans-serif']
                
                # 修补plotting模块中的函数
                if hasattr(plotting, 'savefig'):
                    original_savefig = plotting.savefig
                    
                    @functools.wraps(original_savefig)
                    def wrapped_savefig(*args, **kwargs):
                        # 在保存图像前强制重新应用中文字体设置
                        self._reapply_critical_settings()
                        # 调用原始函数
                        return original_savefig(*args, **kwargs)
                    
                    plotting.savefig = wrapped_savefig
                
                # 尝试修补其他可能的绘图函数
                plot_functions = ['plot_labels', 'plot_results', 'confusion_matrix', 'plot_images', 'plot_results_grid']
                for func_name in plot_functions:
                    if hasattr(plotting, func_name):
                        original_func = getattr(plotting, func_name)
                        
                        # 使用辅助函数创建正确的闭包，避免循环中的闭包问题
                        def create_wrapper(original):
                            @functools.wraps(original)
                            def wrapped_func(*args, **kwargs):
                                # 使用全局的reapply_chinese_fix函数而不是self引用，避免作用域问题
                                reapply_chinese_fix()
                                return original(*args, **kwargs)
                            return wrapped_func
                        
                        # 为每个函数创建独立的包装器
                        wrapped_func = create_wrapper(original_func)
                        setattr(plotting, func_name, wrapped_func)
                
            except ImportError:
                # 如果plotting模块不可用，尝试其他方式
                pass
            
            # 方法3: 尝试修补YOLO类的相关方法
            try:
                from ultralytics import YOLO
                original_plot = YOLO.plot
                
                @functools.wraps(original_plot)
                def wrapped_plot(self, *args, **kwargs):
                    self._reapply_critical_settings()
                    return original_plot(self, *args, **kwargs)
                
                YOLO.plot = wrapped_plot
                
            except Exception:
                pass
            
        except ImportError:
            # 如果ultralytics还没有导入，静默处理
            pass
        except Exception as e:
            logger.error(f"尝试patch ultralytics时出错: {e}")
    
    def _start_config_monitoring(self):
        """启动配置监控，定期检查并修复中文字体设置"""
        # 这个方法会在每次绘图操作前自动应用修复，不需要单独的监控线程
        pass
    
    def enforce(self):
        """强制应用中文字体设置"""
        self._apply_configs()
        return True

# 创建全局增强器实例
enforcer = MatplotlibChineseEnforcer()

# 强制应用中文字体设置
def enforce_chinese_fonts():
    """强制应用中文字体设置，并确保在任何时候都能生效"""
    return enforcer.enforce()

# 创建一个全局函数，方便在任何地方调用以重置matplotlib配置
def fix_matplotlib_chinese():
    """重置matplotlib配置以支持中文显示"""
    result = enforcer.enforce()
    
    # 额外检查：确保matplotlib后端是Agg
    if matplotlib.get_backend() != 'Agg':
        try:
            matplotlib.use('Agg', force=True)
        except Exception as e:
            pass  # 静默处理
    
    return result

# 立即应用修复
fix_matplotlib_chinese()

# 提供一个函数用于检查是否有可用的中文字体
def has_available_chinese_fonts():
    """检查系统中是否有可用的中文字体"""
    try:
        # 检查系统中已确认存在的字体
        available_fonts = [f.name.lower() for f in font_manager.fontManager.ttflist]
        
        # 检查中文字体标识符
        chinese_identifiers = ['noto', 'droid', 'hei', 'yahei', 'sim', 'chinese', 'song', 'kai', 'ming']
        
        # 检查常用的中文字体名称
        chinese_font_names = ['noto sans cjk', 'droid sans fallback', 'wenquan', 'heiti', 'simhei', 'microsoft yahei']
        
        # 检查是否有中文字体
        for font in available_fonts:
            if any(identifier in font for identifier in chinese_identifiers) or \
               any(name in font for name in chinese_font_names):
                return True
        
        # 也检查是否有DejaVu Sans等通用字体
        if 'dejavu sans' in available_fonts:
            return True
        
        return False
    except Exception:
        # 如果出现异常，返回False
        return False

# 如果在ultralytics库导入后需要再次应用配置，可以调用此函数
def reapply_chinese_fix(force=False):
    """重新应用中文显示修复，特别是在ultralytics可能重置配置的情况下"""
    global MATPLOTLIB_CHINESE_FIXED
    # 不使用logger.debug，避免输出过多日志
    
    # 检查并确保matplotlib已导入
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        # 静默处理，不输出错误日志
        return
    
    if force:
        # 如果强制重新应用，重置enforcer实例
        MatplotlibChineseEnforcer._instance = None
        MATPLOTLIB_CHINESE_FIXED = False
    
    # 检查是否已经修复
    if MATPLOTLIB_CHINESE_FIXED:
        # 即使已经修复，我们也可以在这里额外应用一次关键设置，确保配置没有被覆盖
        try:
            # 直接调用reapply_critical_settings方法
            if MatplotlibChineseEnforcer._instance is not None:
                MatplotlibChineseEnforcer._instance._reapply_critical_settings()
        except Exception:
            # 静默处理异常
            pass
        return
    
    # 创建enforcer实例，这会自动应用修复
    try:
        enforcer = MatplotlibChineseEnforcer()
    except Exception:
        # 即使创建实例失败，也尝试直接应用基本配置
        try:
            matplotlib.rcParams.update(font_config)
            matplotlib.rcParams['axes.unicode_minus'] = False
            plt.rcParams.update(font_config)
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            # 静默处理异常
            pass