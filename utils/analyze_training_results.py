import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import yaml

# 先导入中文显示修复模块，确保它在matplotlib之前加载
from utils.simple_fix_chinese import fix_matplotlib_chinese, enforce_chinese_fonts
# 应用中文显示修复
fix_matplotlib_chinese()
enforce_chinese_fonts()

# 现在再导入matplotlib，确保它使用正确的字体设置
import matplotlib.pyplot as plt

# 额外强制应用一次字体设置，确保所有组件都使用正确的字体
plt.rcParams.update({'font.family': ['sans-serif']})
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 训练运行配置
training_runs = [
    {
        'name': 'hyg_ncj_train_20251016174423_n100',
        'label': 'ncj_train_100轮',
        'color': '#1f77b4'
    },
    {
        'name': 'hyg_ncj_train_20251017141207',
        'label': 'ncj_train_新训练',
        'color': '#ff7f0e'
    }
]

def load_training_data(run):
    """加载单个训练运行的数据"""
    base_path = f"/root/hyg/projects/hushang_fridge_detect/{run['name']}"
    results_path = f"{base_path}/results.csv"
    args_path = f"{base_path}/args.yaml"
    
    try:
        # 加载结果数据
        df = pd.read_csv(results_path)
        
        # 加载配置数据
        with open(args_path, 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
        
        return df, args
    except Exception as e:
        print(f"加载 {run['name']} 数据失败: {e}")
        return None, None

def load_all_training_data():
    """加载所有训练运行的数据"""
    all_data = []
    for run in training_runs:
        df, args = load_training_data(run)
        if df is not None:
            all_data.append({
                'name': run['name'],
                'label': run['label'],
                'color': run['color'],
                'data': df,
                'args': args
            })
    return all_data

def extract_max_metrics(data_list, metrics):
    """提取每个训练运行的最大指标值"""
    max_metrics = []
    for item in data_list:
        max_values = {}
        for metric in metrics:
            try:
                # 过滤掉非数值和无穷大的值
                valid_values = item['data'][metric].replace([np.inf, -np.inf], np.nan).dropna()
                if not valid_values.empty:
                    max_values[metric] = valid_values.max()
                    # 找到最大值对应的轮次
                    max_idx = valid_values.idxmax()
                    max_values[f'{metric}_epoch'] = item['data'].iloc[max_idx]['epoch']
                else:
                    max_values[metric] = None
                    max_values[f'{metric}_epoch'] = None
            except Exception as e:
                print(f"提取 {item['name']} 的 {metric} 时出错: {e}")
                max_values[metric] = None
                max_values[f'{metric}_epoch'] = None
        
        max_metrics.append({
            'name': item['name'],
            'label': item['label'],
            'args': item['args'],
            **max_values
        })
    return max_metrics

def plot_metrics(data_list, metrics, title, ylabel, output_file):
    """绘制指标对比图"""
    plt.figure(figsize=(12, 6))
    
    for item in data_list:
        df = item['data']
        for metric in metrics:
            if metric in df.columns:
                # 过滤掉非数值和无穷大的值
                valid_data = df[[metric]].replace([np.inf, -np.inf], np.nan).dropna()
                if not valid_data.empty:
                    plt.plot(valid_data.index + 1, valid_data[metric], 
                             label=f"{item['label']} ({metric})")
    
    plt.title(title)
    plt.xlabel('训练轮次')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison_charts(data_list, output_dir):
    """生成各种对比图表"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制mAP50对比图
    plot_metrics(data_list, ['metrics/mAP50(B)'], 
                 '不同训练运行的mAP50对比', 'mAP50值', 
                 f'{output_dir}/mAP50_comparison.png')
    
    # 绘制mAP50-95对比图
    plot_metrics(data_list, ['metrics/mAP50-95(B)'], 
                 '不同训练运行的mAP50-95对比', 'mAP50-95值', 
                 f'{output_dir}/mAP50_95_comparison.png')
    
    # 绘制精确率对比图
    plot_metrics(data_list, ['metrics/precision(B)'], 
                 '不同训练运行的精确率对比', '精确率值', 
                 f'{output_dir}/precision_comparison.png')
    
    # 绘制召回率对比图
    plot_metrics(data_list, ['metrics/recall(B)'], 
                 '不同训练运行的召回率对比', '召回率值', 
                 f'{output_dir}/recall_comparison.png')
    
    # 绘制损失函数对比图
    plot_metrics(data_list, ['train/box_loss', 'val/box_loss'], 
                 '不同训练运行的框损失对比', '框损失值', 
                 f'{output_dir}/box_loss_comparison.png')
    
    plot_metrics(data_list, ['train/cls_loss', 'val/cls_loss'], 
                 '不同训练运行的分类损失对比', '分类损失值', 
                 f'{output_dir}/cls_loss_comparison.png')

def generate_summary_table(max_metrics, output_dir):
    """生成性能指标汇总表格"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备表格数据
    table_data = []
    for item in max_metrics:
        table_data.append({
            '训练运行': item['label'],
            '模型': item['args'].get('model', 'N/A'),
            '训练轮数': item['args'].get('epochs', 'N/A'),
            'Patience': item['args'].get('patience', 'N/A'),
            '最大mAP50': f"{item['metrics/mAP50(B)']:.4f} (轮 {int(item['metrics/mAP50(B)_epoch'])})" if item['metrics/mAP50(B)'] is not None else 'N/A',
            '最大mAP50-95': f"{item['metrics/mAP50-95(B)']:.4f} (轮 {int(item['metrics/mAP50-95(B)_epoch'])})" if item['metrics/mAP50-95(B)'] is not None else 'N/A',
            '最大精确率': f"{item['metrics/precision(B)']:.4f} (轮 {int(item['metrics/precision(B)_epoch'])})" if item['metrics/precision(B)'] is not None else 'N/A',
            '最大召回率': f"{item['metrics/recall(B)']:.4f} (轮 {int(item['metrics/recall(B)_epoch'])})" if item['metrics/recall(B)'] is not None else 'N/A',
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(table_data)
    
    # 保存为CSV
    csv_file = f"{output_dir}/performance_summary.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    # 保存为HTML
    html_file = f"{output_dir}/performance_summary.html"
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>训练性能汇总</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ text-align: center; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .best {{ color: green; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>训练性能汇总分析</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        {df.to_html(escape=False, index=False)}
    </body>
    </html>
    """
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return df

def find_best_performer(max_metrics):
    """找出各个指标的最佳模型"""
    metrics_to_compare = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']
    best_performers = {}
    
    for metric in metrics_to_compare:
        valid_items = [item for item in max_metrics if item[metric] is not None]
        if valid_items:
            best_item = max(valid_items, key=lambda x: x[metric])
            best_performers[metric] = {
                'label': best_item['label'],
                'value': best_item[metric],
                'epoch': best_item[f'{metric}_epoch']
            }
    
    return best_performers

def generate_analysis_report(data_list, max_metrics, best_performers, output_dir):
    """生成分析报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = f"{output_dir}/analysis_report.md"
    report_content = f"""
# 训练性能分析报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 概述

本报告对比分析了5个不同训练配置的YOLO模型训练结果，包括不同模型架构（yolo11n和yolo11s）、不同训练轮数（100轮和200轮）以及不同patience参数设置（80、90、100）。

## 2. 最佳性能模型

| 指标 | 最佳模型 | 最佳值 | 对应轮次 |
|------|----------|--------|----------|
| mAP50 | {best_performers.get('metrics/mAP50(B)', {}).get('label', 'N/A')} | {best_performers.get('metrics/mAP50(B)', {}).get('value', 'N/A'):.4f} | {int(best_performers.get('metrics/mAP50(B)', {}).get('epoch', 0))} |
| mAP50-95 | {best_performers.get('metrics/mAP50-95(B)', {}).get('label', 'N/A')} | {best_performers.get('metrics/mAP50-95(B)', {}).get('value', 'N/A'):.4f} | {int(best_performers.get('metrics/mAP50-95(B)', {}).get('epoch', 0))} |
| 精确率 | {best_performers.get('metrics/precision(B)', {}).get('label', 'N/A')} | {best_performers.get('metrics/precision(B)', {}).get('value', 'N/A'):.4f} | {int(best_performers.get('metrics/precision(B)', {}).get('epoch', 0))} |
| 召回率 | {best_performers.get('metrics/recall(B)', {}).get('label', 'N/A')} | {best_performers.get('metrics/recall(B)', {}).get('value', 'N/A'):.4f} | {int(best_performers.get('metrics/recall(B)', {}).get('epoch', 0))} |

## 3. 配置分析

### 3.1 模型架构对比

- **yolo11n vs yolo11s**: 分析不同模型大小对性能的影响

### 3.2 训练轮数对比

- **100轮 vs 200轮**: 分析训练轮数对性能的影响

### 3.3 Patience参数对比

- **patience=80/90/100**: 分析早停参数对训练稳定性的影响

## 4. 训练趋势分析

### 4.1 收敛速度

不同模型和配置的收敛速度差异分析

### 4.2 过拟合情况

通过比较训练损失和验证损失，分析各模型的过拟合程度

## 5. 建议

基于分析结果，提出以下建议：

1. **最佳配置推荐**：基于综合性能指标，推荐使用...

2. **超参数调整建议**：
   - 学习率调整
   - Batch size优化
   - 数据增强策略

3. **后续实验建议**：
   - 尝试其他模型架构
   - 调整数据分割比例
   - 增加正则化策略

## 6. 结论

总结分析结果，指出最佳训练配置及其优势。
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

def main():
    # 设置输出目录
    output_dir = f"/root/hyg/projects/hushang_fridge_detect/training_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("开始加载训练数据...")
    data_list = load_all_training_data()
    print(f"成功加载 {len(data_list)} 个训练运行的数据")
    
    # 提取关键指标
    metrics_to_extract = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']
    max_metrics = extract_max_metrics(data_list, metrics_to_extract)
    
    # 找出最佳模型
    best_performers = find_best_performer(max_metrics)
    
    # 生成对比图表
    print("生成对比图表...")
    plot_comparison_charts(data_list, output_dir)
    
    # 生成汇总表格
    print("生成性能汇总表格...")
    summary_df = generate_summary_table(max_metrics, output_dir)
    
    # 生成分析报告
    print("生成分析报告...")
    generate_analysis_report(data_list, max_metrics, best_performers, output_dir)
    
    print(f"分析完成！结果保存在: {output_dir}")
    print("\n最佳性能模型:")
    for metric, performer in best_performers.items():
        metric_name = metric.replace('metrics/', '').replace('(B)', '')
        print(f"- {metric_name}: {performer['label']} - {performer['value']:.4f} (轮 {int(performer['epoch'])})")

if __name__ == "__main__":
    main()