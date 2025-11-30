#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ================= 全局配置（根据需求修改）=================
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
# 
CONFIG = {
    "train_years": ["2016", "2017", "2018", "2019"],  # 训练集年份（字符串格式）
    "test_years": ["2020", "2021", "2022"],           # 测试集年份（字符串格式，按年评估）
    "data_root_no": r"D:\APIgraph\vectors\direct_vector",       # 无Graph特征根目录（包含train/test子目录）
    "data_root_graph": r"D:\APIgraph\vectors\graph_vector",  # 有Graph特征根目录（无train/test区分）
    "out_root": r"D:\APIgraph\output",  # 结果图片输出目录
    "vector_prefix": {"mal": "mal_", "ben": "ben_"}    # 向量文件前缀（恶意：mal_xxx.npy，良性：be_xxx.npy）
}

# 模型超参数（保持不变）
MODEL_PARAMS = {
    'svm': {'kernel': 'linear', 'C': 1.0, 'class_weight': 'balanced', 'probability': True, 'random_state': 42},
    'rf': {'n_estimators': 300, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
}

# 所有模型类型（用于批量训练和对比）
ALL_MODELS = ['SVM', 'RF', 'ENS']

# ================= 工具函数（保持不变）=================
def load_vector_with_split(dir_path, year, kind, data_split):
    """加载无Graph特征向量（需区分train/test子目录）"""
    prefix = CONFIG["vector_prefix"][kind]
    file_path = os.path.join(dir_path, data_split, f"{prefix}{year}.npy")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"无Graph特征文件不存在：{file_path}")
    vec = np.load(file_path)
    print(f"加载无Graph[{data_split}]：{file_path} | 样本数：{vec.shape[0]} | 维度：{vec.shape[1]}")
    return vec

def load_vector_no_split(dir_path, year, kind):
    """加载有Graph特征向量（无train/test区分）"""
    prefix = CONFIG["vector_prefix"][kind]
    file_path = os.path.join(dir_path, f"{prefix}{year}.npy")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"有Graph特征文件不存在：{file_path}")
    vec = np.load(file_path)
    print(f"加载有Graph：{file_path} | 样本数：{vec.shape[0]} | 维度：{vec.shape[1]}")
    return vec

def prepare_train_data():
    """准备联合训练集（两种特征分别拼接所有训练年份）"""
    print("\n===== 准备训练集（年份：{}）=====".format(",".join(CONFIG["train_years"])))
    
    # 无Graph特征训练集
    X_train_no_mal = []
    X_train_no_ben = []
    for year in CONFIG["train_years"]:
        mal_vec = load_vector_with_split(CONFIG["data_root_no"], year, "mal", "train")
        ben_vec = load_vector_with_split(CONFIG["data_root_no"], year, "ben", "train")
        X_train_no_mal.append(mal_vec)
        X_train_no_ben.append(ben_vec)
    
    X_train_no_mal = np.vstack(X_train_no_mal) if X_train_no_mal else np.array([])
    X_train_no_ben = np.vstack(X_train_no_ben) if X_train_no_ben else np.array([])
    X_train_no = np.vstack([X_train_no_mal, X_train_no_ben]) if (X_train_no_mal.size and X_train_no_ben.size) else np.array([])
    y_train_no = np.concatenate([
        np.ones(X_train_no_mal.shape[0]),
        np.zeros(X_train_no_ben.shape[0])
    ])
    
    # 有Graph特征训练集
    X_train_graph_mal = []
    X_train_graph_ben = []
    for year in CONFIG["train_years"]:
        mal_vec = load_vector_no_split(CONFIG["data_root_graph"], year, "mal")
        ben_vec = load_vector_no_split(CONFIG["data_root_graph"], year, "ben")
        X_train_graph_mal.append(mal_vec)
        X_train_graph_ben.append(ben_vec)
    
    X_train_graph_mal = np.vstack(X_train_graph_mal) if X_train_graph_mal else np.array([])
    X_train_graph_ben = np.vstack(X_train_graph_ben) if X_train_graph_ben else np.array([])
    X_train_graph = np.vstack([X_train_graph_mal, X_train_graph_ben]) if (X_train_graph_mal.size and X_train_graph_ben.size) else np.array([])
    y_train_graph = np.concatenate([
        np.ones(X_train_graph_mal.shape[0]),
        np.zeros(X_train_graph_ben.shape[0])
    ])
    
    print(f"\n无Graph训练集：{X_train_no.shape[0]}个样本 | 维度：{X_train_no.shape[1]}")
    print(f"有Graph训练集：{X_train_graph.shape[0]}个样本 | 维度：{X_train_graph.shape[1]}")
    
    return (X_train_no, y_train_no), (X_train_graph, y_train_graph)

def prepare_test_year_data(year):
    """准备单个测试年份的数据集（两种特征）"""
    print(f"\n===== 准备测试年份 {year} 的数据集 =====")
    
    # 无Graph特征测试集
    X_test_no_mal = load_vector_with_split(CONFIG["data_root_no"], year, "mal", "test")
    X_test_no_ben = load_vector_with_split(CONFIG["data_root_no"], year, "ben", "test")
    X_test_no = np.vstack([X_test_no_mal, X_test_no_ben])
    y_test_no = np.concatenate([
        np.ones(X_test_no_mal.shape[0]),
        np.zeros(X_test_no_ben.shape[0])
    ])
    
    # 有Graph特征测试集
    X_test_graph_mal = load_vector_no_split(CONFIG["data_root_graph"], year, "mal")
    X_test_graph_ben = load_vector_no_split(CONFIG["data_root_graph"], year, "ben")
    X_test_graph = np.vstack([X_test_graph_mal, X_test_graph_ben])
    y_test_graph = np.concatenate([
        np.ones(X_test_graph_mal.shape[0]),
        np.zeros(X_test_graph_ben.shape[0])
    ])
    
    print(f"无Graph测试集：{X_test_no.shape[0]}个样本 | 有Graph测试集：{X_test_graph.shape[0]}个样本")
    return (X_test_no, y_test_no), (X_test_graph, y_test_graph)

def train_eval_model(model, X_train, y_train, X_test, y_test):
    """训练模型并返回4个指标"""
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("警告：训练集或测试集为空，返回全0指标")
        return (0.0, 0.0, 0.0, 0.0)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return (
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, average='binary', zero_division=0),
        precision_score(y_test, y_pred, average='binary', zero_division=0),
        recall_score(y_test, y_pred, average='binary', zero_division=0)
    )

def create_model(model_type):
    """创建指定类型的模型"""
    if model_type == 'SVM':
        return SVC(**MODEL_PARAMS['svm'])
    elif model_type == 'RF':
        return RandomForestClassifier(**MODEL_PARAMS['rf'])
    elif model_type == 'ENS':
        return VotingClassifier(
            estimators=[('svm', SVC(**MODEL_PARAMS['svm'])), ('rf', RandomForestClassifier(**MODEL_PARAMS['rf']))],
            voting='soft'
        )
    else:
        raise ValueError("模型仅支持 SVM/RF/ENS")

# ================= 结果可视化（核心修改部分）=================
def plot_single_model_comparison(all_results, test_years):
    """
    单个模型的有无Graph对比图（每个模型一张图，包含4个指标子图）
    移除训练集对比，只展示测试集各年份对比
    """
    metrics = ["准确率", "F1值", "精确率", "召回率"]
    styles = {
        '无Graph': {'color': '#d62728', 'marker': 'o', 'line': '-', 'linewidth': 3, 'markersize': 10},
        '有Graph': {'color': '#1f77b4', 'marker': 's', 'line': '--', 'linewidth': 3, 'markersize': 10}
    }
    
    train_years_str = ",".join(CONFIG['train_years'])
    test_years_str = ",".join(CONFIG['test_years'])
    
    for model_type in ALL_MODELS:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # 绘制无Graph曲线
            ax.plot(
                test_years,
                all_results[model_type]['无Graph'][metric],
                color=styles['无Graph']['color'],
                marker=styles['无Graph']['marker'],
                linestyle=styles['无Graph']['line'],
                linewidth=styles['无Graph']['linewidth'],
                markersize=styles['无Graph']['markersize'],
                label='无Graph'
            )
            
            # 绘制有Graph曲线
            ax.plot(
                test_years,
                all_results[model_type]['有Graph'][metric],
                color=styles['有Graph']['color'],
                marker=styles['有Graph']['marker'],
                linestyle=styles['有Graph']['line'],
                linewidth=styles['有Graph']['linewidth'],
                markersize=styles['有Graph']['markersize'],
                label='有Graph'
            )
            
            # 添加数值标签
            for i, year in enumerate(test_years):
                # 无Graph数值（上方）
                ax.text(
                    i,
                    all_results[model_type]['无Graph'][metric][i] + 0.01,
                    f"{all_results[model_type]['无Graph'][metric][i]:.4f}",
                    ha='center', va='bottom', fontsize=9, color=styles['无Graph']['color'], fontweight='bold'
                )
                # 有Graph数值（下方）
                ax.text(
                    i,
                    all_results[model_type]['有Graph'][metric][i] - 0.02,
                    f"{all_results[model_type]['有Graph'][metric][i]:.4f}",
                    ha='center', va='top', fontsize=9, color=styles['有Graph']['color'], fontweight='bold'
                )
            
            # 子图美化
            ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
            ax.set_xlabel('测试年份', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_ylim(0.5, 1.02)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.legend(fontsize=11)
        
        # 总标题
        fig.suptitle(
            f'{model_type}模型 - 有无Graph特征对比\n训练集：{train_years_str} | 测试集：{test_years_str}',
            fontsize=16, fontweight='bold', y=0.98
        )
        
        # 保存图片
        save_path = os.path.join(CONFIG["out_root"], f'{model_type}_有无Graph对比图.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"已保存图表：{save_path}")

def plot_all_models_comparison(all_results, test_years):
    """
    三模型综合对比图（每个指标一张图，包含所有模型的有无Graph曲线）
    """
    metrics = ["准确率", "F1值", "精确率", "召回率"]
    # 模型样式配置（不同模型+有无Graph组合）
    model_styles = {
        'SVM-无Graph': {'color': '#d62728', 'marker': 'o', 'line': '-', 'linewidth': 2.5, 'markersize': 8},
        'SVM-有Graph': {'color': '#ff6b6b', 'marker': 'o', 'line': '-', 'linewidth': 2.5, 'markersize': 8},
        'RF-无Graph': {'color': '#1f77b4', 'marker': 's', 'line': '--', 'linewidth': 2.5, 'markersize': 8},
        'RF-有Graph': {'color': '#74b9ff', 'marker': 's', 'line': '--', 'linewidth': 2.5, 'markersize': 8},
        'ENS-无Graph': {'color': '#2ca02c', 'marker': '^', 'line': '-.', 'linewidth': 2.5, 'markersize': 8},
        'ENS-有Graph': {'color': '#74f974', 'marker': '^', 'line': '-.', 'linewidth': 2.5, 'markersize': 8}
    }
    
    train_years_str = ",".join(CONFIG['train_years'])
    test_years_str = ",".join(CONFIG['test_years'])
    
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        
        # 绘制所有模型的曲线
        for model_type in ALL_MODELS:
            for graph_type in ['无Graph', '有Graph']:
                label = f'{model_type}-{graph_type}'
                style = model_styles[label]
                
                plt.plot(
                    test_years,
                    all_results[model_type][graph_type][metric],
                    color=style['color'],
                    marker=style['marker'],
                    linestyle=style['line'],
                    linewidth=style['linewidth'],
                    markersize=style['markersize'],
                    label=label,
                    alpha=0.9
                )
        
        # 添加数值标签（只标注有Graph的模型，避免图表过于拥挤）
        for model_type in ALL_MODELS:
            for i, year in enumerate(test_years):
                value = all_results[model_type]['有Graph'][metric][i]
                style = model_styles[f'{model_type}-有Graph']
                plt.text(
                    i,
                    value + 0.005,
                    f"{value:.3f}",
                    ha='center', va='bottom', fontsize=8, color=style['color'], fontweight='bold'
                )
        
        # 图表美化
        plt.title(
            f'{metric} - 三模型综合对比\n训练集：{train_years_str} | 测试集：{test_years_str}',
            fontsize=15, fontweight='bold', pad=20
        )
        plt.xlabel('测试年份', fontsize=13)
        plt.ylabel(metric, fontsize=13)
        plt.ylim(0.5, 1.02)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.legend(fontsize=11, loc='lower right', framealpha=0.9, ncol=2)
        
        # 保存图片
        save_path = os.path.join(CONFIG["out_root"], f'三模型综合对比_{metric}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"已保存图表：{save_path}")

# ================= 主实验流程（修改为批量训练所有模型）=================
def main():
    # 1. 创建结果目录
    os.makedirs(CONFIG["out_root"], exist_ok=True)
    print(f"结果图片保存至：{CONFIG['out_root']}")
    
    # 2. 准备训练集（两种特征）
    (X_train_no, y_train_no), (X_train_graph, y_train_graph) = prepare_train_data()
    
    # 3. 初始化所有模型的结果存储结构
    all_results = {}
    for model_type in ALL_MODELS:
        all_results[model_type] = {
            '无Graph': {'准确率': [], 'F1值': [], '精确率': [], '召回率': []},
            '有Graph': {'准确率': [], 'F1值': [], '精确率': [], '召回率': []}
        }
    
    # 4. 按年份评估所有模型
    print("\n===== 按年份评估所有模型性能 =====")
    test_years = CONFIG["test_years"]
    
    for year in test_years:
        # 准备该年份测试集
        (X_test_no, y_test_no), (X_test_graph, y_test_graph) = prepare_test_year_data(year)
        
        # 评估每个模型的两种特征配置
        for model_type in ALL_MODELS:
            print(f"\n{year}年 - {model_type}模型评估：")
            
            # 初始化模型
            model = create_model(model_type)
            
            # 无Graph特征评估
            metrics_no = train_eval_model(model, X_train_no, y_train_no, X_test_no, y_test_no)
            all_results[model_type]['无Graph']['准确率'].append(metrics_no[0])
            all_results[model_type]['无Graph']['F1值'].append(metrics_no[1])
            all_results[model_type]['无Graph']['精确率'].append(metrics_no[2])
            all_results[model_type]['无Graph']['召回率'].append(metrics_no[3])
            
            # 有Graph特征评估
            metrics_graph = train_eval_model(model, X_train_graph, y_train_graph, X_test_graph, y_test_graph)
            all_results[model_type]['有Graph']['准确率'].append(metrics_graph[0])
            all_results[model_type]['有Graph']['F1值'].append(metrics_graph[1])
            all_results[model_type]['有Graph']['精确率'].append(metrics_graph[2])
            all_results[model_type]['有Graph']['召回率'].append(metrics_graph[3])
            
            # 输出该年份该模型结果
            print(f"  无Graph - 准确率：{metrics_no[0]:.4f} | F1：{metrics_no[1]:.4f} | 精确率：{metrics_no[2]:.4f} | 召回率：{metrics_no[3]:.4f}")
            print(f"  有Graph - 准确率：{metrics_graph[0]:.4f} | F1：{metrics_graph[1]:.4f} | 精确率：{metrics_graph[2]:.4f} | 召回率：{metrics_graph[3]:.4f}")
    
    # 5. 绘制可视化图表
    print("\n===== 生成可视化图表 =====")
    # 5.1 单个模型的有无Graph对比图（3张图：SVM、RF、ENS各一张）
    plot_single_model_comparison(all_results, test_years)
    # 5.2 三模型综合对比图（4张图：每个指标一张）
    plot_all_models_comparison(all_results, test_years)
    
    print(f"\n实验完成！所有图表已保存至：{CONFIG['out_root']}")
    print(f"生成的图表包括：")
    for model_type in ALL_MODELS:
        print(f"  - {model_type}_有无Graph对比图.png")
    for metric in ["准确率", "F1值", "精确率", "召回率"]:
        print(f"  - 三模型综合对比_{metric}.png")

# ================= 执行主程序 =================
if __name__ == "__main__":
    main()