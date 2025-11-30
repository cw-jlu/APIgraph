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

# 模型超参数
MODEL_PARAMS = {
    'svm': {'kernel': 'linear', 'C': 1.0, 'class_weight': 'balanced', 'probability': True, 'random_state': 42},
    'rf': {'n_estimators': 300, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
}

# 选择默认模型（这里选集成模型ENS，可改为SVM/RF）
DEFAULT_MODEL = 'ENS'

# ================= 工具函数 =================
def load_vector_with_split(dir_path, year, kind, data_split):
    """
    加载无Graph特征向量（需区分train/test子目录）
    :param dir_path: 无Graph特征根目录（含train/test）
    :param year: 年份（字符串）
    :param kind: 样本类型（mal/ben）
    :param data_split: 数据类型（train/test）
    :return: 特征向量数组
    """
    # 路径：data_root_no/split/前缀_年份.npy（如：D:\xxx\train\mal_2016.npy）
    prefix = CONFIG["vector_prefix"][kind]
    file_path = os.path.join(dir_path, data_split, f"{prefix}{year}.npy")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"无Graph特征文件不存在：{file_path}")
    vec = np.load(file_path)
    print(f"加载无Graph[{data_split}]：{file_path} | 样本数：{vec.shape[0]} | 维度：{vec.shape[1]}")
    return vec

def load_vector_no_split(dir_path, year, kind):
    """
    加载有Graph特征向量（无train/test区分）
    :param dir_path: 有Graph特征根目录
    :param year: 年份（字符串）
    :param kind: 样本类型（mal/ben）
    :return: 特征向量数组
    """
    prefix = CONFIG["vector_prefix"][kind]
    file_path = os.path.join(dir_path, f"{prefix}{year}.npy")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"有Graph特征文件不存在：{file_path}")
    vec = np.load(file_path)
    print(f"加载有Graph：{file_path} | 样本数：{vec.shape[0]} | 维度：{vec.shape[1]}")
    return vec

def prepare_train_data():
    """
    准备联合训练集（两种特征分别拼接所有训练年份）
    :return: (X_train_no, y_train_no) 无Graph训练集；(X_train_graph, y_train_graph) 有Graph训练集
    """
    print("\n===== 准备训练集（年份：{}）=====".format(",".join(CONFIG["train_years"])))
    
    # ---------------- 无Graph特征训练集（需区分train子目录）----------------
    X_train_no_mal = []
    X_train_no_ben = []
    for year in CONFIG["train_years"]:
        # 加载该年份训练集的恶意/良性样本
        mal_vec = load_vector_with_split(CONFIG["data_root_no"], year, "mal", "train")
        ben_vec = load_vector_with_split(CONFIG["data_root_no"], year, "ben", "train")
        X_train_no_mal.append(mal_vec)
        X_train_no_ben.append(ben_vec)
    
    # 拼接所有训练年份
    X_train_no_mal = np.vstack(X_train_no_mal) if X_train_no_mal else np.array([])
    X_train_no_ben = np.vstack(X_train_no_ben) if X_train_no_ben else np.array([])
    X_train_no = np.vstack([X_train_no_mal, X_train_no_ben]) if (X_train_no_mal.size and X_train_no_ben.size) else np.array([])
    y_train_no = np.concatenate([
        np.ones(X_train_no_mal.shape[0]),
        np.zeros(X_train_no_ben.shape[0])
    ])
    
    # ---------------- 有Graph特征训练集（无train/test区分）----------------
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
    
    # 输出训练集统计
    print(f"\n无Graph训练集：{X_train_no.shape[0]}个样本 | 维度：{X_train_no.shape[1]}")
    print(f"有Graph训练集：{X_train_graph.shape[0]}个样本 | 维度：{X_train_graph.shape[1]}")
    
    return (X_train_no, y_train_no), (X_train_graph, y_train_graph)

def prepare_test_year_data(year):
    """
    准备单个测试年份的数据集（两种特征）
    :param year: 测试年份（字符串）
    :return: (X_test_no, y_test_no) 无Graph测试集；(X_test_graph, y_test_graph) 有Graph测试集
    """
    print(f"\n===== 准备测试年份 {year} 的数据集 =====")
    
    # ---------------- 无Graph特征测试集（需区分test子目录）----------------
    X_test_no_mal = load_vector_with_split(CONFIG["data_root_no"], year, "mal", "test")
    X_test_no_ben = load_vector_with_split(CONFIG["data_root_no"], year, "ben", "test")
    X_test_no = np.vstack([X_test_no_mal, X_test_no_ben])
    y_test_no = np.concatenate([
        np.ones(X_test_no_mal.shape[0]),
        np.zeros(X_test_no_ben.shape[0])
    ])
    
    # ---------------- 有Graph特征测试集（无train/test区分）----------------
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

# ================= 主实验流程 =================
def main():
    # 1. 创建结果目录
    os.makedirs(CONFIG["out_root"], exist_ok=True)
    print(f"结果图片保存至：{CONFIG['out_root']}")
    
    # 2. 准备训练集（两种特征）
    (X_train_no, y_train_no), (X_train_graph, y_train_graph) = prepare_train_data()
    
    # 3. 初始化默认模型（按CONFIG选择）
    print(f"\n===== 初始化模型：{DEFAULT_MODEL} =====")
    if DEFAULT_MODEL == 'SVM':
        model_no = SVC(**MODEL_PARAMS['svm'])
        model_graph = SVC(**MODEL_PARAMS['svm'])
    elif DEFAULT_MODEL == 'RF':
        model_no = RandomForestClassifier(**MODEL_PARAMS['rf'])
        model_graph = RandomForestClassifier(**MODEL_PARAMS['rf'])
    elif DEFAULT_MODEL == 'ENS':
        model_no = VotingClassifier(
            estimators=[('svm', SVC(**MODEL_PARAMS['svm'])), ('rf', RandomForestClassifier(**MODEL_PARAMS['rf']))],
            voting='soft'
        )
        model_graph = VotingClassifier(
            estimators=[('svm', SVC(**MODEL_PARAMS['svm'])), ('rf', RandomForestClassifier(**MODEL_PARAMS['rf']))],
            voting='soft'
        )
    else:
        raise ValueError("默认模型仅支持 SVM/RF/ENS")
    
    # 4. 评估训练集性能（可选，用于对比）
    print("\n===== 评估训练集性能 =====")
    train_metrics_no = train_eval_model(model_no, X_train_no, y_train_no, X_train_no, y_train_no)
    train_metrics_graph = train_eval_model(model_graph, X_train_graph, y_train_graph, X_train_graph, y_train_graph)
    print(f"无Graph训练集 - 准确率：{train_metrics_no[0]:.4f} | F1：{train_metrics_no[1]:.4f}")
    print(f"有Graph训练集 - 准确率：{train_metrics_graph[0]:.4f} | F1：{train_metrics_graph[1]:.4f}")
    
    # 5. 按年份评估测试集性能
    print("\n===== 按年份评估测试集性能 =====")
    test_results = {
        '无Graph': {'准确率': [], 'F1值': [], '精确率': [], '召回率': []},
        '有Graph': {'准确率': [], 'F1值': [], '精确率': [], '召回率': []}
    }
    test_years = CONFIG["test_years"]
    
    for year in test_years:
        # 准备该年份测试集
        (X_test_no, y_test_no), (X_test_graph, y_test_graph) = prepare_test_year_data(year)
        
        # 评估无Graph模型
        metrics_no = train_eval_model(model_no, X_train_no, y_train_no, X_test_no, y_test_no)
        test_results['无Graph']['准确率'].append(metrics_no[0])
        test_results['无Graph']['F1值'].append(metrics_no[1])
        test_results['无Graph']['精确率'].append(metrics_no[2])
        test_results['无Graph']['召回率'].append(metrics_no[3])
        
        # 评估有Graph模型
        metrics_graph = train_eval_model(model_graph, X_train_graph, y_train_graph, X_test_graph, y_test_graph)
        test_results['有Graph']['准确率'].append(metrics_graph[0])
        test_results['有Graph']['F1值'].append(metrics_graph[1])
        test_results['有Graph']['精确率'].append(metrics_graph[2])
        test_results['有Graph']['召回率'].append(metrics_graph[3])
        
        # 输出该年份结果
        print(f"\n{year}年测试结果：")
        print(f"无Graph - 准确率：{metrics_no[0]:.4f} | F1：{metrics_no[1]:.4f} | 精确率：{metrics_no[2]:.4f} | 召回率：{metrics_no[3]:.4f}")
        print(f"有Graph - 准确率：{metrics_graph[0]:.4f} | F1：{metrics_graph[1]:.4f} | 精确率：{metrics_graph[2]:.4f} | 召回率：{metrics_graph[3]:.4f}")
    
    # 6. 整理训练集结果（用于绘图）
    train_results = {
        '无Graph': {'准确率': train_metrics_no[0], 'F1值': train_metrics_no[1], '精确率': train_metrics_no[2], '召回率': train_metrics_no[3]},
        '有Graph': {'准确率': train_metrics_graph[0], 'F1值': train_metrics_graph[1], '精确率': train_metrics_graph[2], '召回率': train_metrics_graph[3]}
    }
    
    # 7. 绘图（训练集+测试集各年份的有无Graph对比）
    plot_comparison(train_results, test_results, test_years)
    print(f"\n实验完成！所有图表已保存至：{CONFIG['out_root']}")

# ================= 结果可视化（修复f-string报错）=================
def plot_comparison(train_results, test_results, test_years):
    """
    绘制对比图：每个指标一张图，包含：
    - 训练集：无Graph vs 有Graph（单个柱状图）
    - 测试集：各年份无Graph vs 有Graph（折线图+标记点）
    """
    metrics = [
        ("准确率", 0),
        ("F1值", 1),
        ("精确率", 2),
        ("召回率", 3)
    ]
    # 绘图样式
    styles = {
        '无Graph': {'color': '#d62728', 'marker': 'o', 'line': '-', 'train_color': '#ff9999'},
        '有Graph': {'color': '#1f77b4', 'marker': 's', 'line': '--', 'train_color': '#99ccff'}
    }
    
    # 修复：用字符串拼接替代f-string换行（兼容所有Python3版本）
    train_years_str = ",".join(CONFIG['train_years'])
    train_label = "训练集\n(" + train_years_str + ")"  # 避免f-string换行
    
    # 所有x轴标签（训练集 + 测试集年份）
    all_x_labels = [train_label] + test_years
    # 训练集在x轴的位置（0），测试集年份位置（1,2,3...）
    train_x_pos = 0
    test_x_pos = list(range(1, len(test_years)+1))
    
    for metric_name, _ in metrics:
        plt.figure(figsize=(12, 7))
        
        # ---------------- 绘制训练集对比（柱状图）----------------
        # 无Graph训练集
        plt.bar(
            train_x_pos - 0.15,  # 左移0.15避免重叠
            train_results['无Graph'][metric_name],
            width=0.3,
            color=styles['无Graph']['train_color'],
            label='无Graph（训练集）',
            edgecolor=styles['无Graph']['color'],
            linewidth=2
        )
        # 有Graph训练集
        plt.bar(
            train_x_pos + 0.15,  # 右移0.15避免重叠
            train_results['有Graph'][metric_name],
            width=0.3,
            color=styles['有Graph']['train_color'],
            label='有Graph（训练集）',
            edgecolor=styles['有Graph']['color'],
            linewidth=2
        )
        
        # ---------------- 绘制测试集对比（折线图+标记点）----------------
        # 无Graph测试集
        plt.plot(
            test_x_pos,
            test_results['无Graph'][metric_name],
            color=styles['无Graph']['color'],
            marker=styles['无Graph']['marker'],
            markersize=10,
            linewidth=3,
            linestyle=styles['无Graph']['line'],
            label='无Graph（测试集）'
        )
        # 有Graph测试集
        plt.plot(
            test_x_pos,
            test_results['有Graph'][metric_name],
            color=styles['有Graph']['color'],
            marker=styles['有Graph']['marker'],
            markersize=10,
            linewidth=3,
            linestyle=styles['有Graph']['line'],
            label='有Graph（测试集）'
        )
        
        # ---------------- 图表美化与标注 ----------------
        # 添加数值标签（训练集）
        plt.text(
            train_x_pos - 0.15,
            train_results['无Graph'][metric_name] + 0.01,
            "{:.4f}".format(train_results['无Graph'][metric_name]),  # 兼容写法
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
        plt.text(
            train_x_pos + 0.15,
            train_results['有Graph'][metric_name] + 0.01,
            "{:.4f}".format(train_results['有Graph'][metric_name]),
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
        
        # 添加数值标签（测试集）
        for i, pos in enumerate(test_x_pos):
            plt.text(
                pos,
                test_results['无Graph'][metric_name][i] + 0.01,
                "{:.4f}".format(test_results['无Graph'][metric_name][i]),
                ha='center', va='bottom', fontsize=10, color=styles['无Graph']['color']
            )
            plt.text(
                pos,
                test_results['有Graph'][metric_name][i] - 0.02,
                "{:.4f}".format(test_results['有Graph'][metric_name][i]),
                ha='center', va='top', fontsize=10, color=styles['有Graph']['color']
            )
        
        # 修复：标题用字符串拼接替代f-string换行
        title_train = ",".join(CONFIG["train_years"])
        title_test = ",".join(CONFIG["test_years"])
        plt.title(metric_name + "对比分析\n模型：" + DEFAULT_MODEL + " | 训练集：" + title_train + " | 测试集：" + title_test,
                  fontsize=14, fontweight='bold', pad=20)
        
        plt.xlabel('数据类型与年份', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.xticks(range(len(all_x_labels)), all_x_labels, fontsize=11)
        plt.ylim(0.5, 1.02)  # 固定y轴范围，增强可读性
        plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        plt.legend(fontsize=11, loc='lower right', framealpha=0.9)
        
        # 保存图片
        save_path = os.path.join(CONFIG["out_root"], metric_name + "_有无Graph对比图.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"已保存图表：{save_path}")

# ================= 执行主程序 =================
if __name__ == "__main__":
    main()