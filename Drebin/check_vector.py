import os
import numpy as np
from typing import Dict, List, Tuple

# ========================= 配置参数（根据你的实际路径修改）=========================
CONFIG = {
    "out_root": r"D:\APIgraph\vectors\direct_vector",  # 与特征提取代码的out_root保持一致
    "train_dir": "train",      # 训练集向量目录（默认无需修改）
    "test_dir": "test"         # 测试集向量目录（默认无需修改）
}

# ========================= 核心检查函数 =========================
def load_vector_files(root_dir: str, sub_dir: str) -> Dict[str, np.ndarray]:
    """
    加载指定目录下的所有向量文件（.npy格式）
    :param root_dir: 输出根目录
    :param sub_dir: 子目录（train/test）
    :return: 字典 {文件名: 向量数组}
    """
    vector_dir = os.path.join(root_dir, sub_dir)
    if not os.path.exists(vector_dir):
        print(f"警告：{vector_dir} 目录不存在，无对应向量文件")
        return {}
    
    # 查找所有.npy文件（仅匹配 mal_xxxx.npy 和 ben_xxxx.npy 格式）
    vector_files = {}
    for filename in os.listdir(vector_dir):
        if filename.endswith(".npy") and (filename.startswith("mal_") or filename.startswith("ben_")):
            file_path = os.path.join(vector_dir, filename)
            try:
                # 加载向量文件
                vector = np.load(file_path, allow_pickle=False)
                vector_files[filename] = vector
            except Exception as e:
                print(f"警告：加载文件 {filename} 失败 | 错误：{str(e)}")
    
    return vector_files

def analyze_vector_stats(vector_files: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
    """
    分析向量文件的统计信息
    :param vector_files: 向量文件字典
    :return: (单个文件统计, 整体统计)
    """
    file_stats = {}
    all_shapes = []  # 存储所有向量的shape (样本数, 特征维度)
    all_dtypes = set()  # 存储所有向量的数据类型
    
    for filename, vector in vector_files.items():
        # 单个文件的统计信息
        n_samples = vector.shape[0]  # 样本数量
        n_features = vector.shape[1] if len(vector.shape) >= 2 else 0  # 特征维度（确保是2D数组）
        dtype = str(vector.dtype)  # 数据类型
        sparsity = (vector == 0).sum() / vector.size * 100  # 稀疏度（0值占比，二进制向量通常接近100%）
        
        file_stats[filename] = {            "样本数量": n_samples,
            "特征维度": n_features,
            "数据类型": dtype,
            "稀疏度(%)": round(sparsity, 2),
            "数组形状": vector.shape,
            "文件路径": os.path.abspath(os.path.join(CONFIG["out_root"], 
                                                   "train" if "train" in filename else "test", 
                                                   filename))
        }
        
        # 累计整体统计
        if len(vector.shape) == 2:  # 只统计2D向量（样本数×特征维度）
            all_shapes.append(vector.shape)
            all_dtypes.add(dtype)
    
    # 整体统计
    overall_stats = {}
    if all_shapes:
        # 检查所有向量的特征维度是否一致（关键验证！）
        feature_dims = [shape[1] for shape in all_shapes]
        dim_consistent = len(set(feature_dims)) == 1  # 所有向量的特征维度是否相同
        
        overall_stats = {
            "总文件数": len(vector_files),
            "总样本数": sum(shape[0] for shape in all_shapes),
            "特征维度是否一致": dim_consistent,
            "统一特征维度": feature_dims[0] if dim_consistent else "不一致",
            "数据类型": list(all_dtypes),
            "各文件样本数": {os.path.basename(f): s[0] for f, s in zip(vector_files.keys(), all_shapes)},
            "各文件特征维度": {os.path.basename(f): s[1] for f, s in zip(vector_files.keys(), all_shapes)}
        }
    else:
        overall_stats = {
            "总文件数": 0,
            "总样本数": 0,
            "特征维度是否一致": False,
            "统一特征维度": "无有效向量",
            "数据类型": [],
            "各文件样本数": {},
            "各文件特征维度": {}
        }
    
    return file_stats, overall_stats

def print_vector_analysis(stats_type: str, file_stats: Dict, overall_stats: Dict):
    """
    格式化打印向量分析结果
    :param stats_type: 类型（训练集/测试集）
    :param file_stats: 单个文件统计
    :param overall_stats: 整体统计
    """
    print("\n" + "="*80)
    print(f"【{stats_type}向量特征分析结果】")
    print("="*80)
    
    # 打印整体统计
    print("\n【整体统计信息】")
    for key, value in overall_stats.items():
        print(f"  {key}: {value}")
    
    # 打印单个文件详细信息
    print("\n【单个文件详细信息】")
    if file_stats:
        for filename, stats in sorted(file_stats.items()):  # 按文件名排序
            print(f"\n  文件名：{filename}")
            for key, value in stats.items():
                print(f"    {key}: {value}")
    else:
        print("  无有效向量文件")
    
    # 关键提示
    if overall_stats["特征维度是否一致"]:
        print(f"\n✅ 验证通过：所有{stats_type}向量的特征维度一致（{overall_stats['统一特征维度']}维）")
    else:
        print(f"\n❌ 警告：{stats_type}向量特征维度不一致！请检查特征提取过程是否有误")

# ========================= 主函数 =========================
def main():
    print("="*80)
    print("APK向量特征检查工具")
    print("="*80)
    print(f"当前配置：输出根目录 = {CONFIG['out_root']}")
    print("="*80)
    
    # 1. 加载训练集向量并分析
    train_vectors = load_vector_files(CONFIG["out_root"], CONFIG["train_dir"])
    train_file_stats, train_overall_stats = analyze_vector_stats(train_vectors)
    print_vector_analysis("训练集", train_file_stats, train_overall_stats)
    
    # 2. 加载测试集向量并分析
    test_vectors = load_vector_files(CONFIG["out_root"], CONFIG["test_dir"])
    test_file_stats, test_overall_stats = analyze_vector_stats(test_vectors)
    print_vector_analysis("测试集", test_file_stats, test_overall_stats)
    
    # 3. 跨集一致性检查（训练集和测试集的特征维度是否一致）
    print("\n" + "="*80)
    print("【跨集一致性检查】")
    print("="*80)
    train_dim = train_overall_stats["统一特征维度"]
    test_dim = test_overall_stats["统一特征维度"]
    
    if train_dim != "无有效向量" and test_dim != "无有效向量":
        if train_dim == test_dim:
            print(f"✅ 训练集与测试集特征维度一致：{train_dim}维")
        else:
            print(f"❌ 警告：训练集（{train_dim}维）与测试集（{test_dim}维）特征维度不一致！")
            print("   原因可能：测试集使用了不同的特征字典，或特征提取过程有误")
    else:
        print("⚠️  无法进行跨集一致性检查（训练集/测试集无有效向量）")
    
    print("\n" + "="*80)
    print("检查完成！")
    print("="*80)

if __name__ == "__main__":
    main()