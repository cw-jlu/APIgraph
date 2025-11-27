# APIgraph

累加训练至 2021 年 → 在 2022 年测试集上达到 **97.1% 准确率**（Ensemble）  
Drebin类似代码同条件仅 85.6% ！

## 核心贡献

| 贡献点                                 | 说明                                                                 |
|---------------------------------------|----------------------------------------------------------------------|
| 提出 **APIgraph** 表示方法            | 基于 API 调用序列构建有向图，固定 2000 维，天然抗概念漂移             |
| 严格的**时序实验设计**                | 2016~2021 年训练，2022 年测试，包含 单年训练 / 累加训练        |
| **目前效果优于Drebin**             | 相同训练条件下，准确率提升 **11.5%+**，F1 提升显著                   |
| 集成学习增强                          | SVM + Random Forest 软投票集成                      |
| 完整复现脚本 + 数据对齐逻辑          | 运行最终可得到对比图                                      |

## 实验结果一览（2022 测试集）

| 训练方式     | 方法                  | 2016 训练 | 累加至 2021 训练 |
|-------------|-----------------------|-----------|------------------|
| 单年训练     | Drebin-Fixed (ENS)    | 87.6%     | —                |
| 单年训练     | **APIgraph (ENS)**    | **97.1%** | —                |
| 累加训练     | Drebin-Fixed (ENS)    | —         | 85.6%            |
| 累加训练     | **APIgraph (ENS)**    | —         | **97.1%**        |

> 结论：**APIgraph 在跨年检测中展现出极强的时序鲁棒性，彻底解决了传统稀疏特征（如 Drebin）随时间快速失效的问题。**

## 免责声明
本项目仅用于学术研究、技术交流与学习目的，严禁用于任何非法或商业用途。
本项目部分代码和思路参考了以下经典工作：
Drebin（Drebin: Effective and Explainable Detection
of Android Malware in Your Pocket）及其公开实现
相关源代码https://github.com/alisakhatipova/Drebin
APIgraph(Enhancing State-of-the-art Classifiers with API Semantics to
 Detect Evolved Android Malware)及其公开实现相关源代码https://github.com/seclab-fudan/APIGraph

所使用的数据集来源于公开学术平台（AndroZoo[https://androzoo.uni.lu/]），仅用于非商业的学术研究。
项目中所有特征提取、模型训练、实验脚本均以现状形式提供，不提供任何明示或暗示的担保，包括但不限于准确性、完整性、适用性。
作者对因使用本项目代码、模型、数据或结果所导致的任何直接或间接损失不承担任何责任。
如您发现本仓库未经授权使用了您的代码、数据或模型，请立即通过 Issue 或邮箱联系我，我将在第一时间补充致谢或删除相关内容。

再次声明：本项目严禁用于任何恶意行为，包括但不限于绕过检测、制作恶意软件、非法传播等。一切法律责任与使用者无关。
感谢所有开源前辈与数据提供方！
## 数据准备说明
该项目中的download.py是下载AndroZoo的代码，若要从别的数据库下载，请重新实现此模块。使用AndroZoo前先与管理者取得联系，获取api密钥，更改download.py中的密钥并且把latest.csv文件放入根目录中
## 环境配置说明
Linux环境下的那一部分可能会有环境冲突，我自己写的只要把import的包全部install即可
## 你可能需要修改的地方
| 文件 | 变量 / 位置 | 含义 | 示例值 | 备注 |
|---|---|---|---|---|
| **auto_extract.py** | `years` | 要提取的年份列表 | `[2015, 2016, 2017]` | 按需增减 |
| **batch_decompile.py** | 末尾 `apk_dirs` | 待反编译的 APK 目录 | `[r'D:\data\2016', r'D:\data\2017']` | 支持批量 |
| **check_vector.py** | `X_mal` | 恶意软件向量路径 | `load(r'vectors/graph_vector\mal_2016.npy')` | 年份 / 文件名保持同步 |
|  | `X_ben` | 正常软件向量路径 | `load(r'vectors/graph_vector\ben_2016.npy')` | 同上 |
| **smali_extractor.py** | 末尾 `decompile_root` | 反编译结果根目录 | `r'D:\decompiled\2016'` | 与 2. 保持一致 |
| **download.py** | `YEARS` | 下载年份 | `[2016]` | 可多选 |
|  | `PER_YEAR` | 每年下载量 | `2000` |  |
|  | `THREADS` | 并发线程 | `16` |  |
|  | `THRESHOLD` | 病毒总分阈值 | `5` | VT 用 |
|  | `API_KEY` | VirusTotal 密钥 | `'yourkey'` | 支持列表循环 |
| **model.py** | `TRAIN_YEARS` | 训练集年份 | `[2015, 2016]` |  |
|  | `TEST_YEARS` | 测试集年份 | `[2017]` |  |
|  | `PARAMS` | 模型超参 | 见下方代码块 | 直接搜 `PARAMS = {` |

可能还有一些我没注意到的地方，主要是路径和参数可能需要修改
## 项目结构说明
  首先该项目包含两大模块，APIgraph和derbin，根目录下的download.py是用来自动下载apk的脚本，model.py则是对最后获得的vector里面的向量进行建模。
  decompile_apks是用来存放download_apks中的apk使用apktools反编译之后的产物。
  result_models是存放最后保存的模型文件。
  结果图片是保存最后结果，即model.py画的图。

  APIgraph：APIgraph目录下是基于原先的GitHub仓库我增加了一些东西，原先的仓库之提供了四个文件，在linux环境或者wsl上跑按执行顺序是getAllEntities.py（获取所有实体也就是每个节点，读取一些api文档api_json之类的东西）——getAllRelations.py（把每个节点的关系链接起来）——TransE.py（训练脚本，生成向量，产出res/method_entity_embedding_TransE.pkl）——clusterEmbedding.py（读上一步的 .pkl，做 K-means，产出res/method_cluster_mapping_2000.pkl）后面的为我们新增的文件都在windows环境下跑即可——map2000.py（抽查结果看看是否正常）——batch_decompile.py（调用apktool反编译下载的download_apk文件，获取smali文件）——smali_extractor.py（提取smali中的信息，然后把其运用TransE中的方法转化成向量）——check_vector.py（检查生成的向量看其是否能很好的映射到生成的map上，即两者提取的特征的交集大不大）——若无误后生成的这些npy文件就可以运用到model.py中进行训练，最终数据保存在vectors/graph_vector

  Derbin：原提供的feature_vector_extraction.py和classify.py都因为一些依赖太老（大概十年前的论文），而逐渐不可用，所以我重构了一下，classify.py的逻辑我放到了model.py中并且做了一些小的修改，而extraction_new.py和auto_extract.py则是对feature_vector_extraction.py的替代物，大体逻辑相同，做了一些小的修改，auto_extract.py主要是一个多次执行extraction_new.py的脚本，因为我们所需要的download_apk数据可能不只一年，最终我们得到了没有使用图，没有使用聚类的npy文件，特征向量，在vectors/direct_vector中


## 项目结构

```text
APIgraph/                              # 项目根目录
├── download.py                        # 自动从 AndroZoo 下载 APK
├── model.py                           # 核心实验脚本（训练 + 出8张图）
├── download_apks/                     # 下载的原始 APK（按年份+良恶性）
├── decompile_apks/                    # apktool 反编译后的 smali 文件
├── result_models/                     # 保存的模型文件（可选）
├── figures_zh/                        # 8张最终中文高清结果图
├── vectors/
│   ├── graph_vector/                  # APIgraph 向量（固定2000维 .npy）
│   └── direct_vector/                 # Drebin-Fixed 向量（维度随年增长）
├── res/                               # TransE 模型 & 聚类映射文件
├── 其他文件/                             
│
├── APIgraph/                          # APIgraph 主模块（核心创新）
│   ├── getAllEntities.py              # 【Linux/WSL】提取实体
│   ├── getAllRelations.py             # 【Linux/WSL】构建关系
│   ├── TransE.py                      # 【Linux/WSL】训练 TransE
│   ├── clusterEmbedding.py            # 【Linux/WSL】K-means 聚类到2000维
│   ├── map2000.py                     # 【Windows】检查聚类映射
│   ├── batch_decompile.py             # 【Windows】批量反编译
│   ├── smali_extractor.py             # 【Windows】smali → 向量
│   └── check_vector.py                # 【Windows】验证向量一致性
├── 其他文件/           
│
└── Drebin/                            # Drebin 重构模块
    ├── extraction_new.py              # 重写的特征提取脚本
    ├── auto_extract.py                # 批量提取多年前 Drebin 向量
    ├── (原 classify.py 逻辑已合并进 model.py)
    └── 其他文件/  


