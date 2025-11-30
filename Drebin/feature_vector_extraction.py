import os
import json
import numpy as np
from tqdm import tqdm
from androguard.misc import AnalyzeAPK
import logging

# ========================= 核心配置参数（重点！手动修改这里控制分批次处理）=========================
PROCESS_CONFIG = {
    "mode": "gen_test_vec",  # 处理模式（必填）：collect_train / build_dict / gen_train_vec / gen_test_vec
    "target_years": ["2022"],  # 目标年份（必填）：可填单个或多个年份，例如 ["2016"] 或 ["2020", "2021"]
    "feat_dict_path": None,  # 特征字典路径（生成向量时可选）：默认使用 out_root/feature_dict.npy
}

# ========================= 基础配置参数（根据需求修改）=========================
CONFIG = {
    "train_years": ["2016", "2017", "2018", "2019"],  # 所有训练集年份（供校验）
    "test_years": ["2020", "2021", "2022", "2023"],   # 所有测试集年份（供校验）
    "data_root": r"D:\latest\downloaded_apks",        # 数据集根目录
    "out_root": r"D:\latest",  # 输出根目录
    "api_json": "api.json",                           # Framework API列表
    "restricted_api": "restricted_api",               # 受限API列表
    "suspicious_api": "suspicious_api"                # 危险API列表
}

# 配置日志（记录跳过的APK信息）
logging.basicConfig(
    filename=os.path.join(CONFIG["out_root"], "extract_errors.log"),
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

import warnings
warnings.filterwarnings('ignore')

# 已知第三方库/SDK列表（过滤非应用自身API）
KNOWN_LIBS = [
    # 大型公司及基础SDK
    'android', 'com.android', 'com.google', 'com.facebook', 'com.adobe',
    'org.apache', 'com.amazon', 'com.amazonaws', 'com.dropbox', 'com.paypal',
    'twitter4j', 'mono', 'gnu',
    
    # 常用工具库
    'org.kobjects', 'com.squareup', 'com.appbrain', 'org.kxml2', 'org.slf4j',
    'org.jsoup', 'org.ksoap2', 'org.xmlpull', 'com.nineoldandroids',
    'com.actionbarsherlock', 'com.viewpagerindicator',
    'com.nostra13.universalimageloader', 'com.appyet',
    'com.fasterxml.jackson', 'org.anddev.andengine', 'org.andengine',
    'uk.co.senab.actionbarpulltorefresh', 'fr.castorflex.android.smoothprogressbar',
    'org.codehaus', 'org.acra', 'com.appmk', 'com.j256.ormlite', 'nl.siegmann.epublib',
    'pl.polidea', 'uk.co.senab', 'com.onbarcode', 'com.googlecode.apdfviewer',
    'com.badlogic.gdx', 'com.crashlytics', 'com.mobeta.android.dslv', 'com.andromo',
    'oauth.signpost', 'com.loopj.android.http', 'com.handmark.pulltorefresh.library',
    'com.bugsense.trace', 'org.cocos2dx.lib', 'com.esotericsoftware', 'javax.inject',
    'com.parse', 'org.joda.time', 'com.androidquery', 'crittercism.android',
    'biz.source_code.base64Coder', 'v2.com.playhaven', 'xmlwise', 'org.springframework',
    'org.scribe', 'org.opencv', 'org.dom4j', 'net.lingala.zip4j', 'jp.basicinc.gamefeat',
    'gnu.kawa', 'com.sun.mail', 'com.playhaven', 'com.commonsware.cwac', 'com.comscore',
    'com.koushikdutta', 'com.mapbar', 'greendroid', 'javax', 'org.intellij',
    
    # 广告SDK
    'com.millennialmedia', 'com.inmobi', 'com.revmob', 'com.mopub', 'com.admob',
    'com.flurry', 'com.adsdk', 'com.Leadbolt', 'com.adwhirl', 'com.airpush',
    'com.chartboost', 'com.pollfish', 'com.getjar', 'com.jb.gosms', 'com.sponsorpay',
    'net.nend.android', 'com.mobclix.android', 'com.tapjoy', 'com.adfonic.android',
    'com.applovin', 'com.adcenix', 'com.ad_stir', 'com.madhouse.android.ads',
    'com.waps', 'net.youmi.android', 'com.vpon.adon', 'cn.domob.android.ads',
    'com.wooboo.adlib_android', 'com.wiyun.ad',
    
    # 其他第三方库
    'com.jeremyfeinstein.slidingmenu.lib', 'com.slidingmenu.lib',
    'it.sephiroth.android.library', 'com.gtp.nextlauncher.library',
    'jp.co.nobot.libAdMaker', 'ch.boye.httpclientandroidlib', 'magmamobile.lib',
    'com.magmamobile'
]

# ========================= 工具函数（保持不变）=========================
def load_config_files(config):
    """加载配置文件并校验完整性"""
    required_files = [config["api_json"], config["restricted_api"], config["suspicious_api"]]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件缺失：{file_path}")
    
    # 加载Framework API（转换为更高效的查找结构）
    with open(config["api_json"], 'r', encoding='utf-8') as f:
        framework_api = json.load(f)
    # 转换为 {class_name: {method_name: True}} 格式，加速查找
    framework_api_dict = {}
    for cls, methods in framework_api.items():
        framework_api_dict[cls] = {method: True for method in methods}
    
    # 加载受限API和危险API
    with open(config["restricted_api"], 'r', encoding='utf-8') as f:
        susp_api = set(line.strip() for line in f if line.strip())
    with open(config["suspicious_api"], 'r', encoding='utf-8') as f:
        dang_api = set(line.strip() for line in f if line.strip())
    
    return framework_api_dict, susp_api, dang_api

def get_apk_file_paths(data_root, target_years, label):
    """
    获取指定年份和标签（malicious/benign）的所有APK文件路径
    :param data_root: 数据集根目录
    :param target_years: 要处理的目标年份列表（可指定部分年份）
    :param label: 标签（malicious/benign）
    :return: 字典 {年份: [APK路径列表]}
    """
    apk_paths = {}
    for year in target_years:
        dir_path = os.path.join(data_root, f"{label}_{year}")
        if not os.path.exists(dir_path):
            print(f"警告：{dir_path} 目录不存在，跳过该年份")
            apk_paths[year] = []
            continue
        
        # 获取目录下所有APK文件
        year_apks = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                     if f.lower().endswith(".apk") and os.path.isfile(os.path.join(dir_path, f))]
        apk_paths[year] = year_apks
        print(f"{year} {label} APK数量：{len(year_apks)}")
    return apk_paths

def extract_single_apk_features(apk_path, framework_api_dict, susp_api, dang_api):
    """
    提取单个APK的特征集合
    遇到异常直接返回None，外部处理跳过
    """
    try:
        a, d, dx = AnalyzeAPK(apk_path)
        # 兼容dex文件列表或单个dex文件
        dexs = d if isinstance(d, list) else [d]
        
        api_calls = set()
        for dex in dexs:
            api_calls.update(get_used_api(dex, framework_api_dict))
        
        intents = get_used_intents(a)
        hw_features = get_used_hw_features(a)
        permissions = a.get_permissions() or []
        receivers = a.get_receivers() or []
        services = a.get_services() or []
        providers = a.get_providers() or []
        activities = a.get_activities() or []
        
        # 构建特征集合
        features = set()
        # API调用特征
        for api in api_calls:
            if api in susp_api:
                features.add(f'api_call::{api}')
            if api in dang_api:
                features.add(f'call::{api}')
        # 其他特征
        for item in intents:
            features.add(f'intent::{item}')
        for item in hw_features:
            features.add(f'feature::{item}')
        for item in permissions:
            features.add(f'permission::{item}')
        for item in receivers:
            features.add(f'service_receiver::{item}')
        for item in services:
            features.add(f'service::{item}')
        for item in providers:
            features.add(f'provider::{item}')
        for item in activities:
            features.add(f'activity::{item}')
        
        return features
    
    except Exception as e:
        # 记录错误日志（包含APK路径和错误信息）
        logging.warning(f"提取APK特征失败：{apk_path} | 错误信息：{str(e)[:200]}")  # 限制错误信息长度
        return None  # 返回None表示提取失败

def get_used_api(dex, framework_api_dict):
    """提取APK中使用的Android Framework API（过滤第三方库）"""
    used_api = set()
    for method in dex.get_methods():
        if not method.get_code():
            continue
        class_name = method.get_class_name()[:-1]  # 去掉末尾的 ';'
        method_name = method.get_name()
        
        # 过滤已知第三方库
        if any(lib in class_name for lib in KNOWN_LIBS):
            continue
        
        # 只保留Android Framework API
        if class_name in framework_api_dict and method_name in framework_api_dict[class_name]:
            descriptor = method.get_descriptor()
            api_str = f"{class_name}->{method_name}{descriptor}"
            used_api.add(api_str)
    return used_api

def get_used_intents(apk):
    """从AndroidManifest.xml提取intent-filter"""
    intents = set()
    try:
        for intent_filter in apk.get_android_manifest_xml().getElementsByTagName("intent-filter"):
            for child in intent_filter.childNodes:
                if child.nodeType == child.ELEMENT_NODE and child.hasAttribute("android:name"):
                    intents.add(child.getAttribute("android:name"))
    except:
        pass
    return intents

def get_used_hw_features(apk):
    """从AndroidManifest.xml提取uses-feature"""
    features = set()
    try:
        for feature in apk.get_android_manifest_xml().getElementsByTagName("uses-feature"):
            if feature.hasAttribute("android:name"):
                features.add(feature.getAttribute("android:name"))
    except:
        pass
    return features

def features_to_vector(features_list, feature_dict):
    """将特征集合转换为二进制向量"""
    feature_size = len(feature_dict)
    vectors = []
    for features in features_list:
        vec = np.zeros(feature_size, dtype=np.uint8)
        for feat in features:
            if feat in feature_dict:
                vec[feature_dict[feat]] = 1
        vectors.append(vec)
    return np.array(vectors, dtype=np.uint8)

# ========================= 分批次处理函数 =========================
def collect_train_features(config, process_config):
    """
    单独执行：收集指定训练年份的特征（用于生成/补充特征字典）
    可多次执行，累计收集所有训练集特征
    """
    target_years = process_config["target_years"]
    print("\n" + "="*50)
    print(f"开始收集训练集特征 - 目标年份：{target_years}")
    print("="*50)
    
    # 加载配置文件
    framework_api_dict, susp_api, dang_api = load_config_files(config)
    
    # 获取指定年份的APK路径
    mal_apk_paths = get_apk_file_paths(config["data_root"], target_years, "malicious")
    ben_apk_paths = get_apk_file_paths(config["data_root"], target_years, "benign")
    
    # 加载已有的特征集合（如果存在）
    temp_features_path = os.path.join(config["out_root"], "temp_all_train_features.npy")
    if os.path.exists(temp_features_path):
        all_train_features = set(np.load(temp_features_path, allow_pickle=True))
        print(f"已加载历史特征：{len(all_train_features)} 个")
    else:
        all_train_features = set()
    
    # 收集恶意样本特征
    for year in target_years:
        apks = mal_apk_paths[year]
        if not apks:
            continue
        print(f"\n收集 {year} 恶意样本特征...")
        failed_count = 0
        for apk_path in tqdm(apks, desc=f"{year} 恶意样本"):
            features = extract_single_apk_features(apk_path, framework_api_dict, susp_api, dang_api)
            if features:
                all_train_features.update(features)
            else:
                failed_count += 1
        print(f"{year} 恶意样本处理完成：成功{len(apks)-failed_count}个，失败{failed_count}个")
    
    # 收集良性样本特征
    for year in target_years:
        apks = ben_apk_paths[year]
        if not apks:
            continue
        print(f"\n收集 {year} 良性样本特征...")
        failed_count = 0
        for apk_path in tqdm(apks, desc=f"{year} 良性样本"):
            features = extract_single_apk_features(apk_path, framework_api_dict, susp_api, dang_api)
            if features:
                all_train_features.update(features)
            else:
                failed_count += 1
        print(f"{year} 良性样本处理完成：成功{len(apks)-failed_count}个，失败{failed_count}个")
    
    # 保存累计的特征集合
    np.save(temp_features_path, list(all_train_features))
    print(f"\n累计收集特征数：{len(all_train_features)}")
    print(f"临时特征集合已保存至：{temp_features_path}")
    
    return all_train_features

def build_feature_dict(config):
    """
    单独执行：基于收集的所有训练集特征，构建并保存最终特征字典
    只需要执行一次（所有训练集特征收集完成后）
    """
    print("\n" + "="*50)
    print("开始构建最终特征字典")
    print("="*50)
    
    temp_features_path = os.path.join(config["out_root"], "temp_all_train_features.npy")
    if not os.path.exists(temp_features_path):
        raise FileNotFoundError("未找到临时特征集合，请先执行collect_train模式")
    
    # 加载所有训练集特征
    all_train_features = set(np.load(temp_features_path, allow_pickle=True))
    print(f"加载到训练集特征数：{len(all_train_features)}")
    
    # 构建并保存特征字典
    feature_list = sorted(all_train_features)
    feature_dict = {feat: idx for idx, feat in enumerate(feature_list)}
    
    feature_dict_path = os.path.join(config["out_root"], "feature_dict.npy")
    np.save(feature_dict_path, feature_dict)
    print(f"特征字典已保存至：{feature_dict_path}")
    print(f"最终特征维度：{len(feature_dict)}")
    
    return feature_dict

def generate_train_vectors(config, process_config):
    """
    单独执行：为指定训练年份生成向量（需要先有特征字典）
    支持分年份、分批次执行
    """
    target_years = process_config["target_years"]
    feat_dict_path = process_config["feat_dict_path"] or os.path.join(config["out_root"], "feature_dict.npy")
    
    print("\n" + "="*50)
    print(f"开始生成训练集向量 - 目标年份：{target_years}")
    print(f"使用特征字典：{feat_dict_path}")
    print("="*50)
    
    # 校验特征字典是否存在
    if not os.path.exists(feat_dict_path):
        raise FileNotFoundError(f"特征字典不存在：{feat_dict_path}，请先执行build_dict模式")
    feature_dict = np.load(feat_dict_path, allow_pickle=True).item()
    
    # 加载配置文件
    framework_api_dict, susp_api, dang_api = load_config_files(config)
    
    # 获取指定年份的APK路径
    mal_apk_paths = get_apk_file_paths(config["data_root"], target_years, "malicious")
    ben_apk_paths = get_apk_file_paths(config["data_root"], target_years, "benign")
    
    # 创建训练集输出目录
    out_dir = os.path.join(config["out_root"], "train")
    os.makedirs(out_dir, exist_ok=True)
    
    # 按年份生成向量
    for year in target_years:
        print(f"\n{'='*30} 处理 {year} 训练集 {'='*30}")
        
        # 处理恶意样本
        mal_apks = mal_apk_paths[year]
        if mal_apks:
            print(f"\n提取 {year} 恶意样本向量...")
            mal_features = []
            failed_count = 0
            for apk_path in tqdm(mal_apks, desc=f"{year} 恶意样本"):
                features = extract_single_apk_features(apk_path, framework_api_dict, susp_api, dang_api)
                if features:
                    mal_features.append(features)
                else:
                    failed_count += 1
            
            if mal_features:
                mal_vec = features_to_vector(mal_features, feature_dict)
                mal_save_path = os.path.join(out_dir, f"mal_{year}.npy")
                np.save(mal_save_path, mal_vec)
                print(f"{year} 恶意向量已保存：{mal_vec.shape} -> {mal_save_path}")
            print(f"{year} 恶意样本处理完成：成功{len(mal_apks)-failed_count}个，失败{failed_count}个")
        
        # 处理良性样本
        ben_apks = ben_apk_paths[year]
        if ben_apks:
            print(f"\n提取 {year} 良性样本向量...")
            ben_features = []
            failed_count = 0
            for apk_path in tqdm(ben_apks, desc=f"{year} 良性样本"):
                features = extract_single_apk_features(apk_path, framework_api_dict, susp_api, dang_api)
                if features:
                    ben_features.append(features)
                else:
                    failed_count += 1
            
            if ben_features:
                ben_vec = features_to_vector(ben_features, feature_dict)
                ben_save_path = os.path.join(out_dir, f"ben_{year}.npy")
                np.save(ben_save_path, ben_vec)
                print(f"{year} 良性向量已保存：{ben_vec.shape} -> {ben_save_path}")
            print(f"{year} 良性样本处理完成：成功{len(ben_apks)-failed_count}个，失败{failed_count}个")

def generate_test_vectors(config, process_config):
    """
    单独执行：为指定测试年份生成向量（需要先有特征字典）
    支持分年份、分批次执行
    """
    target_years = process_config["target_years"]
    feat_dict_path = process_config["feat_dict_path"] or os.path.join(config["out_root"], "feature_dict.npy")
    
    print("\n" + "="*50)
    print(f"开始生成测试集向量 - 目标年份：{target_years}")
    print(f"使用特征字典：{feat_dict_path}")
    print("="*50)
    
    # 校验特征字典是否存在
    if not os.path.exists(feat_dict_path):
        raise FileNotFoundError(f"特征字典不存在：{feat_dict_path}，请先执行build_dict模式")
    feature_dict = np.load(feat_dict_path, allow_pickle=True).item()
    
    # 加载配置文件
    framework_api_dict, susp_api, dang_api = load_config_files(config)
    
    # 获取指定年份的APK路径
    mal_apk_paths = get_apk_file_paths(config["data_root"], target_years, "malicious")
    ben_apk_paths = get_apk_file_paths(config["data_root"], target_years, "benign")
    
    # 创建测试集输出目录
    out_dir = os.path.join(config["out_root"], "test")
    os.makedirs(out_dir, exist_ok=True)
    
    # 按年份生成向量
    for year in target_years:
        print(f"\n{'='*30} 处理 {year} 测试集 {'='*30}")
        
        # 处理恶意样本
        mal_apks = mal_apk_paths[year]
        if mal_apks:
            print(f"\n提取 {year} 恶意测试样本向量...")
            mal_features = []
            failed_count = 0
            for apk_path in tqdm(mal_apks, desc=f"{year} 恶意测试样本"):
                features = extract_single_apk_features(apk_path, framework_api_dict, susp_api, dang_api)
                if features:
                    mal_features.append(features)
                else:
                    failed_count += 1
            
            if mal_features:
                mal_vec = features_to_vector(mal_features, feature_dict)
                mal_save_path = os.path.join(out_dir, f"mal_{year}.npy")
                np.save(mal_save_path, mal_vec)
                print(f"{year} 恶意测试向量已保存：{mal_vec.shape} -> {mal_save_path}")
            print(f"{year} 恶意测试样本处理完成：成功{len(mal_apks)-failed_count}个，失败{failed_count}个")
        
        # 处理良性样本
        ben_apks = ben_apk_paths[year]
        if ben_apks:
            print(f"\n提取 {year} 良性测试样本向量...")
            ben_features = []
            failed_count = 0
            for apk_path in tqdm(ben_apks, desc=f"{year} 良性测试样本"):
                features = extract_single_apk_features(apk_path, framework_api_dict, susp_api, dang_api)
                if features:
                    ben_features.append(features)
                else:
                    failed_count += 1
            
            if ben_features:
                ben_vec = features_to_vector(ben_features, feature_dict)
                ben_save_path = os.path.join(out_dir, f"ben_{year}.npy")
                np.save(ben_save_path, ben_vec)
                print(f"{year} 良性测试向量已保存：{ben_vec.shape} -> {ben_save_path}")
            print(f"{year} 良性测试样本处理完成：成功{len(ben_apks)-failed_count}个，失败{failed_count}个")

# ========================= 主函数（根据PROCESS_CONFIG执行）=========================
def main():
    # 创建输出根目录
    os.makedirs(CONFIG["out_root"], exist_ok=True)
    
    # 校验处理模式合法性
    valid_modes = ["collect_train", "build_dict", "gen_train_vec", "gen_test_vec"]
    if PROCESS_CONFIG["mode"] not in valid_modes:
        raise ValueError(f"无效的处理模式：{PROCESS_CONFIG['mode']}，可选模式：{valid_modes}")
    
    # 校验目标年份合法性
    all_available_years = CONFIG["train_years"] + CONFIG["test_years"]
    for year in PROCESS_CONFIG["target_years"]:
        if year not in all_available_years:
            print(f"警告：年份 {year} 不在配置的可用年份中，可能无对应数据")
    
    # 根据配置的模式执行对应操作
    if PROCESS_CONFIG["mode"] == "collect_train":
        # 收集训练集特征（可分批次执行）
        collect_train_features(CONFIG, PROCESS_CONFIG)
    
    elif PROCESS_CONFIG["mode"] == "build_dict":
        # 构建特征字典（所有训练集特征收集完成后执行）
        build_feature_dict(CONFIG)
    
    elif PROCESS_CONFIG["mode"] == "gen_train_vec":
        # 生成训练集向量（需先有特征字典）
        generate_train_vectors(CONFIG, PROCESS_CONFIG)
    
    elif PROCESS_CONFIG["mode"] == "gen_test_vec":
        # 生成测试集向量（需先有特征字典）
        generate_test_vectors(CONFIG, PROCESS_CONFIG)
    
    print("\n" + "="*50)
    print(f"当前模式任务完成！")
    print(f"输出目录：{os.path.abspath(CONFIG['out_root'])}")
    print("="*50)

if __name__ == "__main__":
    main()