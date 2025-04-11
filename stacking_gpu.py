import os
import random
import numpy as np
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import shap
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import json
import time
import warnings

# 导入模型框架
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV as RSCV
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, make_scorer, precision_score, recall_score

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV as RSCV

from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import csv

from sklearn.preprocessing import LabelEncoder, StandardScaler
from rdkit.Chem import Descriptors
from sklearn.feature_selection import SelectFromModel

# 添加用于模型堆叠的库
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.base import clone
from itertools import combinations
from tqdm import tqdm

# 忽略LightGBM的警告
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# 添加检查点保存和加载功能
def save_checkpoint(checkpoint_data, filepath='/root/autodl-fs/stacking_checkpoint.pkl'):
    """
    保存检查点数据
    
    参数:
    checkpoint_data: 检查点数据字典
    filepath: 保存路径
    """
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"检查点已保存至: {filepath}")

def load_checkpoint(filepath='/root/autodl-fs/stacking_checkpoint.pkl'):
    """
    加载检查点数据
    
    参数:
    filepath: 加载路径
    
    返回:
    checkpoint_data: 检查点数据字典，如果文件不存在则返回None
    """
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)
        print(f"检查点已加载: {filepath}")
        return checkpoint_data
    else:
        print(f"检查点文件不存在: {filepath}")
        return None

# 检查GPU可用性
def check_gpu_availability():
    """检查GPU是否可用"""
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("GPU不可用，将使用CPU")
        return False

# 定义一个通用的Stacking类，可以处理回归和分类问题
class ModelStacking:
    def __init__(self, task_type='classification', cv=5, random_state=42, use_gpu=False):
        """
        初始化模型堆叠类
        
        参数:
        task_type: 'regression' 或 'classification'，指定任务类型
        cv: 交叉验证折数
        random_state: 随机种子
        use_gpu: 是否使用GPU加速
        """
        self.task_type = task_type
        self.cv = cv
        self.random_state = random_state
        self.base_models = []
        self.meta_model = None
        self.stacking_model = None
        self.best_score = float('-inf') if task_type == 'classification' else float('inf')
        self.use_gpu = use_gpu and check_gpu_availability()
        
    def add_base_models(self, models_dict):
        """
        添加基础模型
        
        参数:
        models_dict: 字典，键为模型名称，值为模型对象
        """
        self.base_models = [(name, model) for name, model in models_dict.items()]
        
    def add_pretrained_models(self, model_paths):
        """
        添加预训练的基础模型
        
        参数:
        model_paths: 字典，键为模型名称，值为模型文件路径
        """
        models_dict = {}
        for name, path in model_paths.items():
            with open(path, 'rb') as f:
                model = pickle.load(f)
                # 对支持GPU的模型进行设置
                if self.use_gpu:
                    if isinstance(model, XGBClassifier) or isinstance(model, XGBRegressor):
                        # 设置XGBoost使用GPU
                        model.set_params(tree_method='gpu_hist', gpu_id=0)
                    elif isinstance(model, LGBMClassifier) or isinstance(model, LGBMRegressor):
                        # 设置LightGBM使用GPU并调整参数解决警告
                        model.set_params(
                            device='gpu' if self.use_gpu else 'cpu',
                            gpu_platform_id=0,
                            gpu_device_id=0,
                            min_data_in_leaf=5,  # 增加叶子节点的最小样本数
                            min_gain_to_split=0.0,  # 降低分裂增益阈值
                            verbose=-1  # 关闭冗余输出
                        )
                # 如果是LightGBM模型，无论是否使用GPU都调整参数
                elif isinstance(model, LGBMClassifier) or isinstance(model, LGBMRegressor):
                    model.set_params(
                        min_data_in_leaf=5,
                        min_gain_to_split=0.0,
                        verbose=-1
                    )
                models_dict[name] = model
        
        self.base_models = [(name, model) for name, model in models_dict.items()]
        
    def set_meta_model(self, meta_model):
        """
        设置元模型
        
        参数:
        meta_model: 元模型对象
        """
        self.meta_model = meta_model
        
    def build_stacking_model(self, passthrough=False):
        """
        构建堆叠模型
        
        参数:
        passthrough: 是否将原始特征传递给元模型
        """
        if self.task_type == 'regression':
            self.stacking_model = StackingRegressor(
                estimators=self.base_models,
                final_estimator=self.meta_model,
                cv=self.cv,
                n_jobs=-1,
                passthrough=passthrough
            )
        else:  # classification
            self.stacking_model = StackingClassifier(
                estimators=self.base_models,
                final_estimator=self.meta_model,
                cv=self.cv,
                n_jobs=-1,
                passthrough=passthrough
            )
            
    def fit(self, X, y):
        """
        训练堆叠模型
        
        参数:
        X: 特征矩阵
        y: 目标变量
        """
        self.stacking_model.fit(X, y)
        
    def predict(self, X):
        """
        使用堆叠模型进行预测
        
        参数:
        X: 特征矩阵
        
        返回:
        预测结果
        """
        return self.stacking_model.predict(X)
    
    def predict_proba(self, X):
        """
        使用堆叠模型进行概率预测（仅适用于分类任务）
        
        参数:
        X: 特征矩阵
        
        返回:
        预测概率
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba只适用于分类任务")
        return self.stacking_model.predict_proba(X)
        
    def evaluate(self, X, y, metric=None):
        """
        评估堆叠模型
        
        参数:
        X: 特征矩阵
        y: 目标变量
        metric: 评估指标，默认为None（回归使用MAE，分类使用准确率）
        
        返回:
        score: 评估分数
        """
        if metric is None:
            if self.task_type == 'regression':
                metric = mean_absolute_error
                score = -metric(y, self.stacking_model.predict(X))  # 负MAE，越高越好
            else:  # classification
                score = self.stacking_model.score(X, y)  # 准确率
        else:
            if self.task_type == 'regression':
                score = -metric(y, self.stacking_model.predict(X))  # 负误差，越高越好
            else:  # classification
                y_pred = self.stacking_model.predict(X)
                score = metric(y, y_pred)
                
        return score
    
    def cross_validate(self, X, y, metric=None):
        """
        交叉验证堆叠模型
        
        参数:
        X: 特征矩阵
        y: 目标变量
        metric: 评估指标，默认为None
        
        返回:
        mean_score: 平均评估分数
        """
        if metric is None:
            if self.task_type == 'regression':
                scoring = 'neg_mean_absolute_error'
            else:  # classification
                scoring = 'accuracy'
        else:
            scoring = metric
            
        scores = cross_val_score(
            self.stacking_model, X, y, 
            cv=KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state),
            scoring=scoring, n_jobs=-1
        )
        
        return scores.mean()
    
    def save_model(self, filepath):
        """
        保存堆叠模型
        
        参数:
        filepath: 保存路径
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.stacking_model, f)
            
    def load_model(self, filepath):
        """
        加载堆叠模型
        
        参数:
        filepath: 加载路径
        """
        with open(filepath, 'rb') as f:
            self.stacking_model = pickle.load(f)
            
    def get_feature_importance(self):
        """
        获取特征重要性（如果元模型支持）
        
        返回:
        feature_importance: 特征重要性
        """
        if hasattr(self.stacking_model.final_estimator_, 'feature_importances_'):
            return self.stacking_model.final_estimator_.feature_importances_
        elif hasattr(self.stacking_model.final_estimator_, 'coef_'):
            return np.abs(self.stacking_model.final_estimator_.coef_)
        else:
            raise ValueError("元模型不支持特征重要性")

# 使用预训练模型进行堆叠的函数
def build_stacking_with_pretrained_models(X_train, y_train, X_test, y_test, model_paths, meta_model=None, passthrough=False, use_gpu=False):
    """
    使用预训练模型构建堆叠模型
    
    参数:
    X_train, y_train: 训练数据
    X_test, y_test: 测试数据
    model_paths: 字典，键为模型名称，值为模型文件路径
    meta_model: 元模型，默认为None（使用LogisticRegression）
    passthrough: 是否将原始特征传递给元模型
    use_gpu: 是否使用GPU加速
    
    返回:
    stacking: 堆叠模型
    """
    # 默认使用LogisticRegression作为元模型
    if meta_model is None:
        meta_model = LogisticRegression(random_state=42)
    
    # 创建堆叠模型
    stacking = ModelStacking(task_type='classification', cv=5, use_gpu=use_gpu)
    stacking.add_pretrained_models(model_paths)
    stacking.set_meta_model(meta_model)
    stacking.build_stacking_model(passthrough=passthrough)
    
    # 训练堆叠模型
    stacking.fit(X_train, y_train)
    
    # 评估堆叠模型
    train_score = stacking.evaluate(X_train, y_train)
    test_score = stacking.evaluate(X_test, y_test)
    
    print(f"训练集准确率: {train_score:.4f}")
    print(f"测试集准确率: {test_score:.4f}")
    
    # 打印分类报告
    y_pred = stacking.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 保存模型
    stacking.save_model('/root/autodl-fs/stacking_classification_model.pkl')
    
    return stacking

# 尝试不同元模型的函数
def try_different_meta_models(X_train, y_train, X_val, y_val, model_paths, passthrough=False, use_gpu=False):
    """
    尝试不同的元模型，找到最佳组合
    
    参数:
    X_train, y_train: 训练数据
    X_val, y_val: 验证数据
    model_paths: 字典，键为模型名称，值为模型文件路径
    passthrough: 是否将原始特征传递给元模型
    use_gpu: 是否使用GPU加速
    
    返回:
    best_stacking: 最佳堆叠模型
    """
    # 创建不同的元模型，对LightGBM进行特殊设置以避免警告
    meta_models = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, tree_method='gpu_hist', gpu_id=0) if use_gpu else XGBClassifier(random_state=42),
        'LightGBM': LGBMClassifier(
            random_state=42, 
            device='gpu' if use_gpu else 'cpu',
            gpu_platform_id=0 if use_gpu else -1,
            gpu_device_id=0 if use_gpu else -1,
            min_data_in_leaf=5,
            min_gain_to_split=0.0,
            verbose=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    
    best_score = 0
    best_meta_model = None
    best_stacking = None
    
    for name, meta_model in meta_models.items():
        print(f"\n尝试元模型: {name}")
        
        stacking = ModelStacking(task_type='classification', cv=5, use_gpu=use_gpu)
        stacking.add_pretrained_models(model_paths)
        stacking.set_meta_model(meta_model)
        stacking.build_stacking_model(passthrough=passthrough)
        
        try:
            stacking.fit(X_train, y_train)
            score = stacking.evaluate(X_val, y_val)
            print(f"验证集准确率: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_meta_model = name
                best_stacking = stacking
                print(f"找到更好的元模型! 准确率: {score:.4f}")
        except Exception as e:
            print(f"错误: {e}")
            continue
    
    print(f"\n最佳元模型: {best_meta_model}")
    print(f"最佳准确率: {best_score:.4f}")
    
    return best_stacking

# 定义一个寻找最佳模型组合的函数，支持断点续跑
def find_best_model_combination(X_train, y_train, X_val, y_val, model_paths, meta_models=None, min_models=2, passthrough=False, use_gpu=False, resume=True):
    """
    尝试不同的基础模型组合和元模型，找到最佳堆叠模型
    
    参数:
    X_train, y_train: 训练数据
    X_val, y_val: 验证数据
    model_paths: 字典，键为模型名称，值为模型文件路径
    meta_models: 字典，键为元模型名称，值为元模型对象，默认为None
    min_models: 最小的基础模型数量，默认为2
    passthrough: 是否将原始特征传递给元模型，默认为False
    use_gpu: 是否使用GPU加速
    resume: 是否从检查点恢复，默认为True
    
    返回:
    best_stacking: 最佳堆叠模型
    best_models: 最佳基础模型组合
    best_meta_model: 最佳元模型
    best_score: 最佳分数
    """
    if meta_models is None:
        meta_models = {
            'LogisticRegression': LogisticRegression(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, tree_method='gpu_hist', gpu_id=0) if use_gpu else XGBClassifier(random_state=42),
            'LightGBM': LGBMClassifier(
                random_state=42, 
                device='gpu' if use_gpu else 'cpu',
                gpu_platform_id=0 if use_gpu else -1,
                gpu_device_id=0 if use_gpu else -1,
                min_data_in_leaf=5,
                min_gain_to_split=0.0,
                verbose=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
    
    # 加载所有预训练模型
    all_models = {}
    for name, path in model_paths.items():
        with open(path, 'rb') as f:
            model = pickle.load(f)
            # 对支持GPU的模型进行设置
            if use_gpu:
                if isinstance(model, XGBClassifier) or isinstance(model, XGBRegressor):
                    # 设置XGBoost使用GPU
                    model.set_params(tree_method='gpu_hist', gpu_id=0)
                elif isinstance(model, LGBMClassifier) or isinstance(model, LGBMRegressor):
                    # 设置LightGBM使用GPU并调整参数解决警告
                    model.set_params(
                        device='gpu',
                        gpu_platform_id=0,
                        gpu_device_id=0,
                        min_data_in_leaf=5,
                        min_gain_to_split=0.0,
                        verbose=-1
                    )
            # 如果是LightGBM模型，无论是否使用GPU都调整参数
            elif isinstance(model, LGBMClassifier) or isinstance(model, LGBMRegressor):
                model.set_params(
                    min_data_in_leaf=5,
                    min_gain_to_split=0.0,
                    verbose=-1
                )
            all_models[name] = model
    
    # 检查是否有检查点可以恢复
    checkpoint_path = '/root/autodl-fs/stacking_checkpoint.pkl'
    checkpoint_data = None
    if resume:
        checkpoint_data = load_checkpoint(checkpoint_path)
    
    # 如果有检查点，恢复之前的状态
    if checkpoint_data is not None:
        best_score = checkpoint_data['best_score']
        best_models = checkpoint_data['best_models']
        best_meta_model = checkpoint_data['best_meta_model']
        best_stacking = checkpoint_data['best_stacking']
        best_combination = checkpoint_data['best_combination']
        results = checkpoint_data['results']
        
        # 恢复进度
        start_n_models = checkpoint_data['current_n_models']
        start_model_names_idx = checkpoint_data['current_model_names_idx']
        start_meta_name = checkpoint_data['current_meta_name']
        
        print(f"从检查点恢复: 当前最佳分数 {best_score:.4f}, 已完成组合数 {len(results)}")
    else:
        best_score = 0
        best_models = None
        best_meta_model = None
        best_stacking = None
        best_combination = None
        results = []
        
        # 从头开始
        start_n_models = min_models
        start_model_names_idx = 0
        start_meta_name = None
    
    # 尝试不同数量的模型组合
    max_models = len(all_models)
    
    # 计算总组合数
    total_combinations = sum(len(list(combinations(all_models.keys(), i))) for i in range(min_models, max_models + 1))
    
    print(f"将尝试从{min_models}到{max_models}个模型的所有组合，共{total_combinations}种组合")
    
    # 创建进度条
    with tqdm(total=total_combinations * len(meta_models)) as pbar:
        # 如果从检查点恢复，更新进度条
        if checkpoint_data is not None:
            pbar.update(len(results))
        
        # 遍历模型数量
        for n_models in range(start_n_models, max_models + 1):
            # 获取当前模型数量的所有组合
            model_combinations = list(combinations(all_models.keys(), n_models))
            
            # 如果是恢复的第一个n_models，从上次的索引开始
            start_idx = start_model_names_idx if n_models == start_n_models else 0
            
            # 遍历模型组合
            for idx, model_names in enumerate(model_combinations[start_idx:], start=start_idx):
                # 创建当前组合的模型字典
                current_models = {name: all_models[name] for name in model_names}
                
                # 确定从哪个元模型开始
                meta_items = list(meta_models.items())
                start_meta_idx = 0
                if n_models == start_n_models and idx == start_model_names_idx and start_meta_name is not None:
                    # 找到上次处理的元模型的索引
                    for i, (name, _) in enumerate(meta_items):
                        if name == start_meta_name:
                            start_meta_idx = i + 1  # 从下一个开始
                            break
                
                # 尝试不同的元模型
                for meta_idx, (meta_name, meta_model) in enumerate(meta_items[start_meta_idx:], start=start_meta_idx):
                    pbar.update(1)
                    
                    # 创建堆叠模型
                    estimators = [(name, model) for name, model in current_models.items()]
                    stacking = StackingClassifier(
                        estimators=estimators,
                        final_estimator=clone(meta_model),
                        cv=5,
                        n_jobs=-1,
                        passthrough=passthrough
                    )
                    
                    try:
                        # 训练堆叠模型
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            stacking.fit(X_train, y_train)
                        
                        # 评估堆叠模型
                        score = stacking.score(X_val, y_val)
                        
                        # 记录结果
                        result = {
                            'model_combination': model_names,
                            'meta_model': meta_name,
                            'n_models': n_models,
                            'accuracy': score
                        }
                        results.append(result)
                        
                        # 更新最佳模型
                        if score > best_score:
                            best_score = score
                            best_models = current_models
                            best_meta_model = meta_model
                            best_stacking = stacking
                            best_combination = model_names
                            
                            print(f"\n新的最佳组合! 模型: {model_names}")
                            print(f"元模型: {meta_name}")
                            print(f"验证集准确率: {score:.4f}")
                    except Exception as e:
                        print(f"\n错误: 模型组合 {model_names}, 元模型 {meta_name}")
                        print(f"错误信息: {e}")
                        continue
                    
                    # 每处理10个组合保存一次检查点
                    if len(results) % 10 == 0:
                        checkpoint_data = {
                            'best_score': best_score,
                            'best_models': best_models,
                            'best_meta_model': best_meta_model,
                            'best_stacking': best_stacking,
                            'best_combination': best_combination,
                            'results': results,
                            'current_n_models': n_models,
                            'current_model_names_idx': idx,
                            'current_meta_name': meta_name,
                            'timestamp': time.time()
                        }
                        save_checkpoint(checkpoint_data, checkpoint_path)
    
    # 将结果转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    # 保存结果
    results_df.to_csv('/root/autodl-fs/stacking_results.csv', index=False)
    
    # 打印最佳组合
    print("\n===== 最佳模型组合 =====")
    print(f"基础模型: {best_combination}")
    print(f"元模型: {type(best_meta_model).__name__}")
    print(f"验证集准确率: {best_score:.4f}")
    
    # 打印前10个最佳组合
    print("\n===== 前10个最佳组合 =====")
    print(results_df.head(10))
    
    # 可视化前10个最佳组合
    top10 = results_df.head(10)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        [f"{', '.join(comb)} + {meta}" for comb, meta in zip(top10['model_combination'], top10['meta_model'])],
        top10['accuracy']
    )
    plt.xlabel('模型组合')
    plt.ylabel('准确率')
    plt.title('前10个最佳模型组合')
    plt.xticks(rotation=90)
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig('/root/autodl-fs/top10_combinations.png')
    plt.show()
    
    return best_stacking, best_models, best_meta_model, best_score

# 可视化模型重要性的函数
def visualize_model_importance(stacking_model):
    """
    可视化堆叠模型中各基础模型的重要性
    
    参数:
    stacking_model: 堆叠模型
    """
    try:
        importances = stacking_model.get_feature_importance()
        model_names = [name for name, _ in stacking_model.base_models]
        
        if len(importances.shape) > 1:  # 多分类情况
            importances = np.mean(importances, axis=0)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, importances)
        plt.xlabel('基础模型')
        plt.ylabel('重要性')
        plt.title('堆叠模型中各基础模型的重要性')
        plt.xticks(rotation=45)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{height:.4f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig('/root/autodl-fs/model_importance.png')
        plt.show()
    except Exception as e:
        print(f"无法获取特征重要性: {e}")

# 主函数：寻找最佳模型组合并保存
def main(use_gpu=False, resume=True):
    """
    主函数：寻找最佳模型组合并保存
    
    参数:
    use_gpu: 是否使用GPU加速
    resume: 是否从检查点恢复
    """
    # 忽略所有警告
    warnings.filterwarnings("ignore")
    
    # 检查GPU可用性
    if use_gpu:
        use_gpu = check_gpu_availability()
    
    # 定义模型路径
    model_paths = {
        'RandomForest': "/root/autodl-fs/best_model_RF.pkl",
        'GradientBoosting': "/root/autodl-fs/best_model_GradientBoosting.pkl",
        'SVM': "/root/autodl-fs/best_model_SVM.pkl",
        'XGBoost': "/root/autodl-fs/best_model_XGB.pkl",
        'LightGBM': "/root/autodl-fs/best_model_LGBM.pkl",
        'AdaBoost': "/root/autodl-fs/best_model_AdaBoost.pkl",
        'LogisticRegression': "/root/autodl-fs/best_model_LogisticRegression.pkl",
        'DecisionTree': "/root/autodl-fs/best_model_DT.pkl",
        'KNN': "/root/autodl-fs/best_model_kNN.pkl",
        'NaiveBayes': "/root/autodl-fs/best_model_NaiveBayes.pkl"
    }
    
    # 加载数据
    # 假设你已经有了 X 和 y
    # X = df.drop(columns=['Three_Class']).values
    # y = df['Three_Class'].values
    
    # 读取数据
    file_path = r"C:\Users\Yaya\Desktop\final_design\Dataset_filtered_cleaned.csv"
    df = pd.read_csv(file_path)

    # 分离特征和目标变量
    X = df.drop(columns=['Three_Class']).values
    y = df['Three_Class'].values

    # 编码目标变量
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 标准化特征
    stdScale = StandardScaler().fit(X)
    X = stdScale.transform(X)
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # 寻找最佳模型组合
    best_stacking, best_models, best_meta_model, best_val_score = find_best_model_combination(
        X_train, y_train, X_val, y_val, model_paths, use_gpu=use_gpu, resume=resume
    )
    
    # 在完整训练集上重新训练最佳堆叠模型
    print("\n在完整训练集上重新训练最佳堆叠模型...")
    X_train_full = np.vstack((X_train, X_val))
    y_train_full = np.hstack((y_train, y_val))
    
    # 创建最佳模型组合的堆叠模型
    estimators = [(name, model) for name, model in best_models.items()]
    final_stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=clone(best_meta_model),
        cv=5,
        n_jobs=-1
    )
    
    # 训练最终模型
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_stacking.fit(X_train_full, y_train_full)
    
    # 在测试集上评估
    test_score = final_stacking.score(X_test, y_test)
    print(f"测试集准确率: {test_score:.4f}")
    
    # 打印分类报告
    y_pred = final_stacking.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 保存最佳模型
    with open('/root/autodl-fs/best_stacking_model.pkl', 'wb') as f:
        pickle.dump(final_stacking, f)
    
    # 保存最佳模型的详细信息
    model_info = {
        'base_models': list(best_models.keys()),
        'meta_model': type(best_meta_model).__name__,
        'validation_accuracy': best_val_score,
        'test_accuracy': test_score
    }
    
    with open('/root/autodl-fs/best_stacking_model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"\n最佳堆叠模型已保存至: /root/autodl-fs/best_stacking_model.pkl")
    print(f"模型信息已保存至: /root/autodl-fs/best_stacking_model_info.pkl")
    
    return final_stacking, model_info

# 执行主函数
if __name__ == "__main__":
    # 设置use_gpu=True来启用GPU加速，resume=True来从检查点恢复
    best_stacking, model_info = main(use_gpu=True, resume=True)