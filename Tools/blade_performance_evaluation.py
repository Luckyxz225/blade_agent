# from agents import function_tool  # LangGraph版本不需要
import os
import sys
import pandas as pd
import torch
import numpy as np
from .ViT import ViT as PViT
import numpy as np

# 获取项目根目录
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
_default_model_dir = os.path.join(_project_root, 'all_file_path')

# 导入系统配置（跨平台支持）
sys.path.insert(0, _project_root)

def get_device():
    """
    获取最佳计算设备（支持 CUDA/MPS/CPU）
    
    优先级：CUDA > MPS > CPU
    """
    # 优先检查 CUDA
    if torch.cuda.is_available():
        return "cuda"
    
    # macOS 上检查 MPS（Apple Silicon）
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                return "mps"
    except Exception:
        pass
    
    return "cpu"

def blade_performance_evaluation(design_paras: list[list[float]], verbose: bool = True) -> dict:
    """
    根据设计参数对其气动性能进行评估，返回预测的流量 等熵效率 总压比，支持多个设计同时评估。
    **参数说明**:
    - design_paras: 一个二维列表，形状应为 (b, 21)，每个子列表代表一个样本。
    - verbose: 是否打印详细过程信息，默认True。优化内部调用时建议设为False。

    返回:
        dict: 包含计算结果的字典。
    """
    design_paras = np.array(design_paras)
    num_designs = design_paras.shape[0]
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🚀 [性能评估任务] 开始")
        print(f"{'='*70}")
        print(f"📋 任务配置:")
        print(f"  • 评估设计数: {num_designs}")
        print(f"  • 参数维度: {design_paras.shape[1]} (21维叶片几何参数)")
    
    design_paras= torch.tensor(design_paras, dtype=torch.float)
    geometry_file = os.path.join(_default_model_dir, 'parameter_geometry_5200.npy')
    label_file = os.path.join(_default_model_dir, 'label_5200.npy')

    # 使用系统配置自动选择最佳设备（支持 CUDA/MPS/CPU）
    device = get_device()
    
    if verbose:
        print(f"\n📦 加载性能评估模型 (ViT)...")
        print(f"  • 计算设备: {device.upper()}")
        print(f"  • 创建ViT模型架构...")
    
    model = PViT(
        lenth=1,
        channels=21,
        num_classes=3,
        dim=256,
        depth=14,
        heads=12,
        mlp_dim=256,
        dropout=0.085,
        emb_dropout=0.072,
        dim_head=256
    )
    
    if verbose:
        print(f"  • 加载模型权重...")
    
    checkpoint = torch.load(
        os.path.join(_default_model_dir, 'best_model_in_testdataset.pth'),
        map_location=device,
        weights_only=False  # 设置为 False 以兼容旧版本模型文件
    )
    model.load_state_dict(checkpoint)
    model.to(device)
    
    if verbose:
        print(f"✅ ViT模型加载完成！")

    # Denormalization function
    def denormalize(data, min_val, max_val):
        return data * (max_val - min_val) + min_val

    # Load label data for denormalization
    label = np.load(label_file)
    max_labels = torch.tensor(label.max(axis=0), dtype=torch.float)
    min_labels = torch.tensor(label.min(axis=0), dtype=torch.float)

    geometryparas = np.load(geometry_file)
    max_geometrys = torch.tensor(geometryparas.max(axis=0), dtype=torch.float)
    min_geometrys = torch.tensor(geometryparas.min(axis=0), dtype=torch.float)

    model.eval()
    
    with torch.no_grad():
        # 确保所有张量在同一设备上（重要：MPS/CUDA 需要所有张量在同一设备）
        design_paras = design_paras.to(device)
        min_geometrys = min_geometrys.to(device)
        max_geometrys = max_geometrys.to(device)
        min_labels = min_labels.to(device)  # 添加：确保在同一设备
        max_labels = max_labels.to(device)  # 添加：确保在同一设备
        
        x = (design_paras-min_geometrys)/(max_geometrys-min_geometrys)
        x = x.reshape(-1,21, 1)
        
        pred = model(x)
        pred_denorm = denormalize(pred, min_labels, max_labels).cpu().numpy()  # MPS/CUDA 需要先 .cpu()
        
        var_names = [
            "mass flow rate（预测流量[kg/s]）", 
            "isentropic efficiency（预测等熵效率[-]）", 
            "total pressure ratio（预测总压比[-]）"
        ]
        result_dict = {}
        for i in range(pred_denorm.shape[0]):
            performance_dict = {}
            for j, var_name in enumerate(var_names):
                # 转换为 Python 原生 float 并保留 3 位小数（避免 JSON 序列化错误）
                performance_dict[var_name] = round(float(pred_denorm[i, j]), 3)
            result_dict[f"第{i + 1}个设计性能结果"] = performance_dict
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ [性能评估任务] 完成！")
            print(f"{'='*70}")
            print(f"📊 评估结果:")
            for i in range(min(num_designs, 3)):  # 只显示前3个
                p = pred_denorm[i]
                print(f"  设计 #{i+1}: 流量={p[0]:.3f} kg/s, 效率={p[1]:.3f}, 压比={p[2]:.3f}")
            if num_designs > 3:
                print(f"  ... 共{num_designs}个设计的评估结果")
            print(f"{'='*70}\n")
        
        return result_dict






