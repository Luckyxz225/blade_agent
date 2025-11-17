# from agents import function_tool  # LangGraph版本不需要
import os
import pandas as pd
import torch
import numpy as np
from .ViT import ViT as PViT
import numpy as np

# 获取项目根目录
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
_default_model_dir = os.path.join(_project_root, 'all_file_path')

def blade_performance_evaluation(design_paras: list[list[float]]) -> dict:
    """
    根据设计参数对其气动性能进行评估，返回预测的流量 等熵效率 总压比，支持多个设计同时评估。
    **参数说明**:
    - flow_rate (float): 一个二维列表，形状应为 (b, 21)，每个子列表代表一个样本。其中 b 表示样本数量，21 表示每个样本的特征维度,特征为[
        "root_Angle_in（叶根进口金属角[°]）", "root_Angle_out（叶根出口金属角[°]）", "root_Chord（叶根弦长[m]）", "root_THmax_CH（叶根最大相对厚度[%]）", "root_THmaxP（叶根最大相对厚度位置[-]）", "root_SWA（叶根最大相对厚度位置[m]）", "root_BOWA（叶根掠[m]）",
        "mid_Angle_in（叶中进口金属角[°]）", "mid_Angle_out（叶中出口金属角[°]）", "mid_Chord（叶中弦长[m]）", "mid_THmax_CH（叶中最大相对厚度[%]）", "mid_THmaxP（叶中最大相对厚度位置[-]）", "mid_SWA（叶中最大相对厚度位置[m]）", "mid_BOWA（叶中掠[m]）",
        "tip_Angle_in（叶顶进口金属角[°]）", "tip_Angle_out（叶顶出口金属角[°]）", "tip_Chord（叶顶弦长[m]）", "tip_THmax_CH（叶顶最大相对厚度[%]）", "tip_THmaxP（叶顶最大相对厚度位置[-]）", "tip_SWA（叶顶最大相对厚度位置[m]）", "tip_BOWA（叶顶掠[m]）"]。

    返回:
        dict: 包含计算结果的字典。
    """
    design_paras = np.array(design_paras)
    design_paras= torch.tensor(design_paras, dtype=torch.float)
    geometry_file = os.path.join(_default_model_dir, 'parameter_geometry_5200.npy')
    label_file = os.path.join(_default_model_dir, 'label_5200.npy')

    # Model setup
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
    # 使用 CPU 加载模型（兼容 macOS）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        os.path.join(_default_model_dir, 'best_model_in_testdataset.pth'),
        map_location=device,
        weights_only=False  # 设置为 False 以兼容旧版本模型文件
    )
    model.load_state_dict(checkpoint)
    model.to(device)

    # Denormalization function
    def denormalize(data, min_val, max_val):
        return data * (max_val - min_val) + min_val

    # Load label data for denormalization
    label = np.load(label_file)
    max_labels = torch.tensor(label.max(axis=0))
    min_labels = torch.tensor(label.min(axis=0))

    geometryparas = np.load(geometry_file)
    max_geometrys = torch.tensor(geometryparas.max(axis=0), dtype=torch.float)
    min_geometrys = torch.tensor(geometryparas.min(axis=0), dtype=torch.float)

    model.eval()
    with torch.no_grad():
        # 确保所有张量在同一设备上
        design_paras = design_paras.to(device)
        min_geometrys = min_geometrys.to(device)
        max_geometrys = max_geometrys.to(device)
        x = (design_paras-min_geometrys)/(max_geometrys-min_geometrys)
        x = x.reshape(-1,21, 1)
        pred = model(x)
        pred_denorm = denormalize(pred, min_labels, max_labels).numpy()
        var_names= [
        "mass flow rate（预测流量[kg/s]）", "isentropic efficiency（预测等熵效率[-]）", "total pressure ratio（预测总压比[-]）"
    ]
        result_dict = {}
        for i in range(pred_denorm.shape[0]):
            performance_dict = {}
            for j, var_name in enumerate(var_names):
                performance_dict[var_name] = pred_denorm[i, j]
            result_dict[f"第{i + 1}个设计性能结果"] = performance_dict
        return result_dict






