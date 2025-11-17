# from agents import function_tool  # LangGraph版本不需要
import os
import pandas as pd
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from .UNet import UNet1D
from .ViT import ViT as PViT
from .Prior_predict_UNet import UNet1D as Prior_UNet

# 获取项目根目录
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
_default_model_dir = os.path.join(_project_root, 'all_file_path')

def blade_design(flow_rate: float,
                   efficiency: float,
                   pressure_ratio: float,
                   generative_model_path: str = None,
                   prior_model_path: str = None,
                   predictor_model_path: str = None,
                   design_paras_path: str = None,
                   performance_paras_path: str = None,
                   save_path: str = True,
                   batch_size: int = 150,
                   timesteps: int = 500,
                   top_k: int = 1,
                   device: str = None) -> dict:
    """
    基于目标性能参数生成最优涡轮叶片几何设计（返回Top-k个设计方案），每个设计包含21个参数，分为叶根、叶中、叶尖三部分。

    **参数说明**:
    - flow_rate (float): 目标流量（单位: kg/s）
    - efficiency (float): 目标等熵效率（范围: 0-1，典型值: 0.85）
    - pressure_ratio (float): 目标压比（必须>1，典型值: 1.5）
    - generative_model_path (str): 扩散模型路径（默认: 'pre_guide_15200.pth'）
    - prior_model_path (str): 先验模型路径（默认: 'prior_parameters_guide_15200.pth'）
    - predictor_model_path (str): 性能预测模型路径（默认: 'best_model_in_testdataset.pth'）
    - design_paras_path (str): 设计参数基准文件（.npy格式）
    - performance_paras_path (str): 性能参数基准文件（.npy格式）
    - save_path (bool): 是否自动保存结果（默认: True）
    - batch_size (int): 每轮生成候选数（默认: 150）
    - timesteps (int): 扩散时间步数（默认: 500）
    - top_k (int): 设计方案数量（默认: 1）
    - device (str): 计算设备（'cuda'或'cpu'）

    **返回值** (dict):
    返回包含 `top_k` 个设计的字典，结构为：
    ```python
    {
        "第1个设计结果": {
            # 叶根区参数（0-6）
            "root_Angle_in": 值,  # 进口金属角（°）
            "root_Angle_out": 值,  # 出口金属角（°）
            "root_Chord": 值,  # 弦长（米）
            "root_THmax/CH": 值,  # 最大相对厚度（%）
            "root_THmaxP": 值,  # 最大厚度位置（0-1）
            "root_SWA": 值,  # 弯量（米）
            "root_BOWA": 值,  # 掠（米）
            # 叶中区参数（7-13）...
            # 叶尖区参数（14-20）...
        },
        # 其他设计...
    }
    ```
    **单位说明**:
    - 角度: 度（°）
    - 长度: 米（m）
    - 厚度: 百分比（%）
    - 位置: 无量纲（0-1）
    """
    
    # 自动选择设备（优先使用 CUDA，不可用时使用 CPU）
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 设置默认路径（如果未提供）
    if generative_model_path is None:
        generative_model_path = os.path.join(_default_model_dir, 'pre_guide_15200.pth')
    if prior_model_path is None:
        prior_model_path = os.path.join(_default_model_dir, 'prior_parameters_guide_15200.pth')
    if predictor_model_path is None:
        predictor_model_path = os.path.join(_default_model_dir, 'best_model_in_testdataset.pth')
    if design_paras_path is None:
        design_paras_path = os.path.join(_default_model_dir, 'parameter_geometry_5200.npy')
    if performance_paras_path is None:
        performance_paras_path = os.path.join(_default_model_dir, 'label_5200.npy')

    # 构造性能向量并归一化
    target_performance = np.array([[flow_rate, efficiency, pressure_ratio]])
    performance_paras = np.load(performance_paras_path)
    perf_max = performance_paras.max(axis=0)[[0,1,2]]
    perf_min = performance_paras.min(axis=0)[[0,1,2]]
    target_norm = (target_performance - perf_min) / (perf_max - perf_min)
    target_norm = np.clip(target_norm, 0.0, 1.0)

    # 噪声调度器
    def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, timesteps)

    betas = linear_beta_schedule(timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 加载模型（兼容 macOS，使用 weights_only=False）
    generative_model = UNet1D(input_dim=42, time_embed_dim=256, condition_dim=3).to(device)
    generative_model.load_state_dict(torch.load(generative_model_path, map_location=device, weights_only=False))
    generative_model.eval()

    prior_model = Prior_UNet(input_dim=3).to(device)
    prior_model.load_state_dict(torch.load(prior_model_path, map_location=device, weights_only=False))
    prior_model.eval()

    performance_model = PViT(
        lenth=1, channels=21, num_classes=3, dim=256,
        depth=14, heads=12, mlp_dim=256, dropout=0.085, emb_dropout=0.072, dim_head=256).to(device)
    performance_model.load_state_dict(torch.load(predictor_model_path, map_location=device, weights_only=False))
    performance_model = nn.DataParallel(performance_model)
    performance_model.eval()

    # 条件采样
    def sample(model, model1, condition, shape, timesteps, alphas, alphas_cumprod, betas, device):
        x_t = torch.randn(shape).to(device)
        performance = torch.tensor(condition, dtype=torch.float32).to(device)
        prior = model1(performance).repeat(shape[0], 1).unsqueeze(2)
        for t in range(timesteps - 1, -1, -1):
            x = torch.cat((x_t, prior), dim=1)
            t_tensor = torch.full((shape[0],), t, dtype=torch.long).to(device)
            y_tensor = performance.repeat(shape[0], 1)
            with torch.no_grad():
                epsilon_pred = model(x, t_tensor, y_tensor)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])
            x_t = (x_t - (betas[t] / sqrt_one_minus_alpha_cumprod_t) * epsilon_pred) / torch.sqrt(alphas[t])
            if t > 0:
                x_t += torch.sqrt(betas[t]) * torch.randn_like(x_t)
        return x_t

    # 验证并返回 Top-k
    def validate_topk(data, model, condition, k):
        cond = torch.tensor(condition, dtype=torch.float32).to(device)
        pred = model(data).detach()
        mse = torch.mean((pred - cond) ** 2, dim=1)
        topk_indices = torch.topk(mse, k=min(k, len(mse)), largest=False).indices
        pred = pred[topk_indices].cpu().numpy()*(perf_max - perf_min)+perf_min
        print(pred)
        return pred,data[topk_indices]

    # 生成并筛选
    gen_data = sample(generative_model, prior_model, target_norm, (batch_size, 21, 1), timesteps, alphas, alphas_cumprod, betas, device)
    pred,topk_designs = validate_topk(gen_data, performance_model, target_norm, top_k)
    topk_designs=topk_designs.squeeze().cpu().numpy()

    # 确保topk_designs是2D数组
    if topk_designs.ndim == 1:
        topk_designs = topk_designs.reshape(1, -1)

    design_paras = np.load(design_paras_path)
    design_max = design_paras.max(axis=0)
    design_min = design_paras.min(axis=0)
    topk_designs = topk_designs * (design_max - design_min) + design_min

    # 可选保存（保存前k个设计参数）
    if save_path:
        header = 'Index\t' + '\t'.join([
            "Angle_in1", "Angle_out1", "Chord1", "THmax/CH1", "THmaxP1", "SWA1", "BOWA1",
            "Angle_in2", "Angle_out2", "Chord2", "THmax/CH2", "THmaxP2", "SWA2", "BOWA2",
            "Angle_in3", "Angle_out3", "Chord3", "THmax/CH3", "THmaxP3", "SWA3", "BOWA3"
        ])
        indices = np.arange(1, len(topk_designs) + 1).reshape(-1, 1)
        samples_with_indices = np.hstack((indices, topk_designs.reshape(-1, 21)))
        with open(f"..\\geometry_generate\\{flow_rate}_{efficiency}_{pressure_ratio}_{top_k}.csv", 'w') as f:
            f.write(header + '\n')
            np.savetxt(f, samples_with_indices, delimiter='\t', fmt='%d' + '\t' + ('%.6f' + '\t') * 20 + '%.6f')

    var_names = [
        "root_Angle_in（叶根进口金属角[°]）", "root_Angle_out（叶根出口金属角[°]）", "root_Chord（叶根弦长[m]）", "root_THmax_CH（叶根最大相对厚度[%]）", "root_THmaxP（叶根最大相对厚度位置[-]）", "root_SWA（叶根最大相对厚度位置[m]）", "root_BOWA（叶根掠[m]）",
        "mid_Angle_in（叶中进口金属角[°]）", "mid_Angle_out（叶中出口金属角[°]）", "mid_Chord（叶中弦长[m]）", "mid_THmax_CH（叶中最大相对厚度[%]）", "mid_THmaxP（叶中最大相对厚度位置[-]）", "mid_SWA（叶中最大相对厚度位置[m]）", "mid_BOWA（叶中掠[m]）",
        "tip_Angle_in（叶顶进口金属角[°]）", "tip_Angle_out（叶顶出口金属角[°]）", "tip_Chord（叶顶弦长[m]）", "tip_THmax_CH（叶顶最大相对厚度[%]）", "tip_THmaxP（叶顶最大相对厚度位置[-]）", "tip_SWA（叶顶最大相对厚度位置[m]）", "tip_BOWA（叶顶掠[m]）",
        "mass flow rate（预测流量[kg/s]）", "isentropic efficiency（预测等熵效率[-]）", "total pressure ratio（预测总压比[-]）"
    ]
    # df = pd.DataFrame(topk_designs, columns=var_names)
    df=np.hstack((topk_designs,pred))
    # 创建结果字典
    result_dict = {}
    for i in range(df.shape[0]):
        design_dict = {}
        for j, var_name in enumerate(var_names):
            design_dict[var_name] = df[i, j]
        result_dict[f"第{i+1}个设计结果"] = design_dict
    result_dict[f"设计参数保存路径"]=f"{flow_rate}_{efficiency}_{pressure_ratio}_{top_k}.csv"
    return result_dict

if __name__=="__main__":
#     # 查看装饰后的工具元数据
#     tool = blade_design  # 装饰后的函数实际是 FunctionTool 实例
#     print(tool.name)  # 函数名（或 name_override）
#     print(tool.description)  # 从docstring提取的描述
#     print(tool.params_json_schema)  # 包含参数描述的JSON Schema
    passresults = blade_design(
    flow_rate=15.2,
    efficiency=0.83,
    pressure_ratio=1.55)
    print(passresults)