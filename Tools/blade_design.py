# from agents import function_tool  # LangGraph版本不需要
import os
import sys
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

# 导入系统配置（跨平台支持）
sys.path.insert(0, _project_root)

def get_device():
    """
    获取最佳计算设备（支持 CUDA/MPS/CPU）
    
    优先级：CUDA > MPS > CPU
    """
    # 优先检查 CUDA
    if torch.cuda.is_available():
        print(f"[设备检测] 检测到 CUDA 设备")
        return "cuda"
    
    # macOS 上检查 MPS（Apple Silicon）
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # 进一步检查 MPS 是否真正可用
            if torch.backends.mps.is_built():
                print(f"[设备检测] 检测到 MPS 设备 (Apple Silicon GPU)")
                return "mps"
    except Exception as e:
        print(f"[设备检测] MPS 检测异常: {e}")
    
    print(f"[设备检测] 使用 CPU")
    return "cpu"

# 全局进度回调函数（由外部设置）
_progress_callback = None

def set_progress_callback(callback):
    """设置进度回调函数"""
    global _progress_callback
    _progress_callback = callback

def _report_progress(stage: str, progress: float, message: str = ""):
    """报告进度"""
    if _progress_callback:
        try:
            _progress_callback({
                'stage': stage,
                'progress': progress,
                'message': message
            })
        except Exception as e:
            print(f"[进度回调错误] {e}")


def _generate_intermediate_visualization(design_params: np.ndarray, label: str) -> str:
    """
    生成扩散过程中间结果的叶片可视化图片
    
    Args:
        design_params: 21维设计参数数组
        label: 图片标签（如 "10%", "20%" 等）
    
    Returns:
        图片路径（相对于static目录）
    """
    try:
        from .blade_visualization import visualize_blade
        import uuid
        
        # 解析21维参数
        params = {
            'root_Angle_in': design_params[0],
            'root_Angle_out': design_params[1],
            'root_Chord': design_params[2],
            'root_THmax_CH': design_params[3],
            'root_THmaxP': design_params[4],
            'root_SWA': design_params[5],
            'root_BOWA': design_params[6],
            'mid_Angle_in': design_params[7],
            'mid_Angle_out': design_params[8],
            'mid_Chord': design_params[9],
            'mid_THmax_CH': design_params[10],
            'mid_THmaxP': design_params[11],
            'mid_SWA': design_params[12],
            'mid_BOWA': design_params[13],
            'tip_Angle_in': design_params[14],
            'tip_Angle_out': design_params[15],
            'tip_Chord': design_params[16],
            'tip_THmax_CH': design_params[17],
            'tip_THmaxP': design_params[18],
            'tip_SWA': design_params[19],
            'tip_BOWA': design_params[20],
        }
        
        # 调用可视化函数（send_sse=False 避免干扰扩散中间图片的展示）
        result = visualize_blade(**params, save_image=True, send_sse=False)
        
        if result.get('status') == 'success' and result.get('image_path'):
            return result['image_path']
        
        return None
    except Exception as e:
        print(f"[中间可视化错误] {e}")
        return None

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
    
    # 打印任务开始信息
    print(f"\n{'='*70}")
    print(f"🚀 [叶片设计任务] 开始")
    print(f"{'='*70}")
    
    _report_progress('start', 0, '叶片设计任务开始')
    
    # 使用系统配置自动选择最佳设备（支持 CUDA/MPS/CPU）
    if device is None:
        device = get_device()
        # MPS 设备在某些操作上可能不稳定，提供警告
        if device == "mps":
            print(f"  ⚠️ 使用 Apple Silicon GPU (MPS)，如遇问题可手动指定 device='cpu'")
    
    print(f"📋 任务配置:")
    print(f"  • 目标流量: {flow_rate} kg/s")
    print(f"  • 目标效率: {efficiency}")
    print(f"  • 目标压比: {pressure_ratio}")
    print(f"  • 返回方案数: Top-{top_k}")
    print(f"  • 批量大小: {batch_size}")
    print(f"  • 扩散步数: {timesteps}")
    print(f"  • 计算设备: {device.upper()}")
    
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

    # 加载设计参数归一化范围（用于中间结果可视化）
    design_paras = np.load(design_paras_path)
    design_max = design_paras.max(axis=0)
    design_min = design_paras.min(axis=0)

    # 噪声调度器
    def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, timesteps)

    betas = linear_beta_schedule(timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 加载模型（兼容 macOS，使用 weights_only=False）
    print(f"\n📦 加载深度学习模型...")
    _report_progress('loading_models', 5, '正在加载深度学习模型...')
    
    print(f"  [1/3] 加载扩散生成模型 (UNet)...")
    _report_progress('loading_models', 10, '加载扩散生成模型 (UNet)...')
    generative_model = UNet1D(input_dim=42, time_embed_dim=256, condition_dim=3).to(device)
    generative_model.load_state_dict(torch.load(generative_model_path, map_location=device, weights_only=False))
    generative_model.eval()
    print(f"      ✅ UNet模型加载完成")

    print(f"  [2/3] 加载先验预测模型 (Prior-UNet)...")
    _report_progress('loading_models', 15, '加载先验预测模型 (Prior-UNet)...')
    prior_model = Prior_UNet(input_dim=3).to(device)
    prior_model.load_state_dict(torch.load(prior_model_path, map_location=device, weights_only=False))
    prior_model.eval()
    print(f"      ✅ Prior-UNet模型加载完成")

    print(f"  [3/3] 加载性能评估模型 (ViT)...")
    _report_progress('loading_models', 20, '加载性能评估模型 (ViT)...')
    performance_model = PViT(
        lenth=1, channels=21, num_classes=3, dim=256,
        depth=14, heads=12, mlp_dim=256, dropout=0.085, emb_dropout=0.072, dim_head=256).to(device)
    performance_model.load_state_dict(torch.load(predictor_model_path, map_location=device, weights_only=False))
    performance_model = nn.DataParallel(performance_model)
    performance_model.eval()
    print(f"      ✅ ViT模型加载完成")
    
    print(f"✅ 所有模型加载完成！")
    _report_progress('loading_models', 25, '所有模型加载完成')

    # 条件采样（带进度显示和中间结果可视化）
    def sample(model, model1, condition, shape, timesteps, alphas, alphas_cumprod, betas, device):
        print(f"\n{'='*70}")
        print(f"🔄 [叶片设计] 开始扩散采样")
        print(f"{'='*70}")
        print(f"  • 批量大小: {shape[0]}")
        print(f"  • 时间步数: {timesteps}")
        print(f"  • 设备: {device}")
        print(f"  • 目标性能（原始）: 流量={flow_rate:.2f} kg/s, 效率={efficiency:.2f}, 压比={pressure_ratio:.2f}")
        print(f"  • 目标性能（归一化）: 流量={condition[0][0]:.2f}, 效率={condition[0][1]:.2f}, 压比={condition[0][2]:.2f}")
        
        x_t = torch.randn(shape).to(device)
        performance = torch.tensor(condition, dtype=torch.float32).to(device)
        
        print(f"\n🧠 使用先验模型预测初始分布...")
        _report_progress('sampling', 30, '使用先验模型预测初始分布...')
        prior = model1(performance).repeat(shape[0], 1).unsqueeze(2)
        print(f"✅ 先验预测完成")
        
        print(f"\n⏳ 开始扩散去噪过程 (共{timesteps}步)...")
        print(f"{'─'*70}")
        _report_progress('sampling', 35, f'开始扩散去噪 (共{timesteps}步)')
        
        # 设置进度报告间隔
        report_interval = max(1, timesteps // 20)  # 每5%报告一次
        
        # 设置中间结果可视化间隔（每10%生成一张图片）
        viz_interval = max(1, timesteps // 10)  # 10%, 20%, ..., 100%
        viz_checkpoints = set([int(timesteps * p / 100) for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
        
        for i, t in enumerate(range(timesteps - 1, -1, -1)):
            x = torch.cat((x_t, prior), dim=1)
            t_tensor = torch.full((shape[0],), t, dtype=torch.long).to(device)
            y_tensor = performance.repeat(shape[0], 1)
            with torch.no_grad():
                epsilon_pred = model(x, t_tensor, y_tensor)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])
            x_t = (x_t - (betas[t] / sqrt_one_minus_alpha_cumprod_t) * epsilon_pred) / torch.sqrt(alphas[t])
            if t > 0:
                x_t += torch.sqrt(betas[t]) * torch.randn_like(x_t)
            
            current_step = i + 1
            progress = current_step / timesteps * 100
            
            # 显示进度（终端和前端）
            if current_step % report_interval == 0 or i == 0 or i == timesteps - 1:
                bar_length = 40
                filled = int(bar_length * current_step / timesteps)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"  [{bar}] {progress:.1f}% (步骤 {current_step}/{timesteps}, t={t})")
                
                # 报告到前端（35% + 50% * 扩散进度 = 35%-85%）
                frontend_progress = 35 + 50 * current_step / timesteps
                _report_progress('sampling', frontend_progress, f'扩散采样: {progress:.0f}% ({current_step}/{timesteps}步)')
            
            # 生成中间结果可视化图片（每10%）
            if current_step in viz_checkpoints:
                try:
                    # 获取当前最优设计（取第一个样本）
                    current_design = x_t[0].squeeze().cpu().numpy()
                    # 反归一化
                    current_design_actual = current_design * (design_max - design_min) + design_min
                    
                    # 生成可视化图片
                    viz_label = f"{int(progress)}%"
                    image_path = _generate_intermediate_visualization(current_design_actual, viz_label)
                    
                    if image_path:
                        print(f"  📸 生成中间结果图片: {viz_label}")
                        # 发送图片路径到前端（直接调用callback，包含image_path）
                        if _progress_callback:
                            _progress_callback({
                                'stage': 'design_image',
                                'image_path': image_path,
                                'label': viz_label,
                                'progress': progress
                            })
                except Exception as e:
                    print(f"  ⚠️ 生成中间图片失败: {e}")
        
        print(f"{'─'*70}")
        print(f"✅ 扩散采样完成！")
        return x_t

    # 验证并返回 Top-k
    def validate_topk(data, model, condition, k):
        print(f"\n🎯 [性能预测] 评估生成的设计...")
        print(f"  • 候选设计数: {data.shape[0]}")
        print(f"  • 选择Top-{k}最优设计")
        _report_progress('evaluation', 87, '评估生成的设计...')
        
        cond = torch.tensor(condition, dtype=torch.float32).to(device)
        pred = model(data).detach()
        mse = torch.mean((pred - cond) ** 2, dim=1)
        topk_indices = torch.topk(mse, k=min(k, len(mse)), largest=False).indices
        pred = pred[topk_indices].cpu().numpy()*(perf_max - perf_min)+perf_min
        
        print(f"✅ 性能预测完成")
        _report_progress('evaluation', 92, '性能预测完成')
        print(f"\n📊 Top-{k}设计的预测性能:")
        for i, p in enumerate(pred):
            print(f"  设计 #{i+1}: 流量={p[0]:.2f} kg/s, 效率={p[1]:.4f}, 压比={p[2]:.4f}")
        
        return pred,data[topk_indices]

    # 生成并筛选
    gen_data = sample(generative_model, prior_model, target_norm, (batch_size, 21, 1), timesteps, alphas, alphas_cumprod, betas, device)
    pred,topk_designs = validate_topk(gen_data, performance_model, target_norm, top_k)
    topk_designs=topk_designs.squeeze().cpu().numpy()

    # 确保topk_designs是2D数组
    if topk_designs.ndim == 1:
        topk_designs = topk_designs.reshape(1, -1)

    # design_max 和 design_min 已在前面加载，直接使用
    topk_designs = topk_designs * (design_max - design_min) + design_min

    # 可选保存（保存前k个设计参数）- 使用时间戳命名以便区分不同任务
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"design_{timestamp}_{flow_rate}_{efficiency}_{pressure_ratio}_topk{top_k}.csv"
    
    # 获取CSV存储路径
    csv_dir = os.path.join(_project_root, 'geometry_generate', 'csv_files')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, csv_filename)
    
    if save_path:
        header = 'Index\t' + '\t'.join([
            "Angle_in1", "Angle_out1", "Chord1", "THmax/CH1", "THmaxP1", "SWA1", "BOWA1",
            "Angle_in2", "Angle_out2", "Chord2", "THmax/CH2", "THmaxP2", "SWA2", "BOWA2",
            "Angle_in3", "Angle_out3", "Chord3", "THmax/CH3", "THmaxP3", "SWA3", "BOWA3"
        ])
        indices = np.arange(1, len(topk_designs) + 1).reshape(-1, 1)
        samples_with_indices = np.hstack((indices, topk_designs.reshape(-1, 21)))
        with open(csv_path, 'w') as f:
            f.write(header + '\n')
            np.savetxt(f, samples_with_indices, delimiter='\t', fmt='%d' + '\t' + ('%.6f' + '\t') * 20 + '%.6f')
        print(f"  📁 CSV文件已保存: {csv_filename}")

    var_names = [
        "root_Angle_in（叶根进口金属角[°]）", "root_Angle_out（叶根出口金属角[°]）", "root_Chord（叶根弦长[m]）", "root_THmax_CH（叶根最大相对厚度[%]）", "root_THmaxP（叶根最大相对厚度位置[-]）", "root_SWA（叶根最大相对厚度位置[m]）", "root_BOWA（叶根掠[m]）",
        "mid_Angle_in（叶中进口金属角[°]）", "mid_Angle_out（叶中出口金属角[°]）", "mid_Chord（叶中弦长[m]）", "mid_THmax_CH（叶中最大相对厚度[%]）", "mid_THmaxP（叶中最大相对厚度位置[-]）", "mid_SWA（叶中最大相对厚度位置[m]）", "mid_BOWA（叶中掠[m]）",
        "tip_Angle_in（叶顶进口金属角[°]）", "tip_Angle_out（叶顶出口金属角[°]）", "tip_Chord（叶顶弦长[m]）", "tip_THmax_CH（叶顶最大相对厚度[%]）", "tip_THmaxP（叶顶最大相对厚度位置[-]）", "tip_SWA（叶顶最大相对厚度位置[m]）", "tip_BOWA（叶顶掠[m]）",
        "mass flow rate（预测流量[kg/s]）", "isentropic efficiency（预测等熵效率[-]）", "total pressure ratio（预测总压比[-]）"
    ]
    # df = pd.DataFrame(topk_designs, columns=var_names)
    df=np.hstack((topk_designs,pred))
    
    print(f"\n📄 整理输出结果...")
    _report_progress('finalizing', 95, '整理输出结果...')
    
    # 创建结果字典
    result_dict = {}
    for i in range(df.shape[0]):
        design_dict = {}
        for j, var_name in enumerate(var_names):
            # 转换为 Python 原生 float 并保留 3 位小数（避免 JSON 序列化错误）
            design_dict[var_name] = round(float(df[i, j]), 3)
        result_dict[f"第{i+1}个设计结果"] = design_dict
    # 保存CSV路径信息（便于后续可视化智能体获取）
    result_dict["csv_filename"] = csv_filename
    result_dict["csv_path"] = csv_path
    result_dict["csv_timestamp"] = timestamp
    result_dict["设计参数保存路径"] = csv_filename  # 保留兼容性
    
    print(f"\n{'='*70}")
    print(f"✅ [叶片设计任务] 完成！")
    print(f"{'='*70}")
    print(f"📊 结果摘要:")
    print(f"  • 成功生成 {top_k} 个优化设计方案")
    print(f"  • 每个方案包含21维几何参数")
    print(f"  • 预测性能已验证")
    if save_path:
        print(f"  • CSV文件: {csv_filename}")
        print(f"  • 时间戳: {timestamp}")
    print(f"{'='*70}\n")
    
    _report_progress('design_complete', 100, '叶片设计完成！')
    
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