"""
叶片优化模块（LLM驱动版）
========================

基于LLM驱动的进化优化算法，优化叶片设计参数以提升性能。
与 LLM_OptimizationV1 实现逻辑一致：
- 第1代：LHS（拉丁超立方采样）全局探索
- 第2代及以后：LLM 根据历史生成均值向量，再高斯采样

适配现有多智能体框架的格式和接口。
"""

import os
import sys
import time
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from scipy.stats import qmc

# LLM 相关
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 添加项目路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

# 导入现有的性能评估工具
from .blade_performance_evaluation import blade_performance_evaluation as evaluate_performance

# 全局进度回调函数（用于实时推送进度到前端）
_optimization_progress_callback = None

def set_optimization_progress_callback(callback):
    """设置优化进度回调函数"""
    global _optimization_progress_callback
    _optimization_progress_callback = callback

def _report_optimization_progress(stage: str, progress: float, message: str = "", **kwargs):
    """报告优化进度"""
    if _optimization_progress_callback:
        try:
            data = {
                'stage': stage,
                'progress': progress,
                'message': message
            }
            data.update(kwargs)
            _optimization_progress_callback(data)
        except Exception as e:
            print(f"[优化进度回调错误] {e}")


# ==================== 参数范围定义 ====================

PARAM_RANGES = [
    (40, 70),     # root_Angle_in
    (-50, -20),   # root_Angle_out
    (0.1, 0.3),   # root_Chord
    (5, 25),      # root_THmax_CH
    (0.3, 0.9),   # root_THmaxP
    (0.0, 0.05),  # root_SWA
    (0.0, 0.05),  # root_BOWA
    (40, 70),     # mid_Angle_in
    (15, 35),     # mid_Angle_out
    (0.1, 0.3),   # mid_Chord
    (3, 15),      # mid_THmax_CH
    (0.3, 0.9),   # mid_THmaxP
    (0.0, 0.05),  # mid_SWA
    (0.0, 0.05),  # mid_BOWA
    (50, 80),     # tip_Angle_in
    (50, 70),     # tip_Angle_out
    (0.08, 0.2),  # tip_Chord
    (3, 15),      # tip_THmax_CH
    (0.3, 0.9),   # tip_THmaxP
    (-0.03, 0.03),# tip_SWA
    (0.0, 0.05),  # tip_BOWA
]

# 内部优化范围（0-1000）
INTERNAL_BOUNDS = [(0, 1000)] * 21


# ==================== 辅助函数 ====================

def convert_to_internal_format(actual_params: List[float]) -> List[float]:
    """将实际参数转换为内部优化格式（0-1000范围）"""
    internal = []
    for val, (min_v, max_v) in zip(actual_params, PARAM_RANGES):
        if max_v - min_v != 0:
            normalized = (val - min_v) / (max_v - min_v) * 1000
        else:
            normalized = 500
        internal.append(max(0, min(1000, normalized)))
    return internal


def convert_to_actual_format(internal_params: List[float]) -> List[float]:
    """将内部优化格式转换回实际参数"""
    actual = []
    for val, (min_v, max_v) in zip(internal_params, PARAM_RANGES):
        real_val = min_v + (val / 1000) * (max_v - min_v)
        actual.append(real_val)
    return actual


def clip_vector(vector: List[float]) -> List[int]:
    """将向量限制在边界内并四舍五入为整数"""
    clipped = []
    for val, (min_b, max_b) in zip(vector, INTERNAL_BOUNDS):
        clipped.append(int(round(max(min_b, min(max_b, val)))))
    return clipped


def generate_lhs_samples(population_size: int, num_variables: int = 21, 
                         current_best: Optional[List[float]] = None) -> List[List[int]]:
    """
    使用拉丁超立方采样生成初始种群（用于第1代）
    
    重要：总是保留当前最优解，确保不会丢失好的解
    """
    candidates = []
    
    # ✅ 关键：总是保留当前最优解
    if current_best is not None:
        candidates.append(clip_vector(current_best))
    
    # LHS 采样填充剩余位置
    remaining = population_size - len(candidates)
    if remaining > 0:
        sampler = qmc.LatinHypercube(d=num_variables, seed=int(time.time()) % 10000)
        lhs_samples = sampler.random(n=remaining)
        
        for sample in lhs_samples:
            candidate = []
            for i, val in enumerate(sample):
                min_b, max_b = INTERNAL_BOUNDS[i]
                candidate.append(int(round(min_b + val * (max_b - min_b))))
            candidates.append(candidate)
    
    return candidates


def generate_gaussian_samples(mean_vector: List[float], std: float, population_size: int,
                              current_best: Optional[List[float]] = None) -> List[List[int]]:
    """
    基于均值向量和标准差进行正态分布采样
    
    重要：总是保留当前最优解，确保不会丢失好的解
    """
    candidates = []
    
    # ✅ 关键：总是保留当前最优解
    if current_best is not None:
        candidates.append(clip_vector(current_best))
    
    # 高斯采样填充剩余位置
    remaining = population_size - len(candidates)
    for _ in range(remaining):
        raw_sample = np.random.normal(mean_vector, std)
        candidate = []
        for i, val in enumerate(raw_sample):
            min_b, max_b = INTERNAL_BOUNDS[i]
            candidate.append(int(round(max(min_b, min(max_b, val)))))
        candidates.append(candidate)
    
    return candidates


def calculate_reward(
    performance: Dict[str, float],
    primary_objective: str,
    constraints: Dict[str, Dict],
    target_value: Optional[float] = None,
    target_type: str = "maximize"
) -> float:
    """
    计算综合奖励值
    
    奖励 = 主目标得分 - 违反约束惩罚
    """
    # 主目标得分
    if target_type == "equal" and target_value is not None:
        # 目标是达到特定值
        actual = performance.get(primary_objective, 0)
        reward = 100 - abs(actual - target_value) * 50  # 越接近目标越好
    elif primary_objective == "efficiency":
        reward = performance.get("efficiency", 0) * 100
    elif primary_objective == "flow":
        reward = performance.get("flow", 0)
    elif primary_objective == "pressure_ratio":
        reward = performance.get("pressure_ratio", 0) * 50
    else:
        reward = performance.get("efficiency", 0) * 100
    
    # 约束惩罚
    for metric, constraint in constraints.items():
        actual_val = performance.get(metric, 0)
        constraint_type = constraint.get("type", "min")
        constraint_val = constraint.get("value", 0)
        
        if constraint_type == "min" and actual_val < constraint_val:
            penalty = (constraint_val - actual_val) * 10
            reward -= penalty
        elif constraint_type == "max" and actual_val > constraint_val:
            penalty = (actual_val - constraint_val) * 10
            reward -= penalty
        elif constraint_type == "equal":
            penalty = abs(actual_val - constraint_val) * 5
            reward -= penalty
    
    return reward


def format_history_entry(perf: Dict, candidate: List, primary_objective: str, constraints: Dict) -> str:
    """格式化历史记录条目"""
    reward = perf.get("reward_F", 0)
    primary_val = perf.get(primary_objective, 0)
    
    constraint_vals = []
    for c_name in constraints.keys():
        constraint_vals.append(f"{perf.get(c_name, 0):.4f}")
    
    if constraint_vals:
        return f"{reward:.4f},{primary_val:.4f},{','.join(constraint_vals)}:{candidate}"
    else:
        return f"{reward:.4f},{primary_val:.4f}:{candidate}"


def build_llm_prompt(
    history: str,
    recent_samples: str,
    current_mean: List[float],
    primary_objective: str,
    constraints: Dict,
    target_value: Optional[float] = None,
    target_type: str = "maximize"
) -> str:
    """构建 LLM 提示词"""
    
    num_vars = 21
    var_list = "x1, x2, ..., x21"
    bounds_desc = "x1-x21 ∈ [0, 1000]"
    
    goal_verb = "maximize" if target_type != "minimize" else "minimize"
    
    # 目标描述
    if target_type == "equal" and target_value is not None:
        goal_desc = f"优化 {primary_objective} 使其接近目标值 {target_value}"
    else:
        goal_desc = f"{goal_verb} {primary_objective}"
    
    # 约束描述
    constraint_descs = []
    for i, (name, config) in enumerate(constraints.items(), 2):
        ctype = config.get("type", "min")
        cval = config.get("value", 0)
        if ctype == "min":
            constraint_descs.append(f"- {name}: ≥ {cval}")
        elif ctype == "max":
            constraint_descs.append(f"- {name}: ≤ {cval}")
        else:
            constraint_descs.append(f"- {name}: ≈ {cval}")
    
    constraints_str = "\n".join(constraint_descs) if constraint_descs else "无额外约束"
    
    prompt = f"""### 任务描述
你是一个智能优化器，擅长基于历史解的模式识别和协作搜索，持续改进参数向量以接近全局最优。
目标：{goal_desc}

### 约束条件
{constraints_str}

### 参数范围
F 依赖于 {num_vars} 个整数决策变量 [{var_list}]，范围：{bounds_desc}

### 优化策略
1. 从历史最优解中学习变量耦合关系、搜索趋势和高效区域
2. 平衡全局探索和局部利用（特征交叉 + 变异）
3. 自适应更新均值向量作为下一代的采样中心
4. 确保每代更新支持向全局最优的迭代改进

### 上下文信息
当前均值向量: {current_mean}

历史最优记录（按 F 值降序排列）:
{history}

最近几代采样结果:
{recent_samples}

### 输出要求
基于以上信息，输出新的均值向量。
只输出格式：[x1, x2, ..., x21]
不要包含任何解释或额外文本。
"""
    return prompt


def parse_llm_response(response: str, num_vars: int = 21) -> Optional[List[float]]:
    """解析 LLM 返回的向量"""
    if not response:
        return None
    
    # 尝试多种模式匹配
    patterns = [
        r'\[([^\]]+)\]',  # [x1, x2, ...]
        r'\(([^\)]+)\)',  # (x1, x2, ...)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                values = [float(x.strip()) for x in match.split(',')]
                if len(values) == num_vars:
                    return clip_vector(values)
            except:
                continue
    
    return None


def safe_llm_call(llm, prompt: str, max_retries: int = 3, verbose: bool = False) -> Optional[str]:
    """安全的 LLM 调用，带重试机制"""
    for attempt in range(1, max_retries + 1):
        try:
            if verbose:
                print(f"   [LLM] 调用尝试 {attempt}/{max_retries}...")
            
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            if content and content.strip():
                return content.strip()
            
        except Exception as e:
            if verbose:
                print(f"   [LLM] 调用失败: {e}")
            if attempt < max_retries:
                wait = min(30, 5 * (2 ** (attempt - 1)))
                time.sleep(wait)
    
    return None


# ==================== 主优化函数 ====================

def blade_optimization(
    initial_design: List[float],
    initial_performance: Dict[str, float],
    optimization_config: Optional[Dict[str, Any]] = None,
    max_generations: int = 10,
    population_size: int = 5,
    verbose: bool = True,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    优化叶片设计参数（LLM驱动版）
    
    与 LLM_OptimizationV1 实现逻辑一致：
    - 第1代：LHS 全局探索
    - 第2代及以后：LLM 根据历史生成均值向量，再高斯采样
    
    Args:
        initial_design: 21维初始设计参数列表
        initial_performance: 初始性能字典
        optimization_config: 优化配置
        max_generations: 最大迭代代数
        population_size: 每代种群大小
        verbose: 是否打印详细过程
        progress_callback: 进度回调函数
        
    Returns:
        优化结果字典
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🔧 [叶片优化] LLM驱动优化开始")
        print(f"{'='*70}")
    
    # ==================== 输入验证 ====================
    if len(initial_design) != 21:
        return {
            "success": False,
            "error": f"设计参数维度错误：期望21维，实际{len(initial_design)}维",
            "message": "优化失败：设计参数维度不正确"
        }
    
    required_metrics = ["flow", "efficiency", "pressure_ratio"]
    for metric in required_metrics:
        if metric not in initial_performance:
            return {
                "success": False,
                "error": f"缺少必要的性能指标：{metric}",
                "message": f"优化失败：初始性能中缺少{metric}"
            }
    
    # ==================== 配置处理 ====================
    config = optimization_config or {}
    primary_objective = config.get("primary_objective", "efficiency")
    constraints = config.get("constraints", {})
    target_value = config.get("target_value")
    target_type = config.get("target_type", "maximize")
    
    # 默认约束：保持流量和压比不退化
    if not constraints:
        constraints = {
            "flow": {"type": "min", "value": initial_performance["flow"] * 0.98},
            "pressure_ratio": {"type": "min", "value": initial_performance["pressure_ratio"] * 0.98}
        }
    
    # 方差调度：从100逐渐减小
    variance_schedule = [
        (1, 2, 120),   # 第1-2代：大方差探索
        (3, 5, 80),    # 第3-5代：中等方差
        (6, 8, 50),    # 第6-8代：较小方差
        (9, 999, 30),  # 第9代以后：小方差收敛
    ]
    
    # 主目标名称映射（全局使用）
    objective_names_global = {"flow": "流量", "efficiency": "效率", "pressure_ratio": "压比"}
    primary_obj_name = objective_names_global.get(primary_objective, primary_objective)
    
    if verbose:
        print(f"\n🔧 优化目标: {primary_obj_name} | 最大代数: {max_generations}")
        print(f"📊 初始: 流量={initial_performance['flow']:.3f}, 效率={initial_performance['efficiency']:.4f}, 压比={initial_performance['pressure_ratio']:.3f}")
    
    # ==================== 初始化 LLM ====================
    try:
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            temperature=0.5,
            max_retries=3,
        )
        if verbose:
            print(f"✅ LLM 初始化成功")
    except Exception as e:
        if verbose:
            print(f"⚠️ LLM 初始化失败: {e}，将使用纯数值优化")
        llm = None
    
    # ==================== 初始化优化状态 ====================
    initial_internal = convert_to_internal_format(initial_design)
    current_mean = initial_internal.copy()
    
    # 计算初始奖励
    initial_reward = calculate_reward(
        initial_performance, primary_objective, constraints, target_value, target_type
    )
    initial_performance["reward_F"] = initial_reward
    
    best_design = initial_internal.copy()
    best_performance = initial_performance.copy()
    
    # 历史记录
    history_lines = []
    recent_generation_data = []
    optimization_history = [{
        "generation": 0,
        "best_performance": best_performance.copy(),
        "best_design": best_design.copy(),
        "timestamp": time.time()
    }]
    
    if progress_callback:
        progress_callback(0, max_generations, "开始优化...")
    
    # 发送优化开始信号
    _report_optimization_progress('optimization_start', 0, '开始LLM驱动优化...')
    
    # ==================== 优化循环 ====================
    no_improve_count = 0
    patience = 5
    convergence_threshold = 0.0001
    
    # 计算可视化检查点（每20%生成一张图片）
    viz_checkpoints = set()
    for percent in [20, 40, 60, 80, 100]:
        checkpoint_gen = max(1, round(max_generations * percent / 100))
        viz_checkpoints.add(checkpoint_gen)
    print(f"[优化] 可视化检查点: {sorted(viz_checkpoints)}")
    
    for generation in range(1, max_generations + 1):
        if verbose:
            print(f"\n--- 第 {generation}/{max_generations} 代 ---")
        
        # 报告进度到前端
        progress_percent = generation / max_generations * 100
        _report_optimization_progress(
            'optimization', 
            progress_percent, 
            f'优化迭代: {generation}/{max_generations} ({progress_percent:.0f}%)'
        )
        
        # 获取当前方差
        current_std = 100.0
        for start_gen, end_gen, std in variance_schedule:
            if start_gen <= generation <= end_gen:
                current_std = std
                break
        
        # ========== 生成候选解 ==========
        if generation == 1:
            # 第1代：LHS 全局探索（但保留当前最优解）
            if verbose:
                print(f"   [LHS] 使用拉丁超立方采样进行全局探索（保留初始解）")
            candidates = generate_lhs_samples(population_size, current_best=best_design)
        else:
            # 第2代及以后：尝试用 LLM 生成新均值
            new_mean = None
            
            if llm is not None:
                # 构建历史记录
                history_str = "\n".join(history_lines[-5:]) if history_lines else "无历史记录"
                
                # 构建最近样本记录
                recent_str = ""
                if recent_generation_data:
                    recent_entries = []
                    for data in recent_generation_data[-3:]:
                        entry = format_history_entry(
                            data["best_perf"], data["best_candidate"],
                            primary_objective, constraints
                        )
                        recent_entries.append(entry)
                    recent_str = "\n".join(recent_entries)
                
                # 构建 Prompt
                prompt = build_llm_prompt(
                    history_str, recent_str, current_mean,
                    primary_objective, constraints, target_value, target_type
                )
                
                if verbose:
                    print(f"   [LLM] 请求生成新均值向量...")
                
                # 调用 LLM
                response = safe_llm_call(llm, prompt, max_retries=2, verbose=verbose)
                
                if response:
                    new_mean = parse_llm_response(response)
                    if new_mean and verbose:
                        print(f"   [LLM] ✅ 成功获取新均值向量")
            
            # 如果 LLM 失败，使用当前最优解作为均值
            if new_mean is None:
                if verbose:
                    print(f"   [Fallback] 使用当前最优解作为均值")
                new_mean = best_design.copy()
            
            current_mean = new_mean
            
            # 基于新均值进行高斯采样（保留当前最优解）
            candidates = generate_gaussian_samples(current_mean, current_std, population_size, current_best=best_design)
        
        if verbose:
            print(f"   [采样] 生成 {len(candidates)} 个候选解，std={current_std}")
        
        # ========== 评估所有候选解 ==========
        performances = []
        for i, candidate in enumerate(candidates):
            actual_params = convert_to_actual_format(candidate)
            
            try:
                eval_result = evaluate_performance([actual_params], verbose=False)
                first_result = eval_result.get("第1个设计性能结果", {})
                
                perf = {
                    "flow": first_result.get("mass flow rate（预测流量[kg/s]）", 0),
                    "efficiency": first_result.get("isentropic efficiency（预测等熵效率[-]）", 0),
                    "pressure_ratio": first_result.get("total pressure ratio（预测总压比[-]）", 0)
                }
                
                perf["reward_F"] = calculate_reward(
                    perf, primary_objective, constraints, target_value, target_type
                )
                performances.append((candidate, perf))
                
            except Exception as e:
                if verbose:
                    print(f"   [评估] 候选 {i+1} 评估失败: {e}")
        
        if not performances:
            no_improve_count += 1
            if verbose:
                print(f"   [警告] 本代无有效候选解")
            continue
        
        # 按奖励排序（降序）
        performances.sort(key=lambda x: x[1]["reward_F"], reverse=True)
        gen_best_candidate, gen_best_perf = performances[0]
        
        # 主目标名称映射
        objective_names = {"flow": "流量", "efficiency": "效率", "pressure_ratio": "压比"}
        obj_name = objective_names.get(primary_objective, primary_objective)
        obj_value = gen_best_perf.get(primary_objective, 0)
        
        if verbose:
            print(f"   [评估] 本代最优: reward={gen_best_perf['reward_F']:.4f}, "
                  f"{obj_name}={obj_value:.4f}")
        
        # ========== 更新最优解 ==========
        improved = gen_best_perf["reward_F"] > best_performance.get("reward_F", -float("inf"))
        
        if improved:
            best_design = gen_best_candidate.copy()
            best_performance = gen_best_perf.copy()
            no_improve_count = 0
            if verbose:
                best_obj_value = best_performance.get(primary_objective, 0)
                print(f"   ✅ 找到更优解！{obj_name}={best_obj_value:.4f}")
        else:
            no_improve_count += 1
            if verbose:
                print(f"   ➖ 无改进 (连续{no_improve_count}代)")
        
        # ========== 更新历史记录 ==========
        history_entry = format_history_entry(
            gen_best_perf, gen_best_candidate, primary_objective, constraints
        )
        history_lines.append(history_entry)
        
        recent_generation_data.append({
            "generation": generation,
            "best_candidate": gen_best_candidate,
            "best_perf": gen_best_perf,
            "all_performances": performances
        })
        
        optimization_history.append({
            "generation": generation,
            "best_performance": best_performance.copy(),
            "best_design": best_design.copy(),
            "gen_best_reward": gen_best_perf["reward_F"],
            "improved": improved,
            "timestamp": time.time()
        })
        
        # 进度回调
        if progress_callback:
            progress_callback(
                generation, max_generations,
                f"第{generation}代完成，{primary_obj_name}={best_performance.get(primary_objective, 0):.4f}"
            )
        
        # ========== 生成可视化检查点图片 ==========
        if generation in viz_checkpoints:
            try:
                # 获取当前最优设计的实际参数
                current_best_actual = convert_to_actual_format(best_design)
                viz_label = f"{int(progress_percent)}%"
                
                # 生成可视化图片
                from .blade_visualization import visualize_blade
                
                viz_params = {
                    'root_Angle_in': current_best_actual[0],
                    'root_Angle_out': current_best_actual[1],
                    'root_Chord': current_best_actual[2],
                    'root_THmax_CH': current_best_actual[3],
                    'root_THmaxP': current_best_actual[4],
                    'root_SWA': current_best_actual[5],
                    'root_BOWA': current_best_actual[6],
                    'mid_Angle_in': current_best_actual[7],
                    'mid_Angle_out': current_best_actual[8],
                    'mid_Chord': current_best_actual[9],
                    'mid_THmax_CH': current_best_actual[10],
                    'mid_THmaxP': current_best_actual[11],
                    'mid_SWA': current_best_actual[12],
                    'mid_BOWA': current_best_actual[13],
                    'tip_Angle_in': current_best_actual[14],
                    'tip_Angle_out': current_best_actual[15],
                    'tip_Chord': current_best_actual[16],
                    'tip_THmax_CH': current_best_actual[17],
                    'tip_THmaxP': current_best_actual[18],
                    'tip_SWA': current_best_actual[19],
                    'tip_BOWA': current_best_actual[20],
                }
                
                viz_result = visualize_blade(**viz_params, save_image=True, send_sse=False)
                
                if viz_result.get('status') == 'success' and viz_result.get('image_path'):
                    image_path = viz_result['image_path']
                    print(f"   📸 生成优化中间图片: {viz_label}")
                    
                    # 发送图片到前端
                    _report_optimization_progress(
                        'optimization_image',
                        progress_percent,
                        viz_label,
                        image_path=image_path,
                        label=f"优化 {viz_label}"
                    )
            except Exception as e:
                print(f"   ⚠️ 生成优化中间图片失败: {e}")
        
        # ========== 收敛检查 ==========
        if no_improve_count >= patience:
            if verbose:
                print(f"\n⏹️ 提前停止（连续{patience}代无改进）")
            
            # 提前停止时，生成最终状态的可视化图片
            try:
                from .blade_visualization import visualize_blade
                final_actual = convert_to_actual_format(best_design)
                final_viz_params = {
                    'root_Angle_in': final_actual[0], 'root_Angle_out': final_actual[1],
                    'root_Chord': final_actual[2], 'root_THmax_CH': final_actual[3],
                    'root_THmaxP': final_actual[4], 'root_SWA': final_actual[5],
                    'root_BOWA': final_actual[6], 'mid_Angle_in': final_actual[7],
                    'mid_Angle_out': final_actual[8], 'mid_Chord': final_actual[9],
                    'mid_THmax_CH': final_actual[10], 'mid_THmaxP': final_actual[11],
                    'mid_SWA': final_actual[12], 'mid_BOWA': final_actual[13],
                    'tip_Angle_in': final_actual[14], 'tip_Angle_out': final_actual[15],
                    'tip_Chord': final_actual[16], 'tip_THmax_CH': final_actual[17],
                    'tip_THmaxP': final_actual[18], 'tip_SWA': final_actual[19],
                    'tip_BOWA': final_actual[20],
                }
                final_viz_result = visualize_blade(**final_viz_params, save_image=True, send_sse=False)
                if final_viz_result.get('status') == 'success' and final_viz_result.get('image_path'):
                    print(f"   📸 生成优化最终图片: 收敛停止")
                    _report_optimization_progress(
                        'optimization_image', 100, '最终结果',
                        image_path=final_viz_result['image_path'], label='优化 最终结果'
                    )
            except Exception as e:
                print(f"   ⚠️ 生成最终优化图片失败: {e}")
            
            # 发送优化完成信号
            _report_optimization_progress('optimization_complete', 100, '优化提前收敛')
            break
    
    # 如果正常结束（没有提前停止），发送完成信号
    if no_improve_count < patience:
        _report_optimization_progress('optimization_complete', 100, '优化迭代完成')
    
    # ==================== 结果整理 ====================
    optimized_design = convert_to_actual_format(best_design)
    
    # 计算改进幅度
    improvement = {}
    for metric in ["flow", "efficiency", "pressure_ratio"]:
        init_val = initial_performance.get(metric, 0)
        opt_val = best_performance.get(metric, 0)
        if init_val != 0:
            improvement[metric] = {
                "before": init_val,
                "after": opt_val,
                "change": opt_val - init_val,
                "percent": (opt_val - init_val) / abs(init_val) * 100
            }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ [叶片优化] 优化完成！")
        print(f"{'='*70}")
        for metric, imp in improvement.items():
            metric_name = {"flow": "流量", "efficiency": "效率", "pressure_ratio": "压比"}.get(metric, metric)
            print(f"  {metric_name}: {imp['before']:.4f} → {imp['after']:.4f} ({imp['percent']:+.2f}%)")
    
    # 根据实际优化目标生成结果说明
    metric_names = {"flow": "流量", "efficiency": "效率", "pressure_ratio": "压比"}
    primary_metric_name = metric_names.get(primary_objective, primary_objective)
    primary_improvement = improvement.get(primary_objective, {})
    
    # 构建完整的结果说明
    message_parts = []
    
    # 主要优化目标的变化
    if primary_improvement:
        primary_change = primary_improvement.get('percent', 0)
        if primary_change > 0:
            message_parts.append(f"{primary_metric_name}提升了{abs(primary_change):.2f}%")
        elif primary_change < 0:
            message_parts.append(f"{primary_metric_name}降低了{abs(primary_change):.2f}%")
        else:
            message_parts.append(f"{primary_metric_name}保持不变")
    
    # 添加其他指标的变化情况
    other_changes = []
    for metric in ["flow", "efficiency", "pressure_ratio"]:
        if metric != primary_objective and metric in improvement:
            imp = improvement[metric]
            metric_name = metric_names.get(metric, metric)
            change = imp.get('percent', 0)
            if abs(change) >= 0.1:  # 变化超过0.1%才显示
                if change > 0:
                    other_changes.append(f"{metric_name}+{change:.2f}%")
                else:
                    other_changes.append(f"{metric_name}{change:.2f}%")
    
    if other_changes:
        message_parts.append(f"同时{', '.join(other_changes)}")
    
    result_message = "优化完成，" + "，".join(message_parts)
    
    return {
        "success": True,
        "optimized_design": optimized_design,
        "optimized_performance": best_performance,
        "initial_performance": initial_performance,
        "improvement": improvement,
        "total_generations": len(optimization_history) - 1,
        "optimization_history": optimization_history,
        "optimization_config": {
            "primary_objective": primary_objective,
            "constraints": constraints,
            "target_value": target_value,
            "target_type": target_type,
            "max_generations": max_generations,
            "population_size": population_size,
            "llm_enabled": llm is not None
        },
        "message": result_message
    }


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("测试 LLM 驱动的叶片优化模块...")
    
    test_design = [57, -33, 0.17, 15.73, 0.76, 0.01, 0.01,
                   55, 25, 0.2, 6.7, 0.6, 0.02, 0.01,
                   68, 61, 0.13, 6.3, 0.6, -0.01, 0.02]
    
    test_performance = {
        "flow": 15.2,
        "efficiency": 0.83,
        "pressure_ratio": 1.52
    }
    
    result = blade_optimization(
        initial_design=test_design,
        initial_performance=test_performance,
        optimization_config={
            "primary_objective": "efficiency",
            "constraints": {
                "flow": {"type": "min", "value": 15.0},
                "pressure_ratio": {"type": "min", "value": 1.5}
            }
        },
        max_generations=5,
        population_size=3,
        verbose=True
    )
    
    if result["success"]:
        print(f"\n优化成功！")
        print(f"效率提升: {result['improvement']['efficiency']['percent']:.2f}%")
        print(f"LLM 参与: {result['optimization_config']['llm_enabled']}")
    else:
        print(f"\n优化失败: {result.get('error', 'Unknown error')}")
