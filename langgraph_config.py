"""
LangGraph 配置文件 - 状态定义与类型
====================================

修改目的：
1. 定义全局状态结构，实现可追踪的状态流转
2. 使用TypedDict确保类型安全
3. 为工作流提供统一的数据接口

为什么需要这个文件：
- 原有架构：状态隐式存储在对话历史中，难以追踪和管理
- 优化后：显式定义状态，每个节点的输入输出清晰可控
"""

from typing import TypedDict, Literal, Optional, Dict, List, Any, Annotated
from langgraph.graph.message import add_messages


class DesignState(TypedDict):
    """
    航发叶片设计系统的全局状态
    
    状态字段说明：
    - messages: 对话历史
    - user_input: 用户原始输入
    - intent: 识别出的用户意图（design/evaluate/visualize/optimize）
    - performance_targets: 性能目标参数（3维：流量、效率、压比）
    - design_params: 设计参数（21维：叶根/叶中/叶尖各7个参数）
    - evaluation_results: 性能评估结果
    - visualization_path: 可视化图像路径
    - optimization_history: 优化历史记录
    - iteration_count: 迭代次数
    - error_message: 错误信息
    - next_action: 下一步动作指令
    """
    
    # 对话管理（使用 add_messages 支持更好的消息管理）
    messages: Annotated[List, add_messages]
    
    # 用户输入与意图
    user_input: str
    intent: Literal["design", "evaluate", "visualize", "optimize", "unknown"]
    
    # 核心数据
    performance_targets: Optional[Dict[str, float]]  # {flow_rate, efficiency, pressure_ratio}
    design_params: Optional[Dict[str, Any]]  # 21维设计参数
    
    # 执行结果
    evaluation_results: Optional[Dict[str, Any]]
    visualization_path: Optional[str]
    
    # 优化相关
    optimization_history: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int
    
    # 错误处理
    error_message: Optional[str]
    
    # 流程控制
    next_action: Optional[str]


class IntentExtractionResult(TypedDict):
    """意图识别结果"""
    intent: str
    confidence: float
    extracted_params: Dict[str, Any]
    reasoning: str


class OptimizationConfig(TypedDict):
    """优化配置"""
    max_iterations: int
    convergence_threshold: float
    optimization_strategy: Literal["gradient", "genetic", "bayesian"]


# 意图类型枚举
INTENT_TYPES = {
    "design": "生成新设计",
    "evaluate": "评估已有设计",
    "visualize": "可视化叶片几何",
    "optimize": "迭代优化设计",
    "unknown": "意图不明确"
}


# 节点名称常量（用于工作流构建）
class NodeNames:
    """节点名称枚举"""
    INTENT_RECOGNITION = "intent_recognition"  # 意图识别节点
    DESIGN_GENERATION = "design_generation"    # 设计生成节点
    PERFORMANCE_EVAL = "performance_evaluation"  # 性能评估节点
    VISUALIZATION = "visualization"             # 可视化节点
    OPTIMIZATION_LOOP = "optimization_loop"     # 优化循环节点
    RESULT_SYNTHESIS = "result_synthesis"       # 结果整合节点
    ERROR_HANDLER = "error_handler"             # 错误处理节点
    END = "__end__"                            # 结束节点


# 边条件判断结果
class EdgeConditions:
    """边条件枚举"""
    TO_DESIGN = "to_design"
    TO_EVALUATE = "to_evaluate"
    TO_VISUALIZE = "to_visualize"
    TO_OPTIMIZE = "to_optimize"
    TO_END = "to_end"
    TO_ERROR = "to_error"
    CONTINUE_OPTIMIZATION = "continue"
    FINISH_OPTIMIZATION = "finish"

