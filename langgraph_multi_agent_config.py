"""
LangGraph 多智能体配置
====================

定义多智能体系统的状态、Agent类型和通信协议
"""

from typing import TypedDict, Literal, Optional, Dict, List, Any, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# ==================== 状态定义 ====================

class MultiAgentState(TypedDict):
    """
    多智能体系统的全局状态（增强版）
    
    与单智能体的区别：
    - 增加了 agent 间通信字段
    - 增加了用户交互字段（等待用户输入）
    - 增加了协作历史
    - 【新增】多任务编排能力
    - 【新增】结果存储与传递
    - 【新增】智能意图判断
    """
    
    # 用户交互
    messages: Annotated[List[BaseMessage], add_messages]  # 完整对话历史
    user_input: str  # 当前用户输入
    
    # 用户交互控制（反问功能）
    waiting_for_user: bool  # 是否等待用户输入
    user_question: Optional[str]  # 要问用户的问题
    pending_agent: Optional[str]  # 等待用户回答后应该恢复哪个Agent
    
    # Agent 协作
    current_agent: str  # 当前执行的Agent
    target_agent: Optional[str]  # 目标Agent（用于路由）
    agent_messages: List[Dict[str, Any]]  # Agent间的消息队列
    
    # 意图与参数
    intent: Literal["design", "evaluate", "visualize", "optimize", "chat", "unknown"]
    performance_targets: Optional[Dict[str, float]]  # {flow_rate, efficiency, pressure_ratio}
    design_params: Optional[Dict[str, Any]]  # 21维设计参数
    partial_blade_params: Optional[Dict[str, float]]  # 用户逐步提供的21维参数（累积）
    
    # 执行结果
    evaluation_results: Optional[Dict[str, Any]]
    visualization_path: Optional[str]
    visualization_json_path: Optional[str]  # 3D数据JSON路径（用于前端Three.js交互）
    
    # 优化相关
    optimization_history: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int
    
    # 协作历史
    collaboration_log: List[Dict[str, Any]]  # 记录Agent间的协作
    
    # 错误与流程控制
    error_message: Optional[str]
    next_action: Optional[str]
    final_result: Optional[str]
    
    # ==================== 新增：多任务编排 ====================
    task_mode: Literal["single", "sequential"]  # 任务模式：单任务 | 顺序执行
    task_queue: List[str]  # 任务队列：["design", "evaluate", "visualize"]
    current_task_index: int  # 当前执行到第几个任务
    original_user_input: Optional[str]  # 【新增】原始用户输入，用于多任务序列中后续任务提取参数
    
    # ==================== 新增：结果存储与传递 ====================
    task_results: Dict[str, Any]  # 存储各任务结果：{"design": {...}, "evaluate": {...}}
    last_task: Optional[str]  # 上次完成的任务类型
    last_task_time: Optional[float]  # 上次任务完成的时间戳
    
    # ==================== 新增：意图细化 ====================
    intent_type: Literal["new", "continue", "use_existing", "sequence"]  
    # new: 全新任务 | continue: 继续提供参数 | use_existing: 使用已有结果 | sequence: 多任务序列
    use_existing_result: Optional[bool]  # 是否使用已有结果（None表示需询问）
    awaiting_confirmation: bool  # 是否正在等待用户确认使用已有结果
    
    # ==================== 新增：多任务中间结果累积 ====================
    accumulated_results: Optional[List[Dict[str, Any]]]  # 累积的中间结果列表
    
    # ==================== 新增：优化智能体专用字段 ====================
    optimization_config: Optional[Dict[str, Any]]  # 优化配置（目标、约束等）
    optimization_result: Optional[Dict[str, Any]]  # 优化结果
    optimization_in_progress: bool  # 优化是否正在进行中


# ==================== Agent 类型定义 ====================

class AgentNames:
    """Agent 名称常量"""
    COORDINATOR = "coordinator"  # 协调者
    DESIGN_EXPERT = "design_expert"  # 设计专家
    EVALUATION_EXPERT = "evaluation_expert"  # 评估专家
    VISUALIZATION_EXPERT = "visualization_expert"  # 可视化专家
    OPTIMIZATION_EXPERT = "optimization_expert"  # 优化专家
    CHAT_EXPERT = "chat_expert"  # 闲聊专家
    USER_INTERFACE = "user_interface"  # 用户交互接口


# ==================== 消息类型定义 ====================

class MessageType:
    """Agent间消息类型"""
    TASK_REQUEST = "task_request"  # 任务请求
    TASK_RESULT = "task_result"  # 任务结果
    QUESTION = "question"  # 提问
    ANSWER = "answer"  # 回答
    FEEDBACK = "feedback"  # 反馈
    ERROR = "error"  # 错误


# ==================== 边条件定义 ====================

class EdgeConditions:
    """边条件枚举"""
    # Agent 路由
    TO_COORDINATOR = "to_coordinator"
    TO_DESIGN = "to_design"
    TO_EVALUATION = "to_evaluation"
    TO_VISUALIZATION = "to_visualization"
    TO_OPTIMIZATION = "to_optimization"  # 优化专家
    TO_CHAT = "to_chat"  # 闲聊专家
    
    # 用户交互
    TO_USER = "to_user"  # 需要用户输入
    FROM_USER = "from_user"  # 收到用户输入
    
    # 流程控制
    TO_END = "to_end"
    TO_ERROR = "to_error"
    CONTINUE = "continue"
    
    # 优化循环
    CONTINUE_OPTIMIZATION = "continue_optimization"
    FINISH_OPTIMIZATION = "finish_optimization"


# ==================== 工具函数 ====================

def create_agent_message(
    from_agent: str,
    to_agent: str,
    message_type: str,
    content: Any
) -> Dict[str, Any]:
    """
    创建Agent间的标准消息
    
    参数：
        from_agent: 发送者Agent名称
        to_agent: 接收者Agent名称
        message_type: 消息类型（MessageType中的值）
        content: 消息内容
    
    返回：
        标准化的消息字典
    """
    from datetime import datetime
    
    return {
        "from": from_agent,
        "to": to_agent,
        "type": message_type,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }


def log_collaboration(
    state: MultiAgentState,
    from_agent: str,
    to_agent: str,
    action: str,
    details: Any = None
) -> None:
    """
    记录Agent协作日志
    
    参数：
        state: 当前状态
        from_agent: 发起者
        to_agent: 目标者
        action: 动作描述
        details: 详细信息
    """
    from datetime import datetime
    
    if "collaboration_log" not in state:
        state["collaboration_log"] = []
    
    state["collaboration_log"].append({
        "from": from_agent,
        "to": to_agent,
        "action": action,
        "details": details,
        "timestamp": datetime.now().isoformat()
    })


# ==================== Agent 配置 ====================

AGENT_CONFIGS = {
    AgentNames.COORDINATOR: {
        "role": "项目协调者",
        "description": "负责理解用户需求、分配任务给专家、整合结果",
        "capabilities": [
            "分析用户需求",
            "识别缺失信息",
            "询问用户补充信息",
            "分配任务给专家",
            "整合各专家结果",
            "生成最终报告"
        ],
        "can_ask_user": True
    },
    
    AgentNames.DESIGN_EXPERT: {
        "role": "叶片设计专家",
        "description": "负责生成叶片设计方案",
        "capabilities": [
            "根据性能目标生成设计参数",
            "检查参数完整性",
            "询问缺失的性能目标",
            "调用设计工具"
        ],
        "can_ask_user": True
    },
    
    AgentNames.EVALUATION_EXPERT: {
        "role": "性能评估专家",
        "description": "负责评估设计性能",
        "capabilities": [
            "评估设计参数的性能",
            "判断是否满足目标",
            "提供改进建议",
            "调用评估工具"
        ],
        "can_ask_user": False
    },
    
    AgentNames.VISUALIZATION_EXPERT: {
        "role": "可视化专家",
        "description": "负责生成3D可视化图",
        "capabilities": [
            "生成叶片3D几何图",
            "调用可视化工具"
        ],
        "can_ask_user": False
    },
    
    AgentNames.CHAT_EXPERT: {
        "role": "闲聊专家",
        "description": "处理与叶片设计无关的问题，并引导用户回到主题",
        "capabilities": [
            "识别问题类型（功能询问/普通闲聊）",
            "生成自然的对话回复",
            "介绍系统功能",
            "引导用户回到叶片设计主题"
        ],
        "can_ask_user": False
    },
    
    AgentNames.OPTIMIZATION_EXPERT: {
        "role": "优化专家",
        "description": "负责优化叶片设计，提升性能指标",
        "capabilities": [
            "分析现有设计的优化潜力",
            "设置优化目标和约束",
            "执行LLM驱动的进化优化",
            "输出优化后的设计方案",
            "对比优化前后性能变化"
        ],
        "can_ask_user": True
    }
}


# ==================== 默认值 ====================

DEFAULT_PERFORMANCE_TARGETS = {
    "flow_rate": 15.0,  # kg/s
    "efficiency": 0.85,  # 0-1
    "pressure_ratio": 1.5  # >1
}

MAX_ITERATIONS = 5  # 最大优化迭代次数

