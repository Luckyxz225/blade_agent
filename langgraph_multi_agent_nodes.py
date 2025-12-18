"""
LangGraph 多智能体节点实现（增强版）
===================================

实现各个Agent节点、任务解析器、任务调度器和用户交互节点
新增：多任务编排、智能意图判断、结果传递
"""

from typing import Dict, Any, List, Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph_multi_agent_config import (
    MultiAgentState, AgentNames, MessageType, EdgeConditions,
    create_agent_message, log_collaboration, AGENT_CONFIGS,
    DEFAULT_PERFORMANCE_TARGETS
)
from langgraph_tools import (
    blade_design_tool,
    blade_performance_evaluation_tool,
    blade_visualization_tool,
    blade_optimization_tool
)
import json
import re
import os
import time


# ==================== 任务结果回调（用于实时输出中间结果到前端）====================

_task_result_callback = None

def set_task_result_callback(callback):
    """设置任务结果回调函数，用于在多任务执行中实时输出每个任务的结果"""
    global _task_result_callback
    _task_result_callback = callback

def _send_task_result(task: str, result: str, task_index: int, total_tasks: int, 
                      image_path: str = None, json_path: str = None):
    """发送任务结果到前端"""
    if _task_result_callback:
        try:
            data = {
                'stage': 'task_result',
                'task': task,
                'result': result,
                'task_index': task_index,
                'total_tasks': total_tasks,
            }
            if image_path:
                data['image_path'] = image_path
            if json_path:
                data['json_path'] = json_path
            _task_result_callback(data)
            print(f"[TaskResult] 发送任务结果到前端: {task} ({task_index}/{total_tasks})")
        except Exception as e:
            print(f"[TaskResult回调错误] {e}")


# ==================== 21维叶片设计参数定义 ====================

BLADE_21D_PARAMS = {
    # 叶根参数（root）
    "root_Angle_in": {"name": "叶根进口金属角", "unit": "°", "range": (40, 70), "default": 57},
    "root_Angle_out": {"name": "叶根出口金属角", "unit": "°", "range": (-50, -20), "default": -33},
    "root_Chord": {"name": "叶根弦长", "unit": "m", "range": (0.1, 0.3), "default": 0.17},
    "root_THmax_CH": {"name": "叶根最大相对厚度", "unit": "%", "range": (5, 25), "default": 15.73},
    "root_THmaxP": {"name": "叶根最大相对厚度位置", "unit": "-", "range": (0.3, 0.9), "default": 0.76},
    "root_SWA": {"name": "叶根弯度", "unit": "m", "range": (0.0, 0.05), "default": 0.01},
    "root_BOWA": {"name": "叶根掠", "unit": "m", "range": (0.0, 0.05), "default": 0.01},
    
    # 叶中参数（mid）
    "mid_Angle_in": {"name": "叶中进口金属角", "unit": "°", "range": (40, 70), "default": 55},
    "mid_Angle_out": {"name": "叶中出口金属角", "unit": "°", "range": (15, 35), "default": 25},
    "mid_Chord": {"name": "叶中弦长", "unit": "m", "range": (0.1, 0.3), "default": 0.2},
    "mid_THmax_CH": {"name": "叶中最大相对厚度", "unit": "%", "range": (3, 15), "default": 6.7},
    "mid_THmaxP": {"name": "叶中最大相对厚度位置", "unit": "-", "range": (0.3, 0.9), "default": 0.6},
    "mid_SWA": {"name": "叶中弯度", "unit": "m", "range": (0.0, 0.05), "default": 0.02},
    "mid_BOWA": {"name": "叶中掠", "unit": "m", "range": (0.0, 0.05), "default": 0.01},
    
    # 叶顶参数（tip）
    "tip_Angle_in": {"name": "叶顶进口金属角", "unit": "°", "range": (50, 80), "default": 68},
    "tip_Angle_out": {"name": "叶顶出口金属角", "unit": "°", "range": (50, 70), "default": 61},
    "tip_Chord": {"name": "叶顶弦长", "unit": "m", "range": (0.08, 0.2), "default": 0.13},
    "tip_THmax_CH": {"name": "叶顶最大相对厚度", "unit": "%", "range": (3, 15), "default": 6.3},
    "tip_THmaxP": {"name": "叶顶最大相对厚度位置", "unit": "-", "range": (0.3, 0.9), "default": 0.6},
    "tip_SWA": {"name": "叶顶弯度", "unit": "m", "range": (-0.03, 0.03), "default": -0.01},
    "tip_BOWA": {"name": "叶顶掠", "unit": "m", "range": (0.0, 0.05), "default": 0.02}
}

BLADE_21D_PARAMS_ORDER = [
    "root_Angle_in", "root_Angle_out", "root_Chord", "root_THmax_CH", "root_THmaxP", "root_SWA", "root_BOWA",
    "mid_Angle_in", "mid_Angle_out", "mid_Chord", "mid_THmax_CH", "mid_THmaxP", "mid_SWA", "mid_BOWA",
    "tip_Angle_in", "tip_Angle_out", "tip_Chord", "tip_THmax_CH", "tip_THmaxP", "tip_SWA", "tip_BOWA"
]


# ==================== 辅助函数 ====================

def parse_user_blade_params(user_input: str, llm: ChatOpenAI = None) -> Dict[str, float]:
    """
    从用户输入中解析21维叶片参数
    
    返回: Dict[param_name, value] - 用户提供的参数
    """
    extracted_params = {}
    
    # 使用LLM提取参数
    if llm:
        try:
            extraction_prompt = f"""从用户输入中提取叶片几何参数，返回JSON格式。

用户输入：{user_input}

参数列表（共21个）：
{', '.join(BLADE_21D_PARAMS_ORDER)}

请识别用户提到的参数及其数值，返回JSON格式：
{{
    "root_Chord": 0.6,
    "mid_Angle_in": 55,
    ...
}}

只提取明确提到的参数，其他参数不要包含在JSON中。
如果用户没有提供任何有效参数，返回空JSON：{{}}

**重要**：必须返回有效的JSON格式！"""
            
            response = llm.invoke([HumanMessage(content=extraction_prompt)])
            result = extract_json_from_text(response.content)
            if result:
                extracted_params = result
                print(f"[ParameterExtraction] LLM提取到 {len(extracted_params)} 个参数")
        except Exception as e:
            print(f"[ParameterExtraction] LLM提取失败: {e}")
    
    # Fallback：关键词匹配
    if not extracted_params:
        print(f"[ParameterExtraction] 尝试关键词匹配...")
        # 尝试匹配 "叶根弦长为0.6m"、"root_Chord=0.6" 等格式
        for param_key, param_info in BLADE_21D_PARAMS.items():
            param_name = param_info["name"]
            
            # 匹配中文描述
            pattern = f"{param_name}[为是:=]\\s*([0-9.]+)"
            match = re.search(pattern, user_input)
            if match:
                extracted_params[param_key] = float(match.group(1))
                continue
            
            # 匹配英文键名
            pattern = f"{param_key}[=:]\\s*([0-9.]+)"
            match = re.search(pattern, user_input)
            if match:
                extracted_params[param_key] = float(match.group(1))
    
    return extracted_params


def validate_blade_params(params: Dict[str, float]) -> tuple[bool, list]:
    """
    验证21维参数是否在合理范围内
    
    返回: (is_valid, error_messages)
    """
    errors = []
    for param_key, value in params.items():
        if param_key not in BLADE_21D_PARAMS:
            errors.append(f"未知参数：{param_key}")
            continue
        
        param_info = BLADE_21D_PARAMS[param_key]
        min_val, max_val = param_info["range"]
        
        if value < min_val or value > max_val:
            errors.append(
                f"{param_info['name']}（{param_key}）= {value} {param_info['unit']} "
                f"超出合理范围（{min_val}-{max_val}）"
            )
    
    return len(errors) == 0, errors


def format_missing_params_question(provided_params: Dict[str, float], task_type: str = "可视化") -> str:
    """
    格式化缺失参数的反问消息
    
    Args:
        provided_params: 用户已提供的参数
        task_type: 任务类型（"可视化" 或 "性能评估"）
    
    Returns:
        格式化的问题字符串
    """
    missing_params = [key for key in BLADE_21D_PARAMS_ORDER if key not in provided_params]
    
    question = f"❓ **{task_type}需要21维叶片几何参数。**\n\n"
    
    if provided_params:
        question += "✅ **已提供的参数：**\n"
        for key, value in provided_params.items():
            if key in BLADE_21D_PARAMS:
                param_info = BLADE_21D_PARAMS[key]
                question += f"  • {param_info['name']}（{key}）= {value} {param_info['unit']}\n"
        question += "\n"
    
    if missing_params:
        question += f"⚠️ **缺失的参数（共{len(missing_params)}个）：**\n"
        
        # 分组显示：叶根、叶中、叶顶
        root_missing = [k for k in missing_params if k.startswith("root_")]
        mid_missing = [k for k in missing_params if k.startswith("mid_")]
        tip_missing = [k for k in missing_params if k.startswith("tip_")]
        
        if root_missing:
            question += "\n**叶根参数（root）：**\n"
            for key in root_missing:
                param_info = BLADE_21D_PARAMS[key]
                question += (
                    f"  • {param_info['name']}（{key}）"
                    f"- 范围：{param_info['range']}，单位：{param_info['unit']}\n"
                )
        
        if mid_missing:
            question += "\n**叶中参数（mid）：**\n"
            for key in mid_missing:
                param_info = BLADE_21D_PARAMS[key]
                question += (
                    f"  • {param_info['name']}（{key}）"
                    f"- 范围：{param_info['range']}，单位：{param_info['unit']}\n"
                )
        
        if tip_missing:
            question += "\n**叶顶参数（tip）：**\n"
            for key in tip_missing:
                param_info = BLADE_21D_PARAMS[key]
                question += (
                    f"  • {param_info['name']}（{key}）"
                    f"- 范围：{param_info['range']}，单位：{param_info['unit']}\n"
                )
        
        question += "\n💡 **建议：**\n"
        question += "1. 如果您没有具体的参数，可以说「我需要设计一个叶片」，系统会根据性能目标自动生成\n"
        question += "2. 如果您有部分参数，请逐个提供，例如：「叶根弦长为0.6m，叶根进口角为57度」\n"
    
    return question


def create_llm(temperature: float = 0.5) -> ChatOpenAI:
    """创建LLM实例（与单智能体一致）"""
    # 使用标准环境变量（与单智能体一致）
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    
    # 调试输出
    print(f"[LLM] 创建LLM实例...")
    print(f"[LLM] API Key: {'已设置' if api_key else '未设置'}")
    print(f"[LLM] Base URL: {base_url}")
    print(f"[LLM] Model: {model}")
    
    if not api_key:
        raise ValueError("API Key未设置！请先配置API设置")
    
    # 构建参数
    kwargs = {
        'model': model,
        'temperature': temperature
    }
    
    # 只在有配置时传入
    if api_key:
        kwargs['api_key'] = api_key
    if base_url:
        kwargs['base_url'] = base_url
    
    return ChatOpenAI(**kwargs)


def extract_json_from_text(text: str) -> Dict:
    """从LLM返回的文本中提取JSON"""
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    return {}


def extract_selected_schemes(user_input: str, available_count: int, llm: ChatOpenAI = None) -> List[int]:
    """
    从用户输入中提取要操作的方案编号（简单版本，不区分任务类型）
    
    Args:
        user_input: 用户输入文本
        available_count: 可用的方案总数
        llm: LLM实例（可选，用于复杂情况的识别）
    
    Returns:
        方案编号列表（1-indexed），如 [1, 3] 表示第1个和第3个方案
        如果返回空列表，表示操作所有方案或默认第一个
    
    示例：
        "评估第2个方案" -> [2]
        "评估方案1和3" -> [1, 3]
        "对第一个和第三个方案进行评估" -> [1, 3]
        "评估所有方案" -> [] (返回空列表表示所有)
        "评估" -> [] (未指定，由调用者决定默认行为)
    """
    import re
    
    selected = []
    
    # 1. 首先尝试正则匹配常见模式
    # 匹配 "第X个" 模式
    pattern1 = r'第\s*([一二三四五六七八九十\d]+)\s*个'
    matches1 = re.findall(pattern1, user_input)
    
    # 匹配 "方案X" 模式
    pattern2 = r'方案\s*([一二三四五六七八九十\d]+)'
    matches2 = re.findall(pattern2, user_input)
    
    # 匹配 "设计X" 模式
    pattern3 = r'设计\s*([一二三四五六七八九十\d]+)'
    matches3 = re.findall(pattern3, user_input)
    
    # 中文数字映射
    cn_num_map = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '第一': 1, '第二': 2, '第三': 3, '第四': 4, '第五': 5,
    }
    
    def parse_num(s: str) -> int:
        """解析数字字符串（支持中文和阿拉伯数字）"""
        s = s.strip()
        if s.isdigit():
            return int(s)
        elif s in cn_num_map:
            return cn_num_map[s]
        else:
            # 尝试解析复杂中文数字如"十一"
            if s.startswith('十'):
                if len(s) == 1:
                    return 10
                rest = s[1:]
                if rest in cn_num_map:
                    return 10 + cn_num_map[rest]
            return 0
    
    # 合并所有匹配结果
    all_matches = matches1 + matches2 + matches3
    for m in all_matches:
        num = parse_num(m)
        if 1 <= num <= available_count and num not in selected:
            selected.append(num)
    
    # 2. 检查是否要求"所有"或"全部"
    if re.search(r'(所有|全部|每个|各个)', user_input):
        return list(range(1, available_count + 1))
    
    # 3. 如果正则未匹配到且有LLM，尝试用LLM识别
    if not selected and llm and available_count > 1:
        try:
            extraction_prompt = f"""从用户输入中识别要操作的方案编号。

用户输入：{user_input}

当前共有 {available_count} 个可用方案（方案1到方案{available_count}）。

请返回JSON格式：
{{
    "scheme_indices": [数字列表，如 [1, 3] 表示第1个和第3个方案],
    "select_all": true/false  // 是否选择所有方案
}}

如果用户没有明确指定方案编号，返回空列表。
如果用户说"所有"、"全部"、"每个"，设置 select_all 为 true。

示例：
- "评估第2个" -> {{"scheme_indices": [2], "select_all": false}}
- "评估方案1和3" -> {{"scheme_indices": [1, 3], "select_all": false}}
- "评估所有" -> {{"scheme_indices": [], "select_all": true}}
- "评估" -> {{"scheme_indices": [], "select_all": false}}

**重要**：返回有效JSON！"""

            response = llm.invoke([HumanMessage(content=extraction_prompt)])
            result = extract_json_from_text(response.content)
            
            if result:
                if result.get("select_all"):
                    return list(range(1, available_count + 1))
                indices = result.get("scheme_indices", [])
                # 验证并过滤有效的编号
                selected = [i for i in indices if isinstance(i, int) and 1 <= i <= available_count]
        except Exception as e:
            print(f"[SchemeExtraction] LLM提取失败: {e}")
    
    # 4. 排序并返回
    selected.sort()
    return selected


def extract_task_specific_schemes(
    user_input: str, 
    task_type: str, 
    available_count: int, 
    llm: ChatOpenAI
) -> List[int]:
    """
    【核心函数】从用户输入中精确提取与特定任务相关的方案编号
    
    这个函数是多任务序列精确识别的关键！
    它使用LLM分析复杂用户输入，区分不同任务各自对应哪些方案。
    
    Args:
        user_input: 完整的用户输入文本
        task_type: 当前任务类型（"evaluate", "optimize", "visualize"）
        available_count: 可用的方案总数
        llm: LLM实例
    
    Returns:
        方案编号列表（1-indexed），仅包含与指定任务相关的方案
        如果返回空列表，表示用户未为该任务指定方案
    
    示例：
        输入: "对第三个方案进行评估，然后对第一个和第五个方案进行优化"
        - task_type="evaluate" -> [3]
        - task_type="optimize" -> [1, 5]
        
        输入: "评估方案2，然后可视化方案1和3，最后优化所有方案"
        - task_type="evaluate" -> [2]
        - task_type="visualize" -> [1, 3]
        - task_type="optimize" -> [1, 2, 3, ...] (所有方案)
    """
    
    # 任务类型对应的中文关键词
    task_keywords = {
        "evaluate": ["评估", "评价", "测试", "性能分析", "分析性能", "测评"],
        "optimize": ["优化", "改进", "提升", "最大化", "最小化", "迭代"],
        "visualize": ["可视化", "展示", "显示", "查看", "画图", "绘制", "看看"],
    }
    
    task_name_cn = {
        "evaluate": "性能评估",
        "optimize": "优化",
        "visualize": "可视化",
    }
    
    keywords = task_keywords.get(task_type, [])
    task_cn = task_name_cn.get(task_type, task_type)
    
    print(f"[TaskSpecificExtraction] 正在提取'{task_cn}'任务相关的方案...")
    
    try:
        extraction_prompt = f"""你是一个精确的语义分析专家。请从用户输入中**仅提取**与"{task_cn}"任务相关的方案编号。

【用户输入】
{user_input}

【当前任务类型】
{task_cn}（关键词: {', '.join(keywords)}）

【可用方案】
共有 {available_count} 个设计方案（方案1到方案{available_count}）

【分析要求】
1. 仔细分析用户输入，理解其中包含的任务序列
2. 只关注与"{task_cn}"直接相关的部分
3. 提取该任务明确提到的方案编号
4. 如果"{task_cn}"没有明确指定方案，返回空列表
5. 区分不同任务的方案指向，不要混淆

【关键示例】
- 输入: "对第三个方案进行评估，然后对第一个和第五个方案进行优化"
  - 如果提取"评估"任务 → [3]（只有第三个方案与评估相关）
  - 如果提取"优化"任务 → [1, 5]（只有第一和第五个方案与优化相关）

- 输入: "评估方案2，然后优化方案1和3"
  - 如果提取"评估"任务 → [2]
  - 如果提取"优化"任务 → [1, 3]

- 输入: "评估所有方案，然后优化第一个"
  - 如果提取"评估"任务 → 所有方案
  - 如果提取"优化"任务 → [1]

- 输入: "优化效率，迭代20轮"（没有指定方案）
  - 如果提取"优化"任务 → []（未指定方案）

请返回JSON格式：
{{
    "task_type": "{task_type}",
    "analysis": "简要分析用户输入中与{task_cn}相关的内容",
    "scheme_indices": [方案编号列表，1-indexed],
    "select_all": true/false,
    "confidence": "high/medium/low"
}}

**注意**：
- scheme_indices 只包含与 "{task_cn}" 直接关联的方案
- 如果用户对该任务说"所有"、"全部"，设置 select_all 为 true
- 如果用户没有为该任务指定方案，返回空列表
- 必须返回有效JSON！"""

        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        result = extract_json_from_text(response.content)
        
        if result:
            analysis = result.get("analysis", "")
            confidence = result.get("confidence", "unknown")
            print(f"[TaskSpecificExtraction] LLM分析: {analysis} (置信度: {confidence})")
            
            if result.get("select_all"):
                print(f"[TaskSpecificExtraction] '{task_cn}'任务: 选择所有方案 (1-{available_count})")
                return list(range(1, available_count + 1))
            
            indices = result.get("scheme_indices", [])
            # 验证并过滤有效的编号
            valid_indices = [i for i in indices if isinstance(i, int) and 1 <= i <= available_count]
            valid_indices.sort()
            
            print(f"[TaskSpecificExtraction] '{task_cn}'任务: 方案 {valid_indices}")
            return valid_indices
            
    except Exception as e:
        print(f"[TaskSpecificExtraction] LLM提取失败: {e}")
    
    # 失败时回退到简单提取
    print(f"[TaskSpecificExtraction] 回退到简单模式...")
    return extract_selected_schemes(user_input, available_count, llm)


def get_available_scheme_count(design_params: Dict) -> int:
    """
    获取可用的设计方案数量
    
    Args:
        design_params: 设计结果字典
    
    Returns:
        可用方案数量
    """
    if not design_params:
        return 0
    
    count = 0
    for key in design_params.keys():
        if key.startswith("第") and "个设计结果" in key:
            count += 1
    return count


# ==================== 新增：任务解析器 ====================

def parse_task_sequence(user_input: str, llm: ChatOpenAI) -> Dict[str, Any]:
    """
    任务解析器：解析用户输入中的任务序列
    
    职责（精简版）：
    - 只负责识别任务类型和任务序列
    - 不提取具体任务参数（参数提取由各专家节点负责）
    
    返回：
    {
        "mode": "single" | "sequential",
        "tasks": ["design", "evaluate", "visualize", "optimize", "chat"],
        "has_dependencies": true/false
    }
    
    注意：优化参数（目标、目标值、迭代次数）由 Optimization Expert 提取
    """
    parse_prompt = f"""分析用户输入，判断用户是在请求执行具体任务，还是在进行知识询问/闲聊。

用户输入：{user_input}

**首先判断意图类型（这是最重要的一步）：**

1. **知识询问/闲聊（tasks=["chat"]）**：用户在询问知识、原理、步骤、概念解释，或者进行无关闲聊。
   识别特征：
   - 询问"什么是"、"什么叫"、"怎么理解"
   - 询问"有哪些步骤"、"分为哪几个阶段"、"包括什么"
   - 询问"为什么"、"原理是什么"、"有什么区别"
   - 一般性的问句，没有明确的执行动作要求
   - 天气、笑话、计算等无关话题
   
   示例：
   - "压气机在设计阶段都分为哪些步骤" → 知识询问
   - "叶片设计的原理是什么" → 知识询问
   - "什么是叶型参数" → 知识询问
   - "今天天气怎么样" → 闲聊

2. **执行任务请求**：用户明确要求系统执行某个操作。
   识别特征：
   - 明确的执行动词："帮我设计"、"生成一个"、"创建叶片"
   - 明确的评估请求："评估一下"、"计算性能"、"测一下"
   - 明确的可视化请求："画出来"、"可视化"、"展示3D"
   - 明确的优化请求："优化一下"、"提升效率"、"改进方案"

**如果是知识询问/闲聊，直接返回：**
{{"mode": "single", "tasks": ["chat"], "has_dependencies": false}}

**如果是执行任务请求，可能的任务类型：**
- design: 叶片设计（关键词：帮我设计、生成叶片、创建设计）
- evaluate: 性能评估（关键词：评估一下、计算性能、测试性能）
- visualize: 3D可视化（关键词：画出来、可视化、展示3D）
- optimize: 优化设计（关键词：优化、改进、提升、提高）

判断规则：
1. 如果只有一个任务 → mode="single"
2. 如果有多个任务（用"然后"、"接着"、"再"、"之后"连接）→ mode="sequential"
3. 任务按提及顺序排列

返回JSON格式：
{{
    "mode": "single或sequential",
    "tasks": ["任务类型列表"],
    "has_dependencies": true/false
}}

示例：
- "压气机在设计阶段都分为哪些步骤" → {{"mode": "single", "tasks": ["chat"], "has_dependencies": false}}
- "帮我设计一个叶片" → {{"mode": "single", "tasks": ["design"], "has_dependencies": false}}
- "优化效率到89%" → {{"mode": "single", "tasks": ["optimize"], "has_dependencies": false}}
- "设计叶片，然后评估，再优化" → {{"mode": "sequential", "tasks": ["design", "evaluate", "optimize"], "has_dependencies": true}}
- "帮我评估性能" → {{"mode": "single", "tasks": ["evaluate"], "has_dependencies": false}}

**重要**：
1. 必须返回有效的JSON格式！
2. **区分知识询问和任务请求是第一优先级！** 不要仅凭关键词判断，要理解用户的真实意图。
3. **不需要提取具体参数**（如优化目标、迭代次数等），这些由各专家节点负责提取。"""

    llm_result = None
    try:
        response = llm.invoke([HumanMessage(content=parse_prompt)])
        result = extract_json_from_text(response.content)
        
        if result and "tasks" in result:
            print(f"[TaskParser] LLM解析结果: mode={result.get('mode')}, tasks={result.get('tasks')}")
            llm_result = result
    except Exception as e:
        print(f"[TaskParser] LLM解析失败: {e}")
    
    # 无论 LLM 是否成功，都用关键词匹配补充/验证
    tasks = []
    user_lower = user_input.lower()
    
    # ==================== 优先使用 LLM 结果 ====================
    if llm_result:
        final_tasks = llm_result.get("tasks", [])
        final_mode = llm_result.get("mode", "single")
        
        return {
            "mode": final_mode,
            "tasks": final_tasks,
            "has_dependencies": len(final_tasks) > 1
        }
    
    # ==================== Fallback：关键词匹配（仅当LLM失败时）====================
    print(f"[parse_task_sequence] LLM解析失败，使用关键词匹配...")
    
    # 首先检测是否是知识询问/闲聊（高优先级）
    knowledge_patterns = [
        "什么是", "什么叫", "怎么理解", "如何理解",
        "有哪些步骤", "分为哪", "包括什么", "包含什么",
        "为什么", "原理是什么", "有什么区别", "有什么不同",
        "是如何", "怎样", "怎么样", "如何计算", "怎么计算",
        "天气", "笑话", "等于", "多少",
        "请问", "请教", "想问", "想了解",
        "介绍一下", "解释一下", "说明一下", "讲一下"
    ]
    
    # 任务执行关键词（只有这些才是真正的任务请求）
    action_patterns = [
        "帮我设计", "帮我生成", "帮我创建", "生成一个", "创建一个", "设计一个",
        "帮我评估", "帮我计算", "评估一下", "计算一下", "测试一下",
        "帮我画", "画出来", "可视化", "展示3d", "画个",
        "帮我优化", "优化一下", "提升", "提高", "改进", "改善"
    ]
    
    is_knowledge_query = any(kw in user_lower for kw in knowledge_patterns)
    is_action_request = any(kw in user_lower for kw in action_patterns)
    
    # 如果明显是知识询问且没有明确的执行动作，归类为 chat
    if is_knowledge_query and not is_action_request:
        print(f"[parse_task_sequence] 检测到知识询问/闲聊，返回chat任务")
        return {
            "mode": "single",
            "tasks": ["chat"],
            "has_dependencies": False
        }
    
    has_sequence = any(kw in user_lower for kw in ["然后", "接着", "再", "之后", "完成后", "再帮我"])
    
    if any(kw in user_lower for kw in ["帮我设计", "生成设计", "创建", "设计一个", "生成一个"]):
        tasks.append("design")
    if any(kw in user_lower for kw in ["评估", "计算性能", "测试性能", "评估一下"]):
        tasks.append("evaluate")
    if any(kw in user_lower for kw in ["画", "可视化", "3d", "展示"]):
        tasks.append("visualize")
    if any(kw in user_lower for kw in ["优化", "改进", "提升", "提高", "改善"]):
        tasks.append("optimize")
    
    if not tasks:
        tasks = ["chat"]  # 默认为闲聊而不是 unknown
    
    # 注意：不再提取 optimization_config，由 Optimization Expert 负责提取
    
    return {
        "mode": "sequential" if has_sequence and len(tasks) > 1 else "single",
        "tasks": tasks,
        "has_dependencies": len(tasks) > 1
    }


def analyze_intent_type(state: MultiAgentState, user_input: str, llm: ChatOpenAI) -> Dict[str, Any]:
    """
    意图类型分析器：判断用户是新任务、继续任务还是使用已有结果
    
    返回：
    {
        "intent_type": "new" | "continue" | "use_existing" | "sequence",
        "use_existing_result": true/false/null,
        "reason": "判断理由"
    }
    """
    # 获取上下文信息
    last_task = state.get("last_task")
    has_design = state.get("design_params") is not None or state.get("task_results", {}).get("design") is not None
    has_eval = state.get("evaluation_results") is not None or state.get("task_results", {}).get("evaluate") is not None
    pending_agent = state.get("pending_agent")
    
    # 计算距离上次任务的时间（如果有）
    last_time = state.get("last_task_time")
    time_since_last = None
    if last_time:
        time_since_last = time.time() - last_time
    
    analyze_prompt = f"""分析用户的真实意图。

当前状态：
- 上次完成的任务：{last_task or "无"}
- 是否有设计结果：{has_design}
- 是否有评估结果：{has_eval}
- 当前是否在等待参数：{pending_agent or "否"}
- 距离上次任务时间：{f'{time_since_last:.0f}秒' if time_since_last else '未知'}

用户输入：{user_input}

请判断用户意图类型：

1. **new（全新任务）** - 用户想开始一个全新任务，需要清空之前状态
   信号词："新的"、"重新"、"另一个"、"不同的"、"换个"
   
2. **continue（继续当前任务）** - 用户在补充参数
   特征：输入主要是数值、参数名+值
   
3. **use_existing（使用已有结果）** - 用户想对之前的结果进行后续操作
   信号词："刚才的"、"那个"、"这个设计"、"它"、"直接"
   场景：刚完成设计，用户说"评估一下"、"画个图"
   
4. **sequence（多任务序列）** - 用户一次描述多个任务
   信号词："然后"、"接着"、"再"、"之后"

返回JSON：
{{
    "intent_type": "new/continue/use_existing/sequence",
    "use_existing_result": true/false/null,
    "reason": "判断理由"
}}

判断关键点：
- 如果用户刚完成设计，又说要"评估"但没提参数 → 很可能想用已有结果
- 如果有设计结果，用户说"优化"、"改进"、"提升" → 很可能想用已有设计进行优化
- 如果用户明确说"新的"、"另一个" → 新任务
- 如果输入主要是数字和参数 → 继续补充参数

**重要**：必须返回有效的JSON格式！"""

    try:
        response = llm.invoke([HumanMessage(content=analyze_prompt)])
        result = extract_json_from_text(response.content)
        
        if result and "intent_type" in result:
            print(f"[IntentAnalyzer] 意图类型: {result.get('intent_type')}, 原因: {result.get('reason', '')}")
            return result
    except Exception as e:
        print(f"[IntentAnalyzer] LLM分析失败: {e}")
    
    # Fallback：基于规则判断
    user_lower = user_input.lower()
    
    # 检查是否是多任务序列
    if any(kw in user_lower for kw in ["然后", "接着", "再", "之后"]):
        return {"intent_type": "sequence", "use_existing_result": None, "reason": "检测到顺序关键词"}
    
    # 检查是否明确新任务
    if any(kw in user_lower for kw in ["新的", "重新", "另一个", "换个"]):
        return {"intent_type": "new", "use_existing_result": False, "reason": "检测到新任务关键词"}
    
    # 检查是否使用已有结果
    if any(kw in user_lower for kw in ["刚才", "那个", "这个设计", "它", "直接"]):
        return {"intent_type": "use_existing", "use_existing_result": True, "reason": "检测到引用关键词"}
    
    # 如果有pending_agent，可能是继续补充参数
    if pending_agent:
        # 检查是否主要是数字
        has_numbers = bool(re.search(r'\d+\.?\d*', user_input))
        if has_numbers:
            return {"intent_type": "continue", "use_existing_result": None, "reason": "正在补充参数"}
    
    # 如果刚完成设计且用户提到评估/可视化
    if has_design and last_task == "design":
        if any(kw in user_lower for kw in ["评估", "性能", "画", "图", "可视化"]):
            return {"intent_type": "use_existing", "use_existing_result": None, "reason": "可能想用已有设计"}
    
    # 如果有设计结果且用户提到优化（类似评估/可视化的处理）
    if has_design and any(kw in user_lower for kw in ["优化", "改进", "提升", "提高", "改善"]):
        return {"intent_type": "use_existing", "use_existing_result": None, "reason": "可能想用已有设计进行优化"}
    
    return {"intent_type": "new", "use_existing_result": False, "reason": "默认为新任务"}


def extract_21d_from_design_result(design_result: Dict) -> Dict[str, float]:
    """
    从设计结果中提取21维参数
    
    Args:
        design_result: 设计工具返回的结果
        
    Returns:
        21维参数字典
    """
    if not design_result:
        return {}
    
    first_design = design_result.get("第1个设计结果", {})
    if not first_design:
        return {}
    
    # 键名映射：设计结果键 → 标准键
    key_mapping = {
        "root_Angle_in（叶根进口金属角[°]）": "root_Angle_in",
        "root_Angle_out（叶根出口金属角[°]）": "root_Angle_out",
        "root_Chord（叶根弦长[m]）": "root_Chord",
        "root_THmax_CH（叶根最大相对厚度[%]）": "root_THmax_CH",
        "root_THmaxP（叶根最大相对厚度位置[-]）": "root_THmaxP",
        "root_SWA（叶根最大相对厚度位置[m]）": "root_SWA",
        "root_BOWA（叶根掠[m]）": "root_BOWA",
        "mid_Angle_in（叶中进口金属角[°]）": "mid_Angle_in",
        "mid_Angle_out（叶中出口金属角[°]）": "mid_Angle_out",
        "mid_Chord（叶中弦长[m]）": "mid_Chord",
        "mid_THmax_CH（叶中最大相对厚度[%]）": "mid_THmax_CH",
        "mid_THmaxP（叶中最大相对厚度位置[-]）": "mid_THmaxP",
        "mid_SWA（叶中最大相对厚度位置[m]）": "mid_SWA",
        "mid_BOWA（叶中掠[m]）": "mid_BOWA",
        "tip_Angle_in（叶顶进口金属角[°]）": "tip_Angle_in",
        "tip_Angle_out（叶顶出口金属角[°]）": "tip_Angle_out",
        "tip_Chord（叶顶弦长[m]）": "tip_Chord",
        "tip_THmax_CH（叶顶最大相对厚度[%]）": "tip_THmax_CH",
        "tip_THmaxP（叶顶最大相对厚度位置[-]）": "tip_THmaxP",
        "tip_SWA（叶顶最大相对厚度位置[m]）": "tip_SWA",
        "tip_BOWA（叶顶掠[m]）": "tip_BOWA"
    }
    
    result = {}
    for long_key, short_key in key_mapping.items():
        if long_key in first_design:
            result[short_key] = first_design[long_key]
    
    return result


def store_task_result(state: MultiAgentState, task: str) -> None:
    """
    存储任务结果到state
    
    Args:
        state: 当前状态
        task: 任务类型 (design/evaluate/visualize/optimize)
    """
    if "task_results" not in state or state["task_results"] is None:
        state["task_results"] = {}
    
    if task == "design" and state.get("design_params"):
        state["task_results"]["design"] = state["design_params"]
    elif task == "evaluate" and state.get("evaluation_results"):
        state["task_results"]["evaluate"] = state["evaluation_results"]
    elif task == "visualize" and state.get("visualization_path"):
        state["task_results"]["visualize"] = state["visualization_path"]
    elif task == "optimize" and state.get("optimization_result"):
        state["task_results"]["optimize"] = state["optimization_result"]
    
    state["last_task"] = task
    state["last_task_time"] = time.time()
    
    print(f"[ResultStore] 存储 {task} 结果, last_task={task}")


def _handle_next_task(state: MultiAgentState, current_task: str, response_msg: str, 
                      image_path: str = None, json_path: str = None) -> MultiAgentState:
    """
    处理多任务序列中的下一个任务
    
    Args:
        state: 当前状态
        current_task: 刚完成的任务类型
        response_msg: 当前任务的响应消息
        image_path: 当前任务产生的图片路径（可选）
        json_path: 当前任务产生的3D数据路径（可选）
        
    Returns:
        更新后的状态
    """
    task_queue = state.get("task_queue", [])
    current_index = state.get("current_task_index", 0)
    total_tasks = len(task_queue)
    
    # ==================== 新增：累积中间结果 ====================
    # 将当前任务的结果累积到 accumulated_results 中
    if "accumulated_results" not in state or state.get("accumulated_results") is None:
        state["accumulated_results"] = []
    
    # 添加当前任务的结果
    task_result_item = {
        "task": current_task,
        "result": response_msg
    }
    if image_path:
        task_result_item["image_path"] = image_path
    if json_path:
        task_result_item["json_path"] = json_path
    state["accumulated_results"].append(task_result_item)
    
    # ==================== 新增：立即发送当前任务结果到前端 ====================
    _send_task_result(
        task=current_task,
        result=response_msg,
        task_index=current_index + 1,
        total_tasks=total_tasks,
        image_path=image_path,
        json_path=json_path
    )
    
    if current_index + 1 < len(task_queue):
        # 还有后续任务
        state["current_task_index"] = current_index + 1
        next_task = task_queue[current_index + 1]
        
        task_names = {"design": "设计", "evaluate": "评估", "visualize": "可视化", "optimize": "优化", "chat": "知识问答"}
        
        # ✅ 关键：将当前任务结果展示给用户，然后提示进入下一任务
        transition_msg = f"\n\n🔄 **任务 {current_index + 1}/{len(task_queue)} 完成**，正在进入下一个任务：{task_names.get(next_task, next_task)}...\n"
        transition_msg += "─" * 50
        
        state["use_existing_result"] = True  # 后续任务默认使用已有结果
        state["partial_blade_params"] = {}  # 清除累积参数
        state["next_action"] = "to_scheduler"
        
        log_collaboration(state, state.get("current_agent", "unknown"), "scheduler",
                        f"{current_task}完成，进入下一任务", {"next_task": next_task})
        
        # 更新响应消息（包含当前结果 + 过渡提示）
        state["final_result"] = response_msg + transition_msg
        
        return state
    else:
        # 所有任务完成 - 由于SSE已经实时发送了每个任务的结果，这里只发送简单的完成提示
        accumulated = state.get("accumulated_results", [])
        
        print(f"[_handle_next_task] 所有任务完成，汇总 {len(accumulated)} 个结果")
        
        if len(accumulated) > 1:
            # 多任务完成 - 只发送简单的完成提示（避免与SSE实时结果重复）
            task_names_list = []
            for item in accumulated:
                task_map = {"design": "设计", "evaluate": "评估", "visualize": "可视化", "optimize": "优化", "chat": "知识问答"}
                task_names_list.append(task_map.get(item["task"], item["task"]))
            
            summary = f"✅ **多任务序列执行完成**\n\n"
            summary += f"已完成的任务: {' → '.join(task_names_list)} （共{len(accumulated)}个）\n\n"
            summary += "💡 各任务的详细结果已在上方实时展示，您可以继续提问或开始新的任务。"
            
            state["final_result"] = summary
            state["messages"].append(AIMessage(content=summary))
            print(f"[_handle_next_task] 多任务完成提示已生成")
        else:
            # 只有一个任务，直接显示
            state["final_result"] = response_msg
        
        state["next_action"] = EdgeConditions.TO_END
        return state


# ==================== 协调者节点（增强版）====================

def coordinator_node(state: MultiAgentState) -> MultiAgentState:
    """
    协调者Agent - 增强版
    
    新增能力：
    1. 多任务序列解析
    2. 智能意图判断（使用已有结果 vs 新任务）
    3. 任务结果传递
    
    职责：
    1. 解析任务序列（单任务/多任务）
    2. 分析用户意图类型（新任务/继续/使用已有结果）
    3. 分配任务给专家或调度器
    """
    
    print(f"\n{'='*60}")
    print(f"[Coordinator] 开始分析用户需求...")
    print(f"{'='*60}")
    
    user_input = state.get("user_input", "")
    
    # ==================== 初始化新增字段 ====================
    if state.get("task_results") is None:
        state["task_results"] = {}
    if state.get("task_mode") is None:
        state["task_mode"] = "single"
    if state.get("task_queue") is None:
        state["task_queue"] = []
    if state.get("current_task_index") is None:
        state["current_task_index"] = 0
    if state.get("awaiting_confirmation") is None:
        state["awaiting_confirmation"] = False
    
    # ==================== 处理确认使用已有结果的回复 ====================
    if state.get("awaiting_confirmation"):
        print(f"[Coordinator] 检测到正在等待用户确认是否使用已有结果")
        
        user_lower = user_input.lower().strip()
        
        # 判断用户选择
        if user_lower in ["1", "是", "对", "好的", "直接用", "使用已有", "用刚才的"]:
            # 用户选择使用已有结果
            state["use_existing_result"] = True
            state["awaiting_confirmation"] = False
            print(f"[Coordinator] 用户确认使用已有结果")
            
            # 路由到之前记录的目标任务
            pending_task = state.get("pending_agent")
            if pending_task == AgentNames.EVALUATION_EXPERT:
                state["target_agent"] = AgentNames.EVALUATION_EXPERT
                state["current_agent"] = AgentNames.EVALUATION_EXPERT
                return state
            elif pending_task == AgentNames.VISUALIZATION_EXPERT:
                state["target_agent"] = AgentNames.VISUALIZATION_EXPERT
                state["current_agent"] = AgentNames.VISUALIZATION_EXPERT
                return state
        
        elif user_lower in ["2", "否", "不", "新的", "重新输入", "我自己提供"]:
            # 用户选择新输入
            state["use_existing_result"] = False
            state["awaiting_confirmation"] = False
            state["partial_blade_params"] = {}  # 清空累积参数
            print(f"[Coordinator] 用户选择新输入参数")
            
            pending_task = state.get("pending_agent")
            if pending_task == AgentNames.EVALUATION_EXPERT:
                state["target_agent"] = AgentNames.EVALUATION_EXPERT
                state["current_agent"] = AgentNames.EVALUATION_EXPERT
                return state
            elif pending_task == AgentNames.VISUALIZATION_EXPERT:
                state["target_agent"] = AgentNames.VISUALIZATION_EXPERT
                state["current_agent"] = AgentNames.VISUALIZATION_EXPERT
                return state
        
        else:
            # 用户回复不明确，再次询问
            question = "抱歉，我没有理解您的选择。请回复：\n1️⃣ - 使用刚才的设计结果\n2️⃣ - 提供新的参数"
            state["user_question"] = question
            state["waiting_for_user"] = True
            state["messages"].append(AIMessage(content=question))
            return state
    
    # ==================== 检查是否是恢复状态（参数补充）====================
    pending_agent = state.get("pending_agent")
    if pending_agent and not state.get("awaiting_confirmation"):
        print(f"[Coordinator] 检测到恢复状态（之前等待的Agent: {pending_agent}）")
        
        try:
            llm = create_llm(temperature=0.1)
            
            task_switch_prompt = f"""判断用户输入是否想切换任务。

用户之前的任务：{pending_agent}（正在等待用户提供参数）

用户当前输入：{user_input}

请判断用户是否想：
1. **切换任务** - 明确表达想做其他事情（如"设计叶片"、"评估性能"、"算了不画了"、"我想要..."、"帮我..."等）
2. **继续当前任务** - 只是提供参数或信息（如"叶根进口角58度"、"流量15"、"效率0.85"等）

返回JSON格式：
{{
    "is_task_switch": true/false,
    "reason": "判断理由"
}}

**判断标准**：
- 如果包含明确的动作意图词（设计、评估、画图、可视化、算了、不要了、换个、改成）→ 切换任务
- 如果只是数字+单位、参数名+值 → 继续当前任务
- 如果是疑问句或闲聊 → 切换任务

**重要**：必须返回有效的JSON格式！"""
            
            response = llm.invoke([HumanMessage(content=task_switch_prompt)])
            result = extract_json_from_text(response.content)
            
            is_task_switch = False
            if result and "is_task_switch" in result:
                is_task_switch = result["is_task_switch"]
                print(f"[Coordinator] 任务切换判断: {is_task_switch} - {result.get('reason', '')}")
            else:
                # Fallback：关键词匹配
                print(f"[Coordinator] LLM判断失败，使用关键词匹配...")
                user_input_lower = user_input.lower()
                switch_keywords = ["设计", "评估", "可视化", "画图", "算了", "不要", "换个", "改成", "帮我", "我想", "你能"]
                param_keywords = ["叶根", "叶中", "叶顶", "进口角", "出口角", "弦长", "厚度", "流量", "效率", "压比"]
                has_numbers = bool(re.search(r'\d+\.?\d*', user_input))
                
                if any(kw in user_input_lower for kw in switch_keywords) and not any(kw in user_input_lower for kw in param_keywords):
                    is_task_switch = True
                elif has_numbers and any(kw in user_input_lower for kw in param_keywords):
                    is_task_switch = False
                else:
                    is_task_switch = len(user_input) > 15 or not has_numbers
            
            if is_task_switch:
                print(f"[Coordinator] 用户想切换任务，清除恢复状态和多任务遗留状态")
                # 清除等待状态
                state["pending_agent"] = None
                state["partial_blade_params"] = {}
                state["performance_targets"] = {}
                state["waiting_for_user"] = False
                state["user_question"] = None
                
                # ✅ 清除多任务遗留状态（修复路由错误问题）
                state["next_action"] = None
                state["task_queue"] = []
                state["task_mode"] = "single"
                state["current_task_index"] = 0
                state["accumulated_results"] = []
                state["optimization_config"] = None
                state["awaiting_confirmation"] = False
                
                # ✅ 【关键】任务切换时，清除原始用户输入
                # 新任务的 prompt 将在后续的正常意图识别流程中被设置为 original_user_input
                state["original_user_input"] = None
                print(f"[Coordinator] 清除原始用户输入，准备使用新任务的 prompt")
                
                # 继续后续的正常意图识别流程
            else:
                print(f"[Coordinator] 用户继续当前任务，路由回 {pending_agent}")
                state["target_agent"] = pending_agent
                state["current_agent"] = pending_agent
                state["pending_agent"] = None
                state["intent_type"] = "continue"
                
                if pending_agent == AgentNames.DESIGN_EXPERT:
                    state["next_action"] = EdgeConditions.TO_DESIGN
                elif pending_agent == AgentNames.EVALUATION_EXPERT:
                    state["next_action"] = EdgeConditions.TO_EVALUATION
                elif pending_agent == AgentNames.VISUALIZATION_EXPERT:
                    state["next_action"] = EdgeConditions.TO_VISUALIZATION
                
                log_collaboration(state, AgentNames.COORDINATOR, pending_agent, 
                                "恢复到等待的Agent", f"用户继续提供参数")
                return state
        
        except Exception as e:
            print(f"[Coordinator] 任务切换判断失败: {e}，默认继续当前任务")
            state["target_agent"] = pending_agent
            state["current_agent"] = pending_agent
            state["pending_agent"] = None
            
            if pending_agent == AgentNames.DESIGN_EXPERT:
                state["next_action"] = EdgeConditions.TO_DESIGN
            elif pending_agent == AgentNames.EVALUATION_EXPERT:
                state["next_action"] = EdgeConditions.TO_EVALUATION
            elif pending_agent == AgentNames.VISUALIZATION_EXPERT:
                state["next_action"] = EdgeConditions.TO_VISUALIZATION
            
            return state
    
    # ==================== 正常意图识别流程（增强版）====================
    try:
        # ✅ 【关键】进入正常意图识别流程，说明这是一个新任务
        # 将当前用户输入设置为原始用户输入，用于后续任务（如优化）提取参数
        state["original_user_input"] = user_input
        print(f"[Coordinator] 设置原始用户输入: {user_input[:50]}...")
        
        llm = create_llm()
        
        # 步骤1：解析任务序列
        print(f"[Coordinator] 步骤1: 解析任务序列...")
        task_info = parse_task_sequence(user_input, llm)
        
        state["task_mode"] = task_info.get("mode", "single")
        state["task_queue"] = task_info.get("tasks", [])
        state["current_task_index"] = 0
        
        print(f"[Coordinator] 任务模式: {state['task_mode']}, 任务队列: {state['task_queue']}")
        
        # 步骤2：分析意图类型（是否使用已有结果）
        print(f"[Coordinator] 步骤2: 分析意图类型...")
        intent_info = analyze_intent_type(state, user_input, llm)
        
        state["intent_type"] = intent_info.get("intent_type", "new")
        state["use_existing_result"] = intent_info.get("use_existing_result")
        
        print(f"[Coordinator] 意图类型: {state['intent_type']}, 使用已有结果: {state['use_existing_result']}")
        
        # ==================== 处理多任务序列 ====================
        if state["task_mode"] == "sequential" and len(state["task_queue"]) > 1:
            print(f"[Coordinator] 检测到多任务序列，进入调度模式")
            
            # 注：original_user_input 已在正常意图识别流程开始时设置
            
            # 设置为顺序执行模式
            state["next_action"] = "to_scheduler"  # 新增：路由到调度器
            
            # 通知用户
            task_names = {"design": "设计", "evaluate": "评估", "visualize": "可视化"}
            task_list = [task_names.get(t, t) for t in state["task_queue"]]
            
            info_msg = f"📋 检测到多任务序列，将依次执行：\n"
            for i, task in enumerate(task_list, 1):
                info_msg += f"  {i}. {task}\n"
            info_msg += "\n开始执行第一个任务..."
            
            state["messages"].append(AIMessage(content=info_msg))
            
            log_collaboration(state, AgentNames.COORDINATOR, "scheduler", 
                            "启动多任务序列", {"tasks": state["task_queue"]})
            
            # 注意：optimization_config 由 Optimization Expert 在执行时提取，不在此处处理
            
            # 开始执行第一个任务
            first_task = state["task_queue"][0]
            
            if first_task == "design":
                state["target_agent"] = AgentNames.DESIGN_EXPERT
                state["current_agent"] = AgentNames.DESIGN_EXPERT
                state["intent"] = "design"
            elif first_task == "evaluate":
                state["target_agent"] = AgentNames.EVALUATION_EXPERT
                state["current_agent"] = AgentNames.EVALUATION_EXPERT
                state["intent"] = "evaluate"
            elif first_task == "visualize":
                state["target_agent"] = AgentNames.VISUALIZATION_EXPERT
                state["current_agent"] = AgentNames.VISUALIZATION_EXPERT
                state["intent"] = "visualize"
            elif first_task == "optimize":
                state["target_agent"] = AgentNames.OPTIMIZATION_EXPERT
                state["current_agent"] = AgentNames.OPTIMIZATION_EXPERT
                state["intent"] = "optimize"
            
            return state
        
        # ==================== 处理单任务（含智能意图判断）====================
        # 获取首个任务
        current_task = state["task_queue"][0] if state["task_queue"] else "unknown"
        state["intent"] = current_task
        
        # 检查是否应该询问用户是否使用已有结果
        has_design_result = (state.get("design_params") is not None or 
                           state.get("task_results", {}).get("design") is not None)
        
        # 条件：有设计结果 + 当前任务是评估或可视化 + 意图类型可能使用已有结果
        if (has_design_result and 
            current_task in ["evaluate", "visualize"] and 
            state["intent_type"] in ["use_existing", "new"] and 
            state["use_existing_result"] is None):
            
            print(f"[Coordinator] 检测到可能使用已有设计结果，询问用户确认")
            
            question = f"""🔍 检测到您之前完成了叶片设计。

请问您想：
1️⃣ 对刚才的设计进行{'性能评估' if current_task == 'evaluate' else '3D可视化'}（回复 1）
2️⃣ 提供新的参数进行{'评估' if current_task == 'evaluate' else '可视化'}（回复 2）"""
            
            state["waiting_for_user"] = True
            state["awaiting_confirmation"] = True
            state["user_question"] = question
            state["pending_agent"] = AgentNames.EVALUATION_EXPERT if current_task == "evaluate" else AgentNames.VISUALIZATION_EXPERT
            state["messages"].append(AIMessage(content=question))
            
            return state
        
        # ==================== 正常路由逻辑 ====================
        if current_task == "design":
            state["target_agent"] = AgentNames.DESIGN_EXPERT
            state["current_agent"] = AgentNames.DESIGN_EXPERT
            log_collaboration(state, AgentNames.COORDINATOR, AgentNames.DESIGN_EXPERT, "分派设计任务", None)
        
        elif current_task == "evaluate":
            state["target_agent"] = AgentNames.EVALUATION_EXPERT
            state["current_agent"] = AgentNames.EVALUATION_EXPERT
            log_collaboration(state, AgentNames.COORDINATOR, AgentNames.EVALUATION_EXPERT, "分派评估任务", None)
        
        elif current_task == "visualize":
            state["target_agent"] = AgentNames.VISUALIZATION_EXPERT
            state["current_agent"] = AgentNames.VISUALIZATION_EXPERT
            log_collaboration(state, AgentNames.COORDINATOR, AgentNames.VISUALIZATION_EXPERT, "分派可视化任务", None)
        
        elif current_task == "optimize":
            state["target_agent"] = AgentNames.OPTIMIZATION_EXPERT
            state["current_agent"] = AgentNames.OPTIMIZATION_EXPERT
            # 注意：optimization_config 由 Optimization Expert 在执行时提取
            log_collaboration(state, AgentNames.COORDINATOR, AgentNames.OPTIMIZATION_EXPERT, "分派优化任务", None)
        
        elif current_task == "chat":
            print(f"[Coordinator] 检测到知识询问/闲聊，分派给闲聊专家")
            state["target_agent"] = AgentNames.CHAT_EXPERT
            state["current_agent"] = AgentNames.CHAT_EXPERT
            log_collaboration(state, AgentNames.COORDINATOR, AgentNames.CHAT_EXPERT, "分派知识询问/闲聊任务", {"user_input": user_input})
        
        else:
            print(f"[Coordinator] 意图未知（{current_task}），分派给闲聊专家")
            state["target_agent"] = AgentNames.CHAT_EXPERT
            state["current_agent"] = AgentNames.CHAT_EXPERT
            log_collaboration(state, AgentNames.COORDINATOR, AgentNames.CHAT_EXPERT, "分派闲聊任务", {"user_input": user_input})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Coordinator] 错误: {str(e)}")
        
        error_msg = f"❌ 处理您的请求时出错：{str(e)}\n\n" \
                   f"可能的原因：\n" \
                   f"1. API配置错误（请检查API Key和Base URL）\n" \
                   f"2. 网络连接问题\n" \
                   f"3. LLM服务暂时不可用\n\n" \
                   f"请稍后重试或检查配置。"
        
        state["final_result"] = error_msg
        state["messages"].append(AIMessage(content=error_msg))
        state["error_message"] = f"协调者执行失败: {str(e)}"
        state["next_action"] = EdgeConditions.TO_END
    
    return state


# ==================== 设计专家节点 ====================

def design_expert_node(state: MultiAgentState) -> MultiAgentState:
    """
    设计专家Agent - 带详细反问机制（与评估和可视化专家一致）
    
    职责：
    1. 从用户输入或state中提取性能目标参数（流量、效率、压比）
    2. 如果参数不全，列出缺失参数并反问
    3. 如果参数不合理，验证并反问
    4. 参数完整且合理，调用设计工具
    """
    
    print(f"\n{'='*60}")
    print(f"[DesignExpert] 开始生成设计...")
    print(f"{'='*60}")
    
    try:
        user_input = state.get("user_input", "")
        
        # ==================== 参数提取与累积 ====================
        # 初始化 performance_targets（如果不存在）
        if state.get("performance_targets") is None:
            state["performance_targets"] = {}
        
        # 从用户输入中提取性能目标参数
        if user_input:
            print(f"[DesignExpert] 尝试从用户输入提取参数...")
            llm = create_llm()
            
            # 使用LLM提取参数（包括性能目标和配置参数）
            extraction_prompt = f"""从用户输入中提取叶片设计的参数，返回JSON格式。

用户输入：{user_input}

需要提取的参数：

**性能目标参数（必填）：**
1. flow_rate（流量）- 单位：kg/s
2. efficiency（效率）- 0-1之间的小数，如果用户给的是百分比（如90、85%），转换为小数（0.90、0.85）
3. pressure_ratio（压比）- 大于1的数值

**配置参数（可选，有默认值）：**
4. top_k（返回方案数）- 整数，默认1，如果用户说"返回3个方案"、"生成5个设计"等
5. batch_size（批量大小）- 整数，默认50，如果用户说"批量100"、"batch_size=200"等
6. timesteps（扩散步数）- 整数，默认100，如果用户说"扩散500步"、"迭代200步"、"步数50"等

返回JSON格式：
{{
    "flow_rate": 数值或null,
    "efficiency": 数值或null,
    "pressure_ratio": 数值或null,
    "top_k": 整数或null,
    "batch_size": 整数或null,
    "timesteps": 整数或null
}}

只提取明确提到的参数。如果某个参数没提到，该字段设为null。
如果用户没有提供任何参数，返回空对象 {{}}

**重要**：必须返回有效的JSON格式！"""
            
            try:
                response = llm.invoke([HumanMessage(content=extraction_prompt)])
                response_text = response.content.strip()
                print(f"[DesignExpert] LLM原始返回: {response_text[:200]}...")
                
                extracted = extract_json_from_text(response_text)
                
                if extracted:
                    # ✅ 后处理：效率值自动转换（如果>1，可能是百分比形式）
                    if "efficiency" in extracted and extracted["efficiency"] is not None and extracted["efficiency"] > 1:
                        original_value = extracted["efficiency"]
                        extracted["efficiency"] = extracted["efficiency"] / 100
                        print(f"[DesignExpert] 效率值自动转换: {original_value} → {extracted['efficiency']}")
                    
                    print(f"[DesignExpert] 本次提取到参数: {extracted}")
                    
                    # ✅ 立即验证本次提取的参数
                    valid_params = {}
                    invalid_params = []
                    
                    for param_key, param_value in extracted.items():
                        # 跳过 None 值
                        if param_value is None:
                            continue
                            
                        # 验证每个参数的范围
                        if param_key == "flow_rate":
                            if 5 <= param_value <= 30:
                                valid_params[param_key] = param_value
                            else:
                                invalid_params.append(f"流量 {param_value} kg/s 超出合理范围（5-30）")
                        elif param_key == "efficiency":
                            if 0.5 <= param_value <= 1.0:
                                valid_params[param_key] = param_value
                            else:
                                invalid_params.append(f"效率 {param_value} 超出合理范围（0.5-1.0）")
                        elif param_key == "pressure_ratio":
                            if 1.0 <= param_value <= 3.0:
                                valid_params[param_key] = param_value
                            else:
                                invalid_params.append(f"压比 {param_value} 超出合理范围（1.0-3.0）")
                        # ✅ 新增：验证配置参数
                        elif param_key == "top_k":
                            if isinstance(param_value, (int, float)) and 1 <= param_value <= 20:
                                valid_params[param_key] = int(param_value)
                            else:
                                invalid_params.append(f"返回方案数 {param_value} 超出合理范围（1-20）")
                        elif param_key == "batch_size":
                            if isinstance(param_value, (int, float)) and 10 <= param_value <= 500:
                                valid_params[param_key] = int(param_value)
                            else:
                                invalid_params.append(f"批量大小 {param_value} 超出合理范围（10-500）")
                        elif param_key == "timesteps":
                            if isinstance(param_value, (int, float)) and 10 <= param_value <= 500:
                                valid_params[param_key] = int(param_value)
                            else:
                                invalid_params.append(f"扩散步数 {param_value} 超出合理范围（10-500）")
                    
                    # 如果有不合理的参数，立即反问
                    if invalid_params:
                        print(f"[DesignExpert] 检测到不合理参数，立即反问")
                        
                        # 先累积合理的参数（如果有）
                        if valid_params:
                            state["performance_targets"].update(valid_params)
                            print(f"[DesignExpert] 保留合理参数: {valid_params}")
                        
                        # 构建反问消息
                        error_msg = "⚠️ **检测到参数不合理：**\n\n"
                        error_msg += "\n".join([f"  • {err}" for err in invalid_params])
                        error_msg += "\n\n请重新提供合理的参数值。"
                        
                        # 显示已接受的合理参数
                        if valid_params:
                            error_msg += f"\n\n✅ **已接受的参数：**\n"
                            param_names = {
                                "flow_rate": "目标流量",
                                "efficiency": "目标效率",
                                "pressure_ratio": "目标压比",
                                "top_k": "返回方案数",
                                "batch_size": "批量大小",
                                "timesteps": "扩散步数"
                            }
                            for key, value in valid_params.items():
                                error_msg += f"  • {param_names.get(key, key)}: {value}\n"
                        
                        state["waiting_for_user"] = True
                        state["user_question"] = error_msg
                        state["pending_agent"] = AgentNames.DESIGN_EXPERT
                        state["messages"].append(AIMessage(content=error_msg))
                        
                        return state
                    
                    # 所有参数都合理，累积参数
                    state["performance_targets"].update(valid_params)
                    print(f"[DesignExpert] 累积后参数: {state['performance_targets']}")
                else:
                    print(f"[DesignExpert] 未能从输入中提取参数")
            except Exception as e:
                print(f"[DesignExpert] 参数提取失败: {e}")
        
        # 从state中获取累积的参数
        targets = state["performance_targets"]
        print(f"[DesignExpert] 当前state中的累积参数: {targets}")
        
        # 检查是否有缺失的参数
        required_params = ["flow_rate", "efficiency", "pressure_ratio"]
        missing_params = [p for p in required_params if p not in targets or targets[p] is None]
        
        # 情况1：参数不全 → 反问用户
        if missing_params:
            print(f"[DesignExpert] 参数不完整，缺失: {missing_params}")
            
            # 构建友好的问题
            param_names = {
                "flow_rate": "目标流量（单位：kg/s，参考范围：10-20）",
                "efficiency": "目标效率（范围：0.7-0.95，推荐0.85）",
                "pressure_ratio": "目标压比（范围：1.2-2.0，推荐1.5）"
            }
            
            question_parts = ["❓ 请提供以下缺失的设计参数：\n"]
            for param in missing_params:
                question_parts.append(f"• {param_names[param]}")
            
            # 如果已有部分参数，显示出来
            if targets:
                question_parts.append(f"\n✅ **已提供的参数：**")
                for key, value in targets.items():
                    question_parts.append(f"• {param_names.get(key, key)}: {value}")
            
            question = "\n".join(question_parts)
            
            state["waiting_for_user"] = True
            state["user_question"] = question
            state["pending_agent"] = AgentNames.DESIGN_EXPERT
            state["messages"].append(AIMessage(content=question))
            
            return state
        
        # 情况2：参数完整，需要验证
        print(f"[DesignExpert] 参数完整，验证合理性...")
        
        # 验证参数范围
        validation_errors = []
        if targets["flow_rate"] < 5 or targets["flow_rate"] > 30:
            validation_errors.append(f"流量 {targets['flow_rate']} kg/s 超出合理范围（5-30）")
        if targets["efficiency"] < 0.5 or targets["efficiency"] > 1.0:
            validation_errors.append(f"效率 {targets['efficiency']} 超出合理范围（0.5-1.0）")
        if targets["pressure_ratio"] < 1.0 or targets["pressure_ratio"] > 3.0:
            validation_errors.append(f"压比 {targets['pressure_ratio']} 超出合理范围（1.0-3.0）")
        
        if validation_errors:
            # 参数不合理 → 反问用户
            error_msg = "⚠️ **检测到参数不合理：**\n\n"
            error_msg += "\n".join([f"  • {err}" for err in validation_errors])
            error_msg += "\n\n请重新提供合理的参数值。"
            
            state["waiting_for_user"] = True
            state["user_question"] = error_msg
            state["pending_agent"] = AgentNames.DESIGN_EXPERT
            state["messages"].append(AIMessage(content=error_msg))
            
            print(f"[DesignExpert] 参数不合理，反问用户")
            return state
        
        # 情况3：参数完整且合理 → 调用设计工具
        print(f"[DesignExpert] 参数验证通过，开始设计...")
        print(f"[DesignExpert] 性能目标: {targets}")
        
        # ==================== 获取配置参数（使用默认值或用户指定值）====================
        top_k = targets.get("top_k", 1)  # 默认返回1个方案
        batch_size = targets.get("batch_size", 50)  # 默认批量大小50
        timesteps = targets.get("timesteps", 100)  # 默认扩散步数100
        
        # 确保 batch_size >= top_k
        if batch_size < top_k:
            batch_size = top_k
            print(f"[DesignExpert] 自动调整 batch_size 为 {batch_size}（需 >= top_k）")
        
        print(f"[DesignExpert] 配置参数: top_k={top_k}, batch_size={batch_size}, timesteps={timesteps}")
        
        # 调用设计工具
        design_result = blade_design_tool.invoke({
            "flow_rate": targets["flow_rate"],
            "efficiency": targets["efficiency"],
            "pressure_ratio": targets["pressure_ratio"],
            "top_k": top_k,
            "batch_size": batch_size,
            "timesteps": timesteps
        })
        
        if "error" in design_result:
            state["error_message"] = design_result["error"]
            state["final_result"] = f"❌ 设计失败: {design_result['error']}"
            state["messages"].append(AIMessage(content=state["final_result"]))
            state["next_action"] = EdgeConditions.TO_END
            return state
        
        print(f"[DesignExpert] 设计生成成功")
        
        # 保存结果
        state["design_params"] = design_result
        
        # ==================== 构建响应消息 ====================
        response_msg = "✅ **叶片设计已完成**\n\n"
        
        # 显示配置信息
        response_msg += "**📋 设计配置：**\n"
        response_msg += f"  • 目标流量: {targets['flow_rate']} kg/s\n"
        response_msg += f"  • 目标效率: {targets['efficiency']}\n"
        response_msg += f"  • 目标压比: {targets['pressure_ratio']}\n"
        response_msg += f"  • 返回方案数: Top-{top_k}\n"
        response_msg += f"  • 批量大小: {batch_size}\n"
        response_msg += f"  • 扩散步数: {timesteps}\n\n"
        
        # 定义参数顺序和格式化（简化版，用于表格）
        param_keys = [
            ("root_Angle_in（叶根进口金属角[°]）", "根进口角[°]"),
            ("root_Angle_out（叶根出口金属角[°]）", "根出口角[°]"),
            ("root_Chord（叶根弦长[m]）", "根弦长[m]"),
            ("root_THmax_CH（叶根最大相对厚度[%]）", "根厚度[%]"),
            ("root_THmaxP（叶根最大相对厚度位置[-]）", "根厚度位置"),
            ("root_SWA（叶根最大相对厚度位置[m]）", "根弯度[m]"),
            ("root_BOWA（叶根掠[m]）", "根掠[m]"),
            ("mid_Angle_in（叶中进口金属角[°]）", "中进口角[°]"),
            ("mid_Angle_out（叶中出口金属角[°]）", "中出口角[°]"),
            ("mid_Chord（叶中弦长[m]）", "中弦长[m]"),
            ("mid_THmax_CH（叶中最大相对厚度[%]）", "中厚度[%]"),
            ("mid_THmaxP（叶中最大相对厚度位置[-]）", "中厚度位置"),
            ("mid_SWA（叶中最大相对厚度位置[m]）", "中弯度[m]"),
            ("mid_BOWA（叶中掠[m]）", "中掠[m]"),
            ("tip_Angle_in（叶顶进口金属角[°]）", "顶进口角[°]"),
            ("tip_Angle_out（叶顶出口金属角[°]）", "顶出口角[°]"),
            ("tip_Chord（叶顶弦长[m]）", "顶弦长[m]"),
            ("tip_THmax_CH（叶顶最大相对厚度[%]）", "顶厚度[%]"),
            ("tip_THmaxP（叶顶最大相对厚度位置[-]）", "顶厚度位置"),
            ("tip_SWA（叶顶最大相对厚度位置[m]）", "顶弯度[m]"),
            ("tip_BOWA（叶顶掠[m]）", "顶掠[m]"),
        ]
        
        # 如果只有1个方案，使用详细列表展示
        if top_k == 1:
            design = design_result.get("第1个设计结果", {})
            response_msg += "**🔧 设计参数详情：**\n\n"
            
            # 叶根参数
            response_msg += "**叶根（Root）：**\n"
            for i in range(7):
                full_key, short_name = param_keys[i]
                value = design.get(full_key, 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.3f}"
                response_msg += f"  • {short_name}: {value}\n"
            
            # 叶中参数
            response_msg += "\n**叶中（Mid）：**\n"
            for i in range(7, 14):
                full_key, short_name = param_keys[i]
                value = design.get(full_key, 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.3f}"
                response_msg += f"  • {short_name}: {value}\n"
            
            # 叶顶参数
            response_msg += "\n**叶顶（Tip）：**\n"
            for i in range(14, 21):
                full_key, short_name = param_keys[i]
                value = design.get(full_key, 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.3f}"
                response_msg += f"  • {short_name}: {value}\n"
        else:
            # 多方案：使用表格展示
            response_msg += f"**🔧 Top-{top_k} 设计方案对比表：**\n\n"
            
            # 构建Markdown表格
            # 表头
            header = "| 参数 |"
            separator = "|:---|"
            for k in range(1, top_k + 1):
                header += f" 方案{k} |"
                separator += ":---:|"
            response_msg += header + "\n" + separator + "\n"
            
            # 表格内容 - 按参数分组
            groups = [
                ("**叶根**", 0, 7),
                ("**叶中**", 7, 14),
                ("**叶顶**", 14, 21),
            ]
            
            for group_name, start, end in groups:
                # 添加分组行
                response_msg += f"| {group_name} |" + " |" * top_k + "\n"
                
                for i in range(start, end):
                    full_key, short_name = param_keys[i]
                    row = f"| {short_name} |"
                    for k in range(1, top_k + 1):
                        design = design_result.get(f"第{k}个设计结果", {})
                        value = design.get(full_key, 'N/A')
                        if isinstance(value, (int, float)):
                            value = f"{value:.3f}"
                        row += f" {value} |"
                    response_msg += row + "\n"
        
        response_msg += "\n💡 所有参数已生成，可用于性能评估和3D可视化。"
        
        state["final_result"] = response_msg
        
        # ==================== 存储任务结果 ====================
        store_task_result(state, "design")
        
        log_collaboration(
            state,
            from_agent=AgentNames.DESIGN_EXPERT,
            to_agent="user",
            action="返回设计结果",
            details=f"生成 Top-{top_k} 设计方案（批量{batch_size}，步数{timesteps}）"
        )
        
        # ==================== 检查是否有后续任务（在添加消息之前检查，避免重复）====================
        if state.get("task_mode") == "sequential" and len(state.get("task_queue", [])) > 1:
            return _handle_next_task(state, "design", response_msg)
        
        # 非多任务序列，正常添加消息
        state["messages"].append(AIMessage(content=response_msg))
        state["next_action"] = EdgeConditions.TO_END
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[DesignExpert] 错误: {str(e)}")
        state["error_message"] = f"设计专家执行失败: {str(e)}"
        state["final_result"] = f"❌ 设计失败: {str(e)}"
        state["messages"].append(AIMessage(content=state["final_result"]))
        state["next_action"] = EdgeConditions.TO_END
    
    return state


# ==================== 评估专家节点（增强版）====================

def evaluation_expert_node(state: MultiAgentState) -> MultiAgentState:
    """
    评估专家Agent - 增强版
    
    新增能力：
    1. 自动使用已有设计结果
    2. 存储评估结果
    3. 支持多任务序列
    
    职责：
    1. 检查是否使用已有设计结果
    2. 从用户输入或state中提取21维参数
    3. 如果参数不全，详细列出缺失参数并反问
    4. 如果参数不合理，验证并反问
    5. 参数完整且合理，调用评估工具
    """
    
    print(f"\n{'='*60}")
    print(f"[EvaluationExpert] 开始评估性能...")
    print(f"{'='*60}")
    
    try:
        user_input = state.get("user_input", "")
        use_existing = state.get("use_existing_result")
        
        # ==================== 新增：检查是否使用已有设计结果 ====================
        design_params = state.get("design_params")
        stored_design = state.get("task_results", {}).get("design")
        
        # 合并获取设计结果
        if not design_params and stored_design:
            design_params = stored_design
            state["design_params"] = stored_design
            print(f"[EvaluationExpert] 从task_results加载设计结果")
        
        # 如果明确要使用已有结果且有设计结果
        if use_existing is True and design_params and "第1个设计结果" in design_params:
            print(f"[EvaluationExpert] 使用已有设计结果进行评估")
            
            # ==================== 新增：识别用户选择的方案 ====================
            available_count = get_available_scheme_count(design_params)
            print(f"[EvaluationExpert] 可用方案数量: {available_count}")
            
            # 【关键改进】使用原始用户输入，精确提取与"评估"任务相关的方案
            # 在多任务序列中，这样可以区分不同任务各自对应哪些方案
            original_input = state.get("original_user_input")
            input_for_extraction = original_input if original_input else user_input
            
            llm = create_llm()
            
            # 使用精确任务方案提取函数
            selected_schemes = extract_task_specific_schemes(
                user_input=input_for_extraction,
                task_type="evaluate",  # 指定当前任务类型
                available_count=available_count,
                llm=llm
            )
            
            # 如果未指定，默认评估所有方案（当有多个时）或第一个（当只有一个时）
            if not selected_schemes:
                if available_count > 1:
                    # 询问用户要评估哪个方案
                    question = f"📋 **检测到 {available_count} 个设计方案**\n\n"
                    question += "请指定要评估的方案：\n"
                    question += "• 输入「评估第2个方案」评估特定方案\n"
                    question += "• 输入「评估方案1和3」评估多个方案\n"
                    question += "• 输入「评估所有方案」评估全部\n"
                    question += "• 或直接回复「继续」评估第1个方案\n"
                    
                    state["waiting_for_user"] = True
                    state["user_question"] = question
                    state["pending_agent"] = AgentNames.EVALUATION_EXPERT
                    state["messages"].append(AIMessage(content=question))
                    return state
                else:
                    selected_schemes = [1]
            
            print(f"[EvaluationExpert] 选中方案: {selected_schemes}")
            
            # 提取21维参数的键列表
            param_keys = [
                "root_Angle_in（叶根进口金属角[°]）", "root_Angle_out（叶根出口金属角[°]）",
                "root_Chord（叶根弦长[m]）", "root_THmax_CH（叶根最大相对厚度[%]）",
                "root_THmaxP（叶根最大相对厚度位置[-]）", "root_SWA（叶根最大相对厚度位置[m]）",
                "root_BOWA（叶根掠[m]）",
                "mid_Angle_in（叶中进口金属角[°]）", "mid_Angle_out（叶中出口金属角[°]）",
                "mid_Chord（叶中弦长[m]）", "mid_THmax_CH（叶中最大相对厚度[%]）",
                "mid_THmaxP（叶中最大相对厚度位置[-]）", "mid_SWA（叶中最大相对厚度位置[m]）",
                "mid_BOWA（叶中掠[m]）",
                "tip_Angle_in（叶顶进口金属角[°]）", "tip_Angle_out（叶顶出口金属角[°]）",
                "tip_Chord（叶顶弦长[m]）", "tip_THmax_CH（叶顶最大相对厚度[%]）",
                "tip_THmaxP（叶顶最大相对厚度位置[-]）", "tip_SWA（叶顶最大相对厚度位置[m]）",
                "tip_BOWA（叶顶掠[m]）"
            ]
            
            # 提取选中方案的参数
            params_list = []
            for scheme_idx in selected_schemes:
                design = design_params.get(f"第{scheme_idx}个设计结果", {})
                if design:
                    params_list.append([design.get(key, 0) for key in param_keys])
            
            if not params_list:
                state["error_message"] = "未找到有效的设计方案"
                state["final_result"] = "❌ 评估失败: 未找到有效的设计方案"
                state["messages"].append(AIMessage(content=state["final_result"]))
                state["next_action"] = EdgeConditions.TO_END
                return state
            
            # 调用评估工具
            print(f"[EvaluationExpert] 调用评估工具，评估 {len(params_list)} 个方案...")
            eval_result = blade_performance_evaluation_tool.invoke({
                "design_params": params_list
            })
            
            if "error" in eval_result:
                state["error_message"] = eval_result["error"]
                state["final_result"] = f"❌ 评估失败: {eval_result['error']}"
                state["messages"].append(AIMessage(content=state["final_result"]))
                state["next_action"] = EdgeConditions.TO_END
                return state
            
            # 保存结果
            state["evaluation_results"] = eval_result
            store_task_result(state, "evaluate")
            
            # ==================== 构建响应消息 ====================
            if len(selected_schemes) == 1:
                # 单方案评估：详细展示
                response_msg = f"✅ **性能评估已完成**（方案{selected_schemes[0]}）\n\n"
                first_result = eval_result.get("第1个设计性能结果", {})
                
                flow_rate = first_result.get("mass flow rate（预测流量[kg/s]）", "N/A")
                efficiency = first_result.get("isentropic efficiency（预测等熵效率[-]）", "N/A")
                pressure_ratio = first_result.get("total pressure ratio（预测总压比[-]）", "N/A")
                
                if isinstance(flow_rate, (int, float)):
                    flow_rate = f"{flow_rate:.3f}"
                if isinstance(efficiency, (int, float)):
                    efficiency = f"{efficiency:.3f}"
                if isinstance(pressure_ratio, (int, float)):
                    pressure_ratio = f"{pressure_ratio:.3f}"
                
                response_msg += f"**性能指标：**\n"
                response_msg += f"- 流量：{flow_rate} kg/s\n"
                response_msg += f"- 效率：{efficiency}\n"
                response_msg += f"- 压比：{pressure_ratio}"
            else:
                # 多方案评估：表格展示
                response_msg = f"✅ **性能评估已完成**（共 {len(selected_schemes)} 个方案）\n\n"
                
                # 构建Markdown表格
                response_msg += "| 方案 | 流量 (kg/s) | 效率 | 压比 |\n"
                response_msg += "|:---:|:---:|:---:|:---:|\n"
                
                for i, scheme_idx in enumerate(selected_schemes):
                    result = eval_result.get(f"第{i+1}个设计性能结果", {})
                    
                    flow_rate = result.get("mass flow rate（预测流量[kg/s]）", "N/A")
                    efficiency = result.get("isentropic efficiency（预测等熵效率[-]）", "N/A")
                    pressure_ratio = result.get("total pressure ratio（预测总压比[-]）", "N/A")
                    
                    if isinstance(flow_rate, (int, float)):
                        flow_rate = f"{flow_rate:.3f}"
                    if isinstance(efficiency, (int, float)):
                        efficiency = f"{efficiency:.3f}"
                    if isinstance(pressure_ratio, (int, float)):
                        pressure_ratio = f"{pressure_ratio:.3f}"
                    
                    response_msg += f"| 方案{scheme_idx} | {flow_rate} | {efficiency} | {pressure_ratio} |\n"
                
                response_msg += "\n💡 可以继续对某个方案进行优化或可视化。"
            
            state["final_result"] = response_msg
            
            # 检查多任务序列（在添加消息之前检查，避免重复）
            if state.get("task_mode") == "sequential":
                return _handle_next_task(state, "evaluate", response_msg)
            
            # 非多任务序列，正常添加消息
            state["messages"].append(AIMessage(content=response_msg))
            state["next_action"] = EdgeConditions.TO_END
            return state
        
        # ==================== 原有逻辑：参数提取与累积 ====================
        # 初始化 partial_blade_params（如果不存在）
        if "partial_blade_params" not in state:
            state["partial_blade_params"] = {}
        
        # 尝试从用户输入中提取参数（无论是否有design_params）
        if user_input and state.get("intent_type") != "use_existing":
            print(f"[EvaluationExpert] 尝试从用户输入提取参数...")
            llm = create_llm()
            extracted_params = parse_user_blade_params(user_input, llm)
            
            if extracted_params:
                print(f"[EvaluationExpert] 本次提取到 {len(extracted_params)} 个参数")
                
                # 立即验证本次提取的参数
                is_valid, validation_errors = validate_blade_params(extracted_params)
                
                if not is_valid:
                    print(f"[EvaluationExpert] 检测到不合理参数，进行筛选")
                    
                    valid_params = {}
                    invalid_info = []
                    
                    for param_key, param_value in extracted_params.items():
                        if param_key in BLADE_21D_PARAMS:
                            param_info = BLADE_21D_PARAMS[param_key]
                            min_val, max_val = param_info["range"]
                            
                            if min_val <= param_value <= max_val:
                                valid_params[param_key] = param_value
                            else:
                                invalid_info.append(
                                    f"{param_info['name']}（{param_key}）= {param_value} {param_info['unit']} "
                                    f"超出合理范围（{min_val}-{max_val}）"
                                )
                    
                    if valid_params:
                        state["partial_blade_params"].update(valid_params)
                        print(f"[EvaluationExpert] 保留 {len(valid_params)} 个合理参数")
                    
                    error_msg = "⚠️ **检测到参数不合理：**\n\n"
                    error_msg += "\n".join([f"  • {err}" for err in invalid_info])
                    error_msg += "\n\n请重新提供合理的参数值。"
                    
                    if valid_params:
                        error_msg += f"\n\n✅ **本次已接受的合理参数：**\n"
                        for key in valid_params.keys():
                            param_info = BLADE_21D_PARAMS[key]
                            error_msg += f"  • {param_info['name']}（{key}）\n"
                    
                    total_accumulated = len(state["partial_blade_params"])
                    error_msg += f"\n📊 **当前累积参数：{total_accumulated}/21**"
                    
                    state["waiting_for_user"] = True
                    state["user_question"] = error_msg
                    state["pending_agent"] = AgentNames.EVALUATION_EXPERT
                    state["messages"].append(AIMessage(content=error_msg))
                    
                    return state
                
                state["partial_blade_params"].update(extracted_params)
                print(f"[EvaluationExpert] 当前累积参数数量: {len(state['partial_blade_params'])}/21")
            else:
                print(f"[EvaluationExpert] 未能从输入中提取参数")
        
        partial_params = state["partial_blade_params"]
        print(f"[EvaluationExpert] 当前state中的累积参数: {len(partial_params)}/21")
        
        # 情况1：用户提供了部分参数，需要继续补充
        if partial_params and len(partial_params) < 21:
            print(f"[EvaluationExpert] 参数不完整（{len(partial_params)}/21），反问用户")
            
            question = format_missing_params_question(partial_params, "性能评估")
            
            state["waiting_for_user"] = True
            state["user_question"] = question
            state["pending_agent"] = AgentNames.EVALUATION_EXPERT
            state["messages"].append(AIMessage(content=question))
            
            return state
        
        # 情况2：参数完整，需要验证
        if partial_params and len(partial_params) == 21:
            print(f"[EvaluationExpert] 参数完整，验证合理性...")
            is_valid, errors = validate_blade_params(partial_params)
            
            if not is_valid:
                error_msg = "⚠️ **检测到参数不合理：**\n\n"
                error_msg += "\n".join([f"  • {err}" for err in errors])
                error_msg += "\n\n请重新提供合理的参数值。"
                
                state["waiting_for_user"] = True
                state["user_question"] = error_msg
                state["pending_agent"] = AgentNames.EVALUATION_EXPERT
                state["messages"].append(AIMessage(content=error_msg))
                
                print(f"[EvaluationExpert] 参数不合理，反问用户")
                return state
            
            print(f"[EvaluationExpert] 参数验证通过，准备评估...")
            params_list = [[partial_params[key] for key in BLADE_21D_PARAMS_ORDER]]
        
        # ==================== 新增情况：明确要求新任务但没有提供完整参数 ====================
        elif state.get("use_existing_result") == False:
            # 用户明确说"新的"任务，不应使用已有结果
            print(f"[EvaluationExpert] 用户要求新任务，不使用已有结果，需要询问参数")
            
            question = """📋 **您要进行一次新的性能评估**

请提供21维叶片设计参数进行评估。您可以：
1. 直接提供参数，例如：「叶根弦长0.17m，叶根进口角57度，叶根出口角-33度...」
2. 或者说「先设计一个叶片」，然后再评估

**需要的参数包括：**
- 叶根(7个): 进口角、出口角、弦长、最大厚度、最大厚度位置、SWA、BOWA
- 叶中(7个): 进口角、出口角、弦长、最大厚度、最大厚度位置、SWA、BOWA  
- 叶顶(7个): 进口角、出口角、弦长、最大厚度、最大厚度位置、SWA、BOWA"""
            
            state["waiting_for_user"] = True
            state["user_question"] = question
            state["pending_agent"] = AgentNames.EVALUATION_EXPERT
            state["messages"].append(AIMessage(content=question))
            
            return state
        
        # 情况3：没有用户参数，但有design_params（且未明确拒绝使用已有结果）
        elif design_params and "第1个设计结果" in design_params:
            print(f"[EvaluationExpert] 使用设计专家提供的参数...")
            
            # ==================== 新增：识别用户选择的方案 ====================
            available_count = get_available_scheme_count(design_params)
            
            # 【关键改进】使用原始用户输入，精确提取与"评估"任务相关的方案
            original_input = state.get("original_user_input")
            input_for_extraction = original_input if original_input else user_input
            
            llm = create_llm()
            selected_schemes = extract_task_specific_schemes(
                user_input=input_for_extraction,
                task_type="evaluate",  # 指定当前任务类型
                available_count=available_count,
                llm=llm
            )
            
            # 如果未指定且有多个方案，询问用户
            if not selected_schemes:
                if available_count > 1:
                    question = f"📋 **检测到 {available_count} 个设计方案**\n\n"
                    question += "请指定要评估的方案：\n"
                    question += "• 输入「评估第2个方案」评估特定方案\n"
                    question += "• 输入「评估方案1和3」评估多个方案\n"
                    question += "• 输入「评估所有方案」评估全部\n"
                    question += "• 或直接回复「继续」评估第1个方案\n"
                    
                    state["waiting_for_user"] = True
                    state["user_question"] = question
                    state["pending_agent"] = AgentNames.EVALUATION_EXPERT
                    state["messages"].append(AIMessage(content=question))
                    return state
                else:
                    selected_schemes = [1]
            
            print(f"[EvaluationExpert] 选中方案: {selected_schemes}")
            
            param_keys = [
                "root_Angle_in（叶根进口金属角[°]）", "root_Angle_out（叶根出口金属角[°]）",
                "root_Chord（叶根弦长[m]）", "root_THmax_CH（叶根最大相对厚度[%]）",
                "root_THmaxP（叶根最大相对厚度位置[-]）", "root_SWA（叶根最大相对厚度位置[m]）",
                "root_BOWA（叶根掠[m]）",
                "mid_Angle_in（叶中进口金属角[°]）", "mid_Angle_out（叶中出口金属角[°]）",
                "mid_Chord（叶中弦长[m]）", "mid_THmax_CH（叶中最大相对厚度[%]）",
                "mid_THmaxP（叶中最大相对厚度位置[-]）", "mid_SWA（叶中最大相对厚度位置[m]）",
                "mid_BOWA（叶中掠[m]）",
                "tip_Angle_in（叶顶进口金属角[°]）", "tip_Angle_out（叶顶出口金属角[°]）",
                "tip_Chord（叶顶弦长[m]）", "tip_THmax_CH（叶顶最大相对厚度[%]）",
                "tip_THmaxP（叶顶最大相对厚度位置[-]）", "tip_SWA（叶顶最大相对厚度位置[m]）",
                "tip_BOWA（叶顶掠[m]）"
            ]
            
            # 提取选中方案的参数
            params_list = []
            for scheme_idx in selected_schemes:
                design = design_params.get(f"第{scheme_idx}个设计结果", {})
                if design:
                    params_list.append([design.get(key, 0) for key in param_keys])
            
            # 保存选中的方案索引供后续展示使用
            state["_selected_schemes"] = selected_schemes
        
        # 情况4：没有任何参数
        else:
            print(f"[EvaluationExpert] 缺少设计参数，询问用户")
            
            question = """❓ 性能评估需要21维设计参数。

您可以：
1. **先生成设计** - 说「我需要设计一个叶片」
2. **提供已有参数** - 直接提供21维设计参数，例如「叶根弦长为0.6m，叶根进口角为57度...」

推荐您先生成设计，然后再进行评估。"""
            
            state["waiting_for_user"] = True
            state["user_question"] = question
            state["pending_agent"] = AgentNames.EVALUATION_EXPERT
            state["messages"].append(AIMessage(content=question))
            
            return state
        
        print(f"[EvaluationExpert] 调用评估工具，评估 {len(params_list)} 个方案...")
        
        eval_result = blade_performance_evaluation_tool.invoke({
            "design_params": params_list
        })
        
        if "error" in eval_result:
            state["error_message"] = eval_result["error"]
            state["final_result"] = f"❌ 评估失败: {eval_result['error']}"
            state["messages"].append(AIMessage(content=state["final_result"]))
            state["next_action"] = EdgeConditions.TO_END
            return state
        
        print(f"[EvaluationExpert] 评估完成")
        
        # 保存结果
        state["evaluation_results"] = eval_result
        
        # ==================== 新增：存储任务结果 ====================
        store_task_result(state, "evaluate")
        
        # ==================== 构建响应消息（支持多方案）====================
        selected_schemes = state.get("_selected_schemes", [1])  # 默认第1个方案
        
        if len(selected_schemes) == 1:
            # 单方案评估：详细展示
            scheme_idx = selected_schemes[0]
            response_msg = f"✅ **性能评估已完成**（方案{scheme_idx}）\n\n"
            response_msg += f"**性能指标：**\n"
            
            first_result = eval_result.get("第1个设计性能结果", {})
            flow_rate = first_result.get("mass flow rate（预测流量[kg/s]）", "N/A")
            efficiency = first_result.get("isentropic efficiency（预测等熵效率[-]）", "N/A")
            pressure_ratio = first_result.get("total pressure ratio（预测总压比[-]）", "N/A")
            
            if isinstance(flow_rate, (int, float)):
                flow_rate = f"{flow_rate:.3f}"
            if isinstance(efficiency, (int, float)):
                efficiency = f"{efficiency:.3f}"
            if isinstance(pressure_ratio, (int, float)):
                pressure_ratio = f"{pressure_ratio:.3f}"
            
            response_msg += f"- 流量：{flow_rate} kg/s\n"
            response_msg += f"- 效率：{efficiency}\n"
            response_msg += f"- 压比：{pressure_ratio}"
        else:
            # 多方案评估：表格展示
            response_msg = f"✅ **性能评估已完成**（共 {len(selected_schemes)} 个方案）\n\n"
            
            # 构建Markdown表格
            response_msg += "| 方案 | 流量 (kg/s) | 效率 | 压比 |\n"
            response_msg += "|:---:|:---:|:---:|:---:|\n"
            
            for i, scheme_idx in enumerate(selected_schemes):
                result = eval_result.get(f"第{i+1}个设计性能结果", {})
                
                flow_rate = result.get("mass flow rate（预测流量[kg/s]）", "N/A")
                efficiency = result.get("isentropic efficiency（预测等熵效率[-]）", "N/A")
                pressure_ratio = result.get("total pressure ratio（预测总压比[-]）", "N/A")
                
                if isinstance(flow_rate, (int, float)):
                    flow_rate = f"{flow_rate:.3f}"
                if isinstance(efficiency, (int, float)):
                    efficiency = f"{efficiency:.3f}"
                if isinstance(pressure_ratio, (int, float)):
                    pressure_ratio = f"{pressure_ratio:.3f}"
                
                response_msg += f"| 方案{scheme_idx} | {flow_rate} | {efficiency} | {pressure_ratio} |\n"
            
            response_msg += "\n💡 可以继续对某个方案进行优化或可视化。"
        
        state["final_result"] = response_msg
        
        log_collaboration(
            state,
            from_agent=AgentNames.EVALUATION_EXPERT,
            to_agent="user",
            action="返回评估结果",
            details=f"评估了 {len(selected_schemes)} 个方案"
        )
        
        # ==================== 检查是否有后续任务（在添加消息之前检查，避免重复）====================
        if state.get("task_mode") == "sequential" and len(state.get("task_queue", [])) > 1:
            return _handle_next_task(state, "evaluate", response_msg)
        
        # 非多任务序列，正常添加消息
        state["messages"].append(AIMessage(content=response_msg))
        state["next_action"] = EdgeConditions.TO_END
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[EvaluationExpert] 错误: {str(e)}")
        state["error_message"] = f"评估专家执行失败: {str(e)}"
        state["final_result"] = f"❌ 评估失败: {str(e)}"
        state["messages"].append(AIMessage(content=state["final_result"]))
        state["next_action"] = EdgeConditions.TO_END
    
    return state


# ==================== 可视化专家节点（增强版）====================

def visualization_expert_node(state: MultiAgentState) -> MultiAgentState:
    """
    可视化专家Agent - 增强版（支持多方案可视化）
    
    功能：
    1. 支持从设计结果CSV可视化指定方案
    2. 支持用户手动输入21维参数的可视化
    3. 支持多方案选择（类似评估和优化专家）
    4. 通过几何生成模块生成真实DAT文件
    5. 使用Plot_blade进行高级3D可视化
    
    工作流程：
    1. 检查是否有设计结果（CSV）
    2. 如果有多个方案，识别用户要可视化哪些方案
    3. 调用几何生成模块生成DAT文件
    4. 调用可视化模块生成3D图片
    """
    
    print(f"\n{'='*60}")
    print(f"[VisualizationExpert] 开始生成可视化...")
    print(f"{'='*60}")
    
    try:
        user_input = state.get("user_input", "")
        use_existing = state.get("use_existing_result")
        
        # ==================== 检查是否使用已有设计结果 ====================
        design_params = state.get("design_params")
        stored_design = state.get("task_results", {}).get("design")
        
        # 合并获取设计结果
        if not design_params and stored_design:
            design_params = stored_design
            state["design_params"] = stored_design
            print(f"[VisualizationExpert] 从task_results加载设计结果")
        
        # ==================== 情况1：使用已有设计结果（支持多方案选择） ====================
        if use_existing is True and design_params and "第1个设计结果" in design_params:
            print(f"[VisualizationExpert] 使用已有设计结果进行可视化")
            
            # 获取可用方案数量
            available_count = get_available_scheme_count(design_params)
            print(f"[VisualizationExpert] 可用方案数量: {available_count}")
            
            # 获取CSV路径（如果有）
            csv_path = design_params.get("csv_path")
            
            # 精确提取可视化任务相关的方案
            original_input = state.get("original_user_input")
            input_for_extraction = original_input if original_input else user_input
            
            llm = create_llm()
            selected_schemes = extract_task_specific_schemes(
                user_input=input_for_extraction,
                task_type="visualize",
                available_count=available_count,
                llm=llm
            )
            
            # 如果未指定方案，默认可视化第一个
            if not selected_schemes:
                if available_count > 1:
                    # 询问用户要可视化哪个方案
                    question = f"📋 **检测到 {available_count} 个设计方案**\n\n"
                    question += "请指定要可视化的方案：\n"
                    question += "• 输入「可视化第2个方案」可视化特定方案\n"
                    question += "• 输入「可视化方案1和3」可视化多个方案\n"
                    question += "• 输入「可视化所有方案」可视化全部\n"
                    question += "• 或直接回复「继续」可视化第1个方案\n"
                    
                    state["waiting_for_user"] = True
                    state["user_question"] = question
                    state["pending_agent"] = AgentNames.VISUALIZATION_EXPERT
                    state["messages"].append(AIMessage(content=question))
                    return state
                else:
                    selected_schemes = [1]
            
            print(f"[VisualizationExpert] 选中方案: {selected_schemes}")
            
            # ==================== 调用可视化工具 ====================
            # 如果有CSV路径，使用几何生成 + 高级可视化
            if csv_path and os.path.exists(csv_path):
                print(f"[VisualizationExpert] 使用CSV几何生成方式")
                from langgraph_tools import visualize_schemes_from_csv
                
                viz_result = visualize_schemes_from_csv(csv_path, selected_schemes)
                
                if viz_result["status"] == "error":
                    state["error_message"] = viz_result["message"]
                    state["final_result"] = f"❌ 可视化失败: {viz_result['message']}"
                    state["messages"].append(AIMessage(content=state["final_result"]))
                    state["next_action"] = EdgeConditions.TO_END
                    return state
                
                # 构建多方案响应
                visualizations = viz_result["visualizations"]
                
                # 导入可视化进度报告函数
                from Tools.blade_visualization import _report_visualization_progress
                
                if len(visualizations) == 1:
                    # 单方案
                    idx = list(visualizations.keys())[0]
                    viz_info = visualizations[idx]
                    state["visualization_path"] = viz_info.get("image_path", "")
                    state["visualization_json_path"] = viz_info.get("json_path", "")
                    
                    # 发送单方案图片到前端
                    _report_visualization_progress(
                        'visualization_image', 100,
                        label=f'方案{idx}',
                        image_path=state["visualization_path"],
                        json_path=state["visualization_json_path"]
                    )
                    
                    response_msg = f"✅ **方案{idx} 3D可视化已生成**\n\n"
                    response_msg += f"📊 图片路径：{state['visualization_path']}"
                else:
                    # 多方案
                    all_paths = []
                    
                    # 发送可视化开始信号（清空前端图库）
                    _report_visualization_progress('visualization_start', 0, '开始生成多方案可视化')
                    
                    for idx, viz_info in visualizations.items():
                        img_path = viz_info.get("image_path", "")
                        json_path = viz_info.get("json_path", "")
                        
                        all_paths.append({
                            "scheme": idx,
                            "image_path": img_path,
                            "json_path": json_path
                        })
                        
                        # 逐个发送每个方案的图片到前端
                        _report_visualization_progress(
                            'visualization_image', 100,
                            label=f'方案{idx}',
                            image_path=img_path,
                            json_path=json_path
                        )
                        print(f"[VisualizationExpert] 已发送方案{idx}图片到前端")
                    
                    # 存储第一个方案的路径作为默认显示
                    first_idx = list(visualizations.keys())[0]
                    state["visualization_path"] = visualizations[first_idx].get("image_path", "")
                    state["visualization_json_path"] = visualizations[first_idx].get("json_path", "")
                    state["_visualization_all_schemes"] = all_paths  # 存储所有方案路径
                    
                    response_msg = f"✅ **多方案3D可视化已生成**（共 {len(visualizations)} 个方案）\n\n"
                    response_msg += "| 方案 | 状态 |\n|:---:|:---:|\n"
                    for idx in visualizations.keys():
                        response_msg += f"| 方案{idx} | ✅ 已生成 |\n"
                    response_msg += '\n💡 使用前端左右箭头（◀ ▶）切换查看不同方案，点击"3D交互"查看3D效果。'
                
            else:
                # 没有CSV，使用传统方式（默认DAT文件）
                print(f"[VisualizationExpert] 使用默认DAT文件方式")
                viz_result = blade_visualization_tool.invoke({})
                
                if "error" in viz_result:
                    state["error_message"] = viz_result["error"]
                    state["final_result"] = f"❌ 可视化失败: {viz_result['error']}"
                    state["messages"].append(AIMessage(content=state["final_result"]))
                    state["next_action"] = EdgeConditions.TO_END
                    return state
                
                state["visualization_path"] = viz_result.get("image_path", "")
                state["visualization_json_path"] = viz_result.get("json_path", "")
                response_msg = "✅ **3D可视化已生成**（使用已有设计结果）\n\n"
                response_msg += f"📊 图片路径：{state['visualization_path']}"
            
            # 存储结果
            store_task_result(state, "visualize")
            state["final_result"] = response_msg
            
            # 检查多任务序列
            if state.get("task_mode") == "sequential":
                return _handle_next_task(state, "visualize", response_msg,
                                        image_path=state.get("visualization_path"),
                                        json_path=state.get("visualization_json_path"))
            
            state["messages"].append(AIMessage(content=response_msg))
            state["next_action"] = EdgeConditions.TO_END
            return state
        
        # ==================== 原有逻辑：参数提取与累积 ====================
        # 初始化 partial_blade_params（如果不存在）
        if "partial_blade_params" not in state:
            state["partial_blade_params"] = {}
        
        # 尝试从用户输入中提取参数（无论是否有design_params）
        if user_input:
            print(f"[VisualizationExpert] 尝试从用户输入提取参数...")
            llm = create_llm()
            extracted_params = parse_user_blade_params(user_input, llm)
            
            if extracted_params:
                print(f"[VisualizationExpert] 本次提取到 {len(extracted_params)} 个参数")
                
                # ✅ 新增：立即验证本次提取的参数
                is_valid, validation_errors = validate_blade_params(extracted_params)
                
                if not is_valid:
                    # 有不合理的参数 → 筛选出合理和不合理的部分
                    print(f"[VisualizationExpert] 检测到不合理参数，进行筛选")
                    
                    valid_params = {}
                    invalid_info = []
                    
                    for param_key, param_value in extracted_params.items():
                        if param_key in BLADE_21D_PARAMS:
                            param_info = BLADE_21D_PARAMS[param_key]
                            min_val, max_val = param_info["range"]
                            
                            if min_val <= param_value <= max_val:
                                # 参数合理，保留
                                valid_params[param_key] = param_value
                            else:
                                # 参数不合理，记录
                                invalid_info.append(
                                    f"{param_info['name']}（{param_key}）= {param_value} {param_info['unit']} "
                                    f"超出合理范围（{min_val}-{max_val}）"
                                )
                    
                    # 累积合理的参数
                    if valid_params:
                        state["partial_blade_params"].update(valid_params)
                        print(f"[VisualizationExpert] 保留 {len(valid_params)} 个合理参数")
                    
                    # 立即反问不合理的参数
                    error_msg = "⚠️ **检测到参数不合理：**\n\n"
                    error_msg += "\n".join([f"  • {err}" for err in invalid_info])
                    error_msg += "\n\n请重新提供合理的参数值。"
                    
                    # 显示已接受的合理参数
                    if valid_params:
                        error_msg += f"\n\n✅ **本次已接受的合理参数：**\n"
                        for key in valid_params.keys():
                            param_info = BLADE_21D_PARAMS[key]
                            error_msg += f"  • {param_info['name']}（{key}）\n"
                    
                    # 显示当前累积进度
                    total_accumulated = len(state["partial_blade_params"])
                    error_msg += f"\n📊 **当前累积参数：{total_accumulated}/21**"
                    
                    state["waiting_for_user"] = True
                    state["user_question"] = error_msg
                    state["pending_agent"] = AgentNames.VISUALIZATION_EXPERT
                    state["messages"].append(AIMessage(content=error_msg))
                    
                    return state
                
                # 所有参数都合理，累积参数
                state["partial_blade_params"].update(extracted_params)
                print(f"[VisualizationExpert] 当前累积参数数量: {len(state['partial_blade_params'])}/21")
            else:
                print(f"[VisualizationExpert] 未能从输入中提取参数")
        
        # 从state中获取累积的参数（直接访问，因为已在TypedDict中定义）
        partial_params = state["partial_blade_params"]
        print(f"[VisualizationExpert] 当前state中的累积参数: {len(partial_params)}/21")
        
        # 情况1：用户提供了部分参数，需要继续补充
        if partial_params and len(partial_params) < 21:
            # 参数不完整 → 详细列出缺失参数
            print(f"[VisualizationExpert] 参数不完整（{len(partial_params)}/21），反问用户")
            
            question = format_missing_params_question(partial_params, "可视化")
            
            state["waiting_for_user"] = True
            state["user_question"] = question
            state["pending_agent"] = AgentNames.VISUALIZATION_EXPERT
            state["messages"].append(AIMessage(content=question))
            
            return state
        
        # 情况2：参数完整，需要验证
        if partial_params and len(partial_params) == 21:
            print(f"[VisualizationExpert] 参数完整，验证合理性...")
            is_valid, errors = validate_blade_params(partial_params)
            
            if not is_valid:
                # 参数不合理 → 反问用户
                error_msg = "⚠️ **检测到参数不合理：**\n\n"
                error_msg += "\n".join([f"  • {err}" for err in errors])
                error_msg += "\n\n请重新提供合理的参数值。"
                
                state["waiting_for_user"] = True
                state["user_question"] = error_msg
                state["pending_agent"] = AgentNames.VISUALIZATION_EXPERT
                state["messages"].append(AIMessage(content=error_msg))
                
                print(f"[VisualizationExpert] 参数不合理，反问用户")
                return state
            
            # 参数完整且合理 → 使用用户提供的参数
            print(f"[VisualizationExpert] 参数验证通过，准备可视化...")
            viz_params = partial_params
        
        # ==================== 新增情况：明确要求新任务但没有提供完整参数 ====================
        elif state.get("use_existing_result") == False:
            # 用户明确说"新的"任务，不应使用已有结果
            print(f"[VisualizationExpert] 用户要求新任务，不使用已有结果，需要询问参数")
            
            question = """📋 **您要进行一次新的3D可视化**

请提供21维叶片设计参数进行可视化。您可以：
1. 直接提供参数，例如：「叶根弦长0.17m，叶根进口角57度，叶根出口角-33度...」
2. 或者说「先设计一个叶片」，然后再可视化

**需要的参数包括：**
- 叶根(7个): 进口角、出口角、弦长、最大厚度、最大厚度位置、SWA、BOWA
- 叶中(7个): 进口角、出口角、弦长、最大厚度、最大厚度位置、SWA、BOWA  
- 叶顶(7个): 进口角、出口角、弦长、最大厚度、最大厚度位置、SWA、BOWA"""
            
            state["waiting_for_user"] = True
            state["user_question"] = question
            state["pending_agent"] = AgentNames.VISUALIZATION_EXPERT
            state["messages"].append(AIMessage(content=question))
            
            return state
        
        # 情况3：没有用户参数，但有design_params（从设计专家获得，且未明确拒绝使用已有结果）
        elif design_params and "第1个设计结果" in design_params:
            print(f"[VisualizationExpert] 使用设计专家提供的参数...")
            # 提取参数
            first_design = design_params.get("第1个设计结果", {})
            
            # 键名映射
            viz_params = {
                "root_Angle_in": first_design.get("root_Angle_in（叶根进口金属角[°]）", 57),
                "root_Angle_out": first_design.get("root_Angle_out（叶根出口金属角[°]）", -33),
                "root_Chord": first_design.get("root_Chord（叶根弦长[m]）", 0.17),
                "root_THmax_CH": first_design.get("root_THmax_CH（叶根最大相对厚度[%]）", 15.73),
                "root_THmaxP": first_design.get("root_THmaxP（叶根最大相对厚度位置[-]）", 0.76),
                "root_SWA": first_design.get("root_SWA（叶根最大相对厚度位置[m]）", 0.01),
                "root_BOWA": first_design.get("root_BOWA（叶根掠[m]）", 0.01),
                
                "mid_Angle_in": first_design.get("mid_Angle_in（叶中进口金属角[°]）", 55),
                "mid_Angle_out": first_design.get("mid_Angle_out（叶中出口金属角[°]）", 25),
                "mid_Chord": first_design.get("mid_Chord（叶中弦长[m]）", 0.2),
                "mid_THmax_CH": first_design.get("mid_THmax_CH（叶中最大相对厚度[%]）", 6.7),
                "mid_THmaxP": first_design.get("mid_THmaxP（叶中最大相对厚度位置[-]）", 0.6),
                "mid_SWA": first_design.get("mid_SWA（叶中最大相对厚度位置[m]）", 0.02),
                "mid_BOWA": first_design.get("mid_BOWA（叶中掠[m]）", 0.01),
                
                "tip_Angle_in": first_design.get("tip_Angle_in（叶顶进口金属角[°]）", 68),
                "tip_Angle_out": first_design.get("tip_Angle_out（叶顶出口金属角[°]）", 61),
                "tip_Chord": first_design.get("tip_Chord（叶顶弦长[m]）", 0.13),
                "tip_THmax_CH": first_design.get("tip_THmax_CH（叶顶最大相对厚度[%]）", 6.3),
                "tip_THmaxP": first_design.get("tip_THmaxP（叶顶最大相对厚度位置[-]）", 0.6),
                "tip_SWA": first_design.get("tip_SWA（叶顶最大相对厚度位置[m]）", -0.01),
                "tip_BOWA": first_design.get("tip_BOWA（叶顶掠[m]）", 0.02)
            }
        
        # 情况4：没有任何参数
        else:
            print(f"[VisualizationExpert] 缺少设计参数，询问用户")
            
            question = """❓ 3D可视化需要21维设计参数。

您可以：
1. **先生成设计** - 说「我需要设计一个叶片」
2. **提供已有参数** - 直接提供21维设计参数，例如「叶根弦长为0.6m，叶根进口角为57度...」

推荐您先生成设计，然后再进行可视化。"""
            
            state["waiting_for_user"] = True
            state["user_question"] = question
            state["pending_agent"] = AgentNames.VISUALIZATION_EXPERT
            state["messages"].append(AIMessage(content=question))
            
            return state
        
        print(f"[VisualizationExpert] 调用可视化工具...")
        
        viz_result = blade_visualization_tool.invoke(viz_params)
        
        if "error" in viz_result:
            state["error_message"] = viz_result["error"]
            state["final_result"] = f"❌ 可视化失败: {viz_result['error']}"
            state["messages"].append(AIMessage(content=state["final_result"]))
            state["next_action"] = EdgeConditions.TO_END
            return state
        
        print(f"[VisualizationExpert] 可视化完成")
        
        # 保存结果
        state["visualization_path"] = viz_result.get("image_path", "")
        state["visualization_json_path"] = viz_result.get("json_path", "")  # 新增：3D数据路径
        
        # ==================== 新增：存储任务结果 ====================
        store_task_result(state, "visualize")
        
        # 构建响应消息
        response_msg = "✅ **3D可视化已生成**\n\n"
        response_msg += f"📊 图片路径：{state['visualization_path']}"
        
        state["final_result"] = response_msg
        state["messages"].append(AIMessage(content=response_msg))
        
        log_collaboration(
            state,
            from_agent=AgentNames.VISUALIZATION_EXPERT,
            to_agent="user",
            action="返回可视化结果",
            details="可视化完成"
        )
        
        # ==================== 检查是否有后续任务（在添加消息之前检查，避免重复）====================
        if state.get("task_mode") == "sequential" and len(state.get("task_queue", [])) > 1:
            return _handle_next_task(state, "visualize", response_msg,
                                    image_path=state.get("visualization_path"),
                                    json_path=state.get("visualization_json_path"))
        
        # 非多任务序列，正常添加消息
        state["messages"].append(AIMessage(content=response_msg))
        state["next_action"] = EdgeConditions.TO_END
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[VisualizationExpert] 错误: {str(e)}")
        state["error_message"] = f"可视化专家执行失败: {str(e)}"
        state["final_result"] = f"❌ 可视化失败: {str(e)}"
        state["messages"].append(AIMessage(content=state["final_result"]))
        state["next_action"] = EdgeConditions.TO_END
    
    return state


# ==================== 闲聊专家节点 ====================

def chat_expert_node(state: MultiAgentState) -> MultiAgentState:
    """
    闲聊专家Agent - 处理无法识别的意图（参考单智能体）
    
    职责：
    1. 判断问题类型（功能询问/普通闲聊）
    2. 生成自然的对话回复
    3. 引导用户回到叶片设计主题
    """
    
    print(f"\n{'='*60}")
    print(f"[ChatExpert] 开始处理闲聊...")
    print(f"{'='*60}")
    
    user_input = state.get("user_input", "")
    
    try:
        llm = create_llm(temperature=0.3)  # 低温度，判断更准确
        
        # 第一步：判断问题类型
        print(f"[ChatExpert] 判断问题类型...")
        
        classification_prompt = f"""请判断用户的问题属于哪种类型。

用户问题：{user_input}

类型定义：
1. "功能询问" - 用户想了解系统的功能、身份、使用方法。例如：
   - "你是谁"
   - "你能做什么"
   - "怎么使用"
   - "有什么功能"
   - "介绍一下你自己"
   - "你好"（问候语，也视为功能询问）
   - "help"
   
2. "知识询问" - 用户询问与压气机/叶片设计相关的专业知识、原理、概念。例如：
   - "压气机在设计阶段都分为哪些步骤"
   - "叶片设计的原理是什么"
   - "什么是叶型参数"
   - "效率和流量有什么关系"
   - "压比是如何计算的"
   - "叶片有哪些类型"
   
3. "普通闲聊" - 用户问的是与压气机/叶片设计完全无关的其他话题。例如：
   - "今天天气怎么样"
   - "讲个笑话"
   - "1+1等于几"
   - "推荐一部电影"

请只回复一个词：
- 如果是功能询问，回复：功能询问
- 如果是知识询问，回复：知识询问
- 如果是普通闲聊，回复：普通闲聊"""

        classification_response = llm.invoke([HumanMessage(content=classification_prompt)])
        question_type = classification_response.content.strip()
        
        print(f"[ChatExpert] 问题类型: {question_type}")
        
        # 根据判断结果决定回复方式
        is_system_inquiry = "功能询问" in question_type
        is_knowledge_inquiry = "知识询问" in question_type
        
    except Exception as e:
        # LLM判断失败时，降级到关键词匹配
        print(f"[ChatExpert] LLM判断失败，使用关键词匹配: {e}")
        user_input_lower = user_input.lower().strip()
        system_inquiry_keywords = [
            "你是谁", "你是什么", "能做什么", "有什么功能",
            "怎么用", "帮助", "help", "你好", "您好", "hi", "hello"
        ]
        knowledge_inquiry_keywords = [
            "什么是", "什么叫", "原理", "步骤", "阶段", "如何计算", 
            "怎么理解", "有什么区别", "有哪些", "包括什么"
        ]
        is_system_inquiry = any(keyword in user_input_lower for keyword in system_inquiry_keywords)
        is_knowledge_inquiry = any(keyword in user_input_lower for keyword in knowledge_inquiry_keywords)
    
    if is_knowledge_inquiry:
        # 知识询问 - 调用 LLM 生成专业知识回答
        print(f"[ChatExpert] 类型：知识询问，生成专业知识回答")
        
        try:
            llm = create_llm(temperature=0.3)  # 低温度，回答更准确
            
            # 检查是否在多任务序列中
            task_queue = state.get("task_queue", [])
            task_mode = state.get("task_mode", "single")
            is_multi_task = task_mode == "sequential" and len(task_queue) > 1
            
            if is_multi_task:
                # 多任务序列中，需要从完整输入中提取知识问题部分
                knowledge_prompt = f"""你是一个航空发动机压气机叶片设计领域的专家。

用户的完整输入是："{user_input}"

这个输入包含多个请求（如设计、评估、知识询问等）。你现在只需要回答其中的**知识询问部分**。

请你：
1. 从用户输入中识别出知识询问/概念解释的部分
2. 只针对这部分进行专业、清晰的回答
3. 忽略其中的任务请求部分（如"帮我设计"、"评估一下"等）
4. 如果问题涉及多个知识点，可以分点说明
5. 使用通俗易懂的语言，必要时可以举例说明

请直接回答用户询问的知识问题："""
            else:
                # 单任务模式，直接回答
                knowledge_prompt = f"""你是一个航空发动机压气机叶片设计领域的专家。用户正在询问一个与压气机叶片相关的专业知识问题。

用户问题：{user_input}

请你：
1. 以专业、清晰的方式回答用户的问题
2. 如果问题涉及多个方面，可以分点说明
3. 使用通俗易懂的语言，必要时可以举例说明
4. 如果问题与本系统的功能相关，可以在最后简要提及系统可以帮助用户做什么
5. 保持回答的专业性和准确性

请直接回答用户的问题："""

            llm_response = llm.invoke([HumanMessage(content=knowledge_prompt)])
            response = llm_response.content
            
        except Exception as e:
            print(f"[ChatExpert] 知识回答生成失败: {e}")
            response = f"抱歉，在回答您的问题时遇到了一些问题。您问的是：{user_input}\n\n如果您需要进行叶片设计、性能评估或可视化，请直接告诉我您的具体需求。"
        
        state["final_result"] = response
        state["messages"].append(AIMessage(content=response))
        
        log_collaboration(
            state,
            from_agent=AgentNames.CHAT_EXPERT,
            to_agent="user",
            action="返回专业知识回答",
            details="知识询问"
        )
    
    elif is_system_inquiry:
        # 功能询问 - 返回系统功能介绍
        print(f"[ChatExpert] 类型：功能询问，返回系统介绍")
        
        response = """您好！我是航空发动机叶片设计助手。😊

我可以帮您：
1. 🔧 **设计叶片** - 告诉我流量、效率、压比等性能目标
   示例："设计一个流量15kg/s、效率0.85、压比1.5的叶片"

2. 📊 **评估性能** - 提供21维设计参数，我来评估性能
   示例："评估性能"

3. 🎨 **可视化** - 生成叶片的3D几何形状
   示例："显示叶片3D图"

**提示**：如果您提供的参数不完整，我会询问您补充信息。

请问您需要什么帮助？"""
        
        state["final_result"] = response
        state["messages"].append(AIMessage(content=response))
        
        log_collaboration(
            state,
            from_agent=AgentNames.CHAT_EXPERT,
            to_agent="user",
            action="返回功能介绍",
            details="功能询问"
        )
    
    else:
        # 普通闲聊 - 调用LLM生成自然回复并引导
        print(f"[ChatExpert] 类型：普通闲聊，生成自然回复")
        
        try:
            llm = create_llm(temperature=0.7)  # 稍高温度，更自然
            
            # 检查是否在多任务序列中
            task_queue = state.get("task_queue", [])
            task_mode = state.get("task_mode", "single")
            is_multi_task = task_mode == "sequential" and len(task_queue) > 1
            
            if is_multi_task:
                # 多任务序列中，从完整输入中提取闲聊部分
                chat_prompt = f"""你是一个航空发动机叶片设计助手，同时也是一个温暖、有见识的对话伙伴。

用户的完整输入是："{user_input}"

这个输入包含多个请求。你现在只需要回应其中的**闲聊/无关问题部分**（如天气、笑话、日常话题等）。

回复要求：
1. 识别出用户输入中与叶片设计无关的闲聊部分
2. 认真、真诚地回应用户的问题，让用户感到被理解和尊重
3. 回复可以适当展开，内容丰富一些，但不要过于冗长
4. 不要重复回应已经由其他专家处理的任务（如设计、评估）
5. 语气温和亲切，像朋友间的交流
6. 不要使用 emoji 表情符号

**排版要求（重要）**：
- 使用清晰的段落分隔，每个要点之间空一行
- 如果有多个方面，使用简洁的小标题或分点
- 避免把所有内容挤在一大段里

现在请回复用户的闲聊问题："""
            else:
                # 单任务模式，正常处理
                chat_prompt = f"""你是一个航空发动机叶片设计助手，同时也是一个温暖、有见识的对话伙伴。

用户问题：{user_input}

虽然这个问题与你的专业领域（叶片设计）没有直接关系，但作为一个友好的助手，你应该真诚地回应用户。

回复要求：
1. 首先认真对待用户的问题，给出有价值的回应（不要敷衍了事）
2. 回复内容可以适当丰富，让用户感到你是在真心交流
3. 根据话题类型灵活回应：
   - 如果是知识类问题（如数学、常识），尽量给出准确答案
   - 如果是需要实时信息的问题（如天气），坦诚说明你的局限，但可以提供一些相关建议
   - 如果是请求类问题（如讲笑话），尽力满足用户的需求
4. 在回复的最后，自然地过渡到你的专业领域，让用户知道你在叶片设计方面可以提供帮助
5. 整体语气要亲切自然，像一个有学识的朋友在聊天
6. 不要使用 emoji 表情符号

**排版要求（非常重要）**：
- 使用清晰的段落分隔，不同话题之间空一行
- 可以使用 Markdown 格式增强可读性
- 避免把所有内容挤在一大段里，让内容层次分明

示例格式：

---
用户："今天天气怎么样？"

回复：
关于天气，作为 AI 助手，我确实没办法获取实时的天气数据，这一点比较遗憾。

**建议**：你可以查看手机上的天气应用，或者直接搜索"当地天气"获取最新信息。

说起天气，其实航空发动机的性能也会受到环境条件的影响。如果你对叶片设计感兴趣，我可以帮你设计一款在特定工况下表现优异的叶片。

---
用户："讲个笑话"

回复：
好呀，我来讲一个工程师的笑话：

> 有一天，一个工程师、一个物理学家和一个数学家住在同一家酒店。
> 
> 半夜工程师的房间着火了，他看到火，拿起灭火器把火扑灭，然后继续睡觉。
> 
> 物理学家的房间也着火了，他计算了火势、距离和水量，精确地用最少的水灭了火。
> 
> 数学家的房间着火时，他看了看灭火器，说"解是存在的"，然后安心睡去了。

虽然讲笑话不是我的强项，但如果你想聊聊航空发动机叶片设计，那我可就是专业的了。

---
用户："1+1等于几"

回复：
**答案**：1 + 1 = 2

这是数学中最基础的加法运算。

**有趣的是**：在不同的数学体系中，这个问题可能有不同的答案——比如在二进制中，1 + 1 = 10。

虽然数学计算不是我的主业，但在叶片设计中确实会用到大量的数学计算，比如流体力学方程、几何参数优化等。如果你有叶片设计方面的需求，我很乐意帮忙。

---

现在请按照上述格式要求回复用户的问题："""

            llm_response = llm.invoke([HumanMessage(content=chat_prompt)])
            response = llm_response.content
            
            print(f"[ChatExpert] LLM回复生成成功")
            
        except Exception as e:
            # LLM调用失败时的降级回复
            print(f"[ChatExpert] LLM调用失败: {e}")
            response = f"""感谢你的提问。关于"{user_input}"这个话题，虽然不是我最擅长的领域，但我很高兴能和你交流。

我的专业领域是航空发动机叶片设计，在这方面我可以为你提供以下帮助：

- 根据性能目标（流量、效率、压比）设计叶片方案
- 评估现有设计的性能表现
- 生成叶片的三维可视化图像
- 优化设计方案以获得更好的性能

如果你对这些感兴趣，随时告诉我你的需求。"""
        
        state["final_result"] = response
        state["messages"].append(AIMessage(content=response))
        
        log_collaboration(
            state,
            from_agent=AgentNames.CHAT_EXPERT,
            to_agent="user",
            action="返回闲聊回复",
            details="普通闲聊"
        )
    
    # 保存结果到 task_results
    if "task_results" not in state:
        state["task_results"] = {}
    state["task_results"]["chat"] = {"response": response}
    
    # 存储结果并处理多任务序列
    store_task_result(state, "chat")
    
    # 检查是否在多任务序列中
    task_mode = state.get("task_mode", "single")
    
    if task_mode == "sequential":
        # 使用 _handle_next_task 统一处理多任务逻辑（和其他专家一致）
        return _handle_next_task(state, "chat", response)
    else:
        # 单任务模式，直接结束
        state["next_action"] = EdgeConditions.TO_END
        return state


# ==================== 任务调度器节点（新增）====================

def task_scheduler_node(state: MultiAgentState) -> MultiAgentState:
    """
    任务调度器 - 控制多任务序列的执行
    
    职责：
    1. 管理task_queue中的任务执行顺序
    2. 检查任务依赖（如评估需要设计结果）
    3. 自动传递上游任务结果
    4. 更新current_task_index
    """
    
    print(f"\n{'='*60}")
    print(f"[Scheduler] 任务调度器执行...")
    print(f"{'='*60}")
    
    task_queue = state.get("task_queue", [])
    current_index = state.get("current_task_index", 0)
    
    print(f"[Scheduler] 任务队列: {task_queue}, 当前索引: {current_index}")
    
    # 检查是否所有任务都已完成
    if current_index >= len(task_queue):
        print(f"[Scheduler] 所有任务已完成")
        
        # 汇总所有任务结果
        summary = "✅ **所有任务已完成**\n\n"
        
        results = state.get("task_results", {})
        if results.get("chat"):
            summary += "💬 **知识问答**: 已回答\n"
        if results.get("design"):
            summary += "📐 **设计结果**: 已生成21维参数\n"
        if results.get("evaluate"):
            eval_r = results.get("evaluate", {})
            summary += f"📊 **评估结果**: 性能已评估\n"
        if results.get("visualize"):
            summary += f"🎨 **可视化结果**: 图片已生成\n"
        if results.get("optimize"):
            summary += f"🔧 **优化结果**: 设计已优化\n"
        
        state["final_result"] = summary
        state["messages"].append(AIMessage(content=summary))
        state["next_action"] = EdgeConditions.TO_END
        
        return state
    
    # 获取当前任务
    current_task = task_queue[current_index]
    print(f"[Scheduler] 当前任务: {current_task} ({current_index + 1}/{len(task_queue)})")
    
    # 检查依赖：评估、可视化和优化需要设计结果
    if current_task in ["evaluate", "visualize", "optimize"]:
        has_design = (state.get("design_params") is not None or 
                     state.get("task_results", {}).get("design") is not None)
        
        if not has_design and "design" not in task_queue[:current_index]:
            # 没有设计结果，需要先设计
            print(f"[Scheduler] {current_task} 缺少设计结果，但队列中没有设计任务")
            
            task_name = {"evaluate": "性能评估", "visualize": "可视化", "optimize": "优化"}[current_task]
            question = f"⚠️ {task_name}需要叶片参数，但没有可用的设计结果。\n\n"
            question += "请先提供参数或执行设计任务。"
            
            state["waiting_for_user"] = True
            state["user_question"] = question
            state["messages"].append(AIMessage(content=question))
            
            return state
        
        # 如果有设计结果，且用户没有明确要求新任务，才自动使用已有结果
        if has_design and state.get("use_existing_result") is None:
            state["use_existing_result"] = True
            print(f"[Scheduler] 自动使用已有设计结果（用户未明确要求新参数）")
        elif state.get("use_existing_result") == False:
            print(f"[Scheduler] 用户要求新任务，不自动使用已有结果")
    
    # 优化任务还需要检查评估结果
    if current_task == "optimize":
        has_eval = (state.get("evaluation_results") is not None or 
                   state.get("task_results", {}).get("evaluate") is not None)
        
        # 如果没有评估结果但有设计结果，优化节点会自动评估，无需阻塞
        if not has_eval:
            print(f"[Scheduler] 优化任务将自动评估初始设计")
    
    # 路由到对应专家
    if current_task == "design":
        state["target_agent"] = AgentNames.DESIGN_EXPERT
        state["current_agent"] = AgentNames.DESIGN_EXPERT
        state["intent"] = "design"
        
    elif current_task == "evaluate":
        state["target_agent"] = AgentNames.EVALUATION_EXPERT
        state["current_agent"] = AgentNames.EVALUATION_EXPERT
        state["intent"] = "evaluate"
        
    elif current_task == "visualize":
        state["target_agent"] = AgentNames.VISUALIZATION_EXPERT
        state["current_agent"] = AgentNames.VISUALIZATION_EXPERT
        state["intent"] = "visualize"
    
    elif current_task == "optimize":
        state["target_agent"] = AgentNames.OPTIMIZATION_EXPERT
        state["current_agent"] = AgentNames.OPTIMIZATION_EXPERT
        state["intent"] = "optimize"
    
    elif current_task == "chat":
        state["target_agent"] = AgentNames.CHAT_EXPERT
        state["current_agent"] = AgentNames.CHAT_EXPERT
        state["intent"] = "chat"
        print(f"[Scheduler] 分派到闲聊专家处理知识询问/闲聊")
    
    else:
        print(f"[Scheduler] 未知任务类型: {current_task}，分派到闲聊专家")
        state["target_agent"] = AgentNames.CHAT_EXPERT
        state["current_agent"] = AgentNames.CHAT_EXPERT
        state["intent"] = "chat"
    
    log_collaboration(
        state,
        from_agent="scheduler",
        to_agent=state["target_agent"],
        action=f"调度任务 {current_index + 1}/{len(task_queue)}",
        details={"task": current_task}
    )
    
    return state


# ==================== 优化专家节点 ====================

def optimization_expert_node(state: MultiAgentState) -> MultiAgentState:
    """
    优化专家Agent - 负责优化叶片设计
    
    职责：
    1. 从用户输入中提取优化参数（优化目标、目标值、迭代次数）
    2. 检查是否有可用的设计和评估结果
    3. 如果缺少必要数据，询问用户或引导用户先执行其他任务
    4. 设置优化目标和约束
    5. 执行LLM驱动的进化优化
    6. 返回优化后的设计方案和性能对比
    
    参数提取方式与 Design Expert 保持一致：
    - 在本节点内部使用 LLM 提取优化参数
    - 对参数进行验证
    - 参数不完整时反问用户
    """
    
    print(f"\n{'='*70}")
    print(f"[OptimizationExpert] 开始优化设计...")
    print(f"{'='*70}")
    
    try:
        user_input = state.get("user_input", "")
        use_existing = state.get("use_existing_result")
        intent_type = state.get("intent_type", "")
        
        # 【关键修复】在多任务序列中，优先使用原始用户输入来提取优化参数
        # 因为在反问过程中，user_input 可能被用户的回答覆盖
        original_input = state.get("original_user_input")
        input_for_extraction = original_input if original_input else user_input
        
        if original_input:
            print(f"[OptimizationExpert] 使用原始用户输入提取参数: {original_input[:50]}...")
        else:
            print(f"[OptimizationExpert] 使用当前用户输入提取参数: {user_input[:50] if user_input else '(空)'}...")
        
        # ==================== 1. 初始化优化配置（类似 Design Expert 的参数累积）====================
        if state.get("optimization_config") is None:
            state["optimization_config"] = {}
        
        # ==================== 2. 从用户输入中提取优化参数【核心修改：与 Design Expert 保持一致】====================
        if input_for_extraction:
            print(f"[OptimizationExpert] 尝试从用户输入提取优化参数...")
            llm = create_llm()
            
            extraction_prompt = f"""从用户输入中提取叶片优化的配置参数，返回JSON格式。

用户输入：{input_for_extraction}

需要提取的参数：
1. primary_objective（优化目标）- 只能是以下之一：
   - "efficiency"：效率（关键词：效率、efficiency）
   - "flow"：流量（关键词：流量、flow、质量流量）
   - "pressure_ratio"：压比（关键词：压比、压力比、pressure ratio）
   
2. target_value（目标数值）- 如果用户指定了具体数值：
   - 效率：如果是百分比形式（如89%、90%），转换为小数（0.89、0.90）
   - 效率：如果已经是小数形式（如0.89），直接使用
   - 流量和压比：直接使用用户给的数值
   - 如果用户没有指定具体数值（只说"最大化"、"提高"等），设为null
   
3. target_type（目标类型）- 只能是以下之一：
   - "equal"：达到具体值（当用户指定了具体数值时）
   - "maximize"：最大化（当用户说"最大化"、"提高"、"提升"但没有具体数值时）
   
4. max_generations（迭代次数）- 整数：
   - 如果用户提到"迭代X次"、"X轮"、"X代"等，提取这个数值
   - 如果用户没有指定，设为null

返回JSON格式：
{{
    "primary_objective": "efficiency/flow/pressure_ratio" 或 null,
    "target_value": 数值 或 null,
    "target_type": "equal/maximize" 或 null,
    "max_generations": 整数 或 null
}}

只提取明确提到的参数。如果某个参数没提到，该字段设为null。

示例：
- "优化效率到89%" → {{"primary_objective": "efficiency", "target_value": 0.89, "target_type": "equal", "max_generations": null}}
- "最大化流量，迭代20次" → {{"primary_objective": "flow", "target_value": null, "target_type": "maximize", "max_generations": 20}}
- "提高压比" → {{"primary_objective": "pressure_ratio", "target_value": null, "target_type": "maximize", "max_generations": null}}
- "优化一下" → {{"primary_objective": null, "target_value": null, "target_type": null, "max_generations": null}}

**重要**：必须返回有效的JSON格式！"""

            try:
                response = llm.invoke([HumanMessage(content=extraction_prompt)])
                response_text = response.content.strip()
                print(f"[OptimizationExpert] LLM原始返回: {response_text[:200]}...")
                
                extracted = extract_json_from_text(response_text)
                
                if extracted:
                    # 后处理：效率值自动转换（如果>1，可能是百分比形式）
                    if "target_value" in extracted and extracted["target_value"] is not None:
                        if extracted.get("primary_objective") == "efficiency" and extracted["target_value"] > 1:
                            original_value = extracted["target_value"]
                            extracted["target_value"] = extracted["target_value"] / 100
                            print(f"[OptimizationExpert] 效率目标值自动转换: {original_value} → {extracted['target_value']}")
                    
                    print(f"[OptimizationExpert] 本次提取到参数: {extracted}")
                    
                    # 验证参数合理性
                    valid_params = {}
                    invalid_params = []
                    
                    # 验证 primary_objective
                    if extracted.get("primary_objective"):
                        if extracted["primary_objective"] in ["efficiency", "flow", "pressure_ratio"]:
                            valid_params["primary_objective"] = extracted["primary_objective"]
                        else:
                            invalid_params.append(f"优化目标 '{extracted['primary_objective']}' 无效，只能是 efficiency/flow/pressure_ratio")
                    
                    # 验证 target_value
                    if extracted.get("target_value") is not None:
                        tv = extracted["target_value"]
                        obj = extracted.get("primary_objective") or valid_params.get("primary_objective")
                        
                        if obj == "efficiency":
                            if 0.5 <= tv <= 1.0:
                                valid_params["target_value"] = tv
                                valid_params["target_type"] = "equal"
                            else:
                                invalid_params.append(f"效率目标值 {tv} 超出合理范围（0.5-1.0）")
                        elif obj == "flow":
                            if 5 <= tv <= 30:
                                valid_params["target_value"] = tv
                                valid_params["target_type"] = "equal"
                            else:
                                invalid_params.append(f"流量目标值 {tv} 超出合理范围（5-30 kg/s）")
                        elif obj == "pressure_ratio":
                            if 1.0 <= tv <= 3.0:
                                valid_params["target_value"] = tv
                                valid_params["target_type"] = "equal"
                            else:
                                invalid_params.append(f"压比目标值 {tv} 超出合理范围（1.0-3.0）")
                    
                    # 验证 target_type（如果没有 target_value 但指定了 maximize）
                    if extracted.get("target_type") == "maximize" and "target_value" not in valid_params:
                        valid_params["target_type"] = "maximize"
                    
                    # 验证 max_generations
                    if extracted.get("max_generations") is not None:
                        mg = extracted["max_generations"]
                        if isinstance(mg, (int, float)) and 1 <= mg <= 100:
                            valid_params["max_generations"] = int(mg)
                        else:
                            invalid_params.append(f"迭代次数 {mg} 超出合理范围（1-100）")
                    
                    # 如果有不合理的参数，立即反问
                    if invalid_params:
                        print(f"[OptimizationExpert] 检测到不合理参数，立即反问")
                        
                        # 先累积合理的参数（如果有）
                        if valid_params:
                            state["optimization_config"].update(valid_params)
                            print(f"[OptimizationExpert] 保留合理参数: {valid_params}")
                        
                        # 构建反问消息
                        error_msg = "⚠️ **检测到优化参数不合理：**\n\n"
                        error_msg += "\n".join([f"  • {err}" for err in invalid_params])
                        error_msg += "\n\n请重新提供合理的参数值。"
                        
                        # 显示已接受的合理参数
                        if valid_params:
                            error_msg += f"\n\n✅ **已接受的参数：**\n"
                            param_names = {
                                "primary_objective": "优化目标",
                                "target_value": "目标数值",
                                "target_type": "目标类型",
                                "max_generations": "迭代次数"
                            }
                            for key, value in valid_params.items():
                                error_msg += f"  • {param_names.get(key, key)}: {value}\n"
                        
                        state["waiting_for_user"] = True
                        state["user_question"] = error_msg
                        state["pending_agent"] = AgentNames.OPTIMIZATION_EXPERT
                        state["messages"].append(AIMessage(content=error_msg))
                        
                        return state
                    
                    # 所有参数都合理，累积参数
                    state["optimization_config"].update(valid_params)
                    print(f"[OptimizationExpert] 累积后参数: {state['optimization_config']}")
                else:
                    print(f"[OptimizationExpert] 未能从输入中提取参数")
            except Exception as e:
                print(f"[OptimizationExpert] 参数提取失败: {e}")
        
        # 从state中获取累积的优化配置
        opt_config = state["optimization_config"]
        print(f"[OptimizationExpert] 当前state中的累积参数: {opt_config}")
        
        # ==================== 3. 获取设计参数 ====================
        design_params = state.get("design_params")
        stored_design = state.get("task_results", {}).get("design")
        
        # 只有明确是"新任务"且不使用已有结果时，才清空历史数据
        # continue/use_existing 等意图应该使用历史数据
        if intent_type == "new" and use_existing is False:
            design_params = None
            print(f"[OptimizationExpert] 新任务，不使用历史数据")
        elif not design_params and stored_design:
            design_params = stored_design
            state["design_params"] = stored_design
            print(f"[OptimizationExpert] 从task_results加载设计结果")
        
        # 检查是否有设计结果
        if not design_params or "第1个设计结果" not in design_params:
            print(f"[OptimizationExpert] 缺少设计结果，询问用户")
            
            question = """❓ **优化需要已有的叶片设计方案。**

您可以：
1. 先执行叶片设计，例如：「设计一个流量15、效率0.85、压比1.5的叶片」
2. 或者提供完整的21维设计参数

优化将在获得设计方案后自动进行。"""
            
            state["waiting_for_user"] = True
            state["user_question"] = question
            state["pending_agent"] = AgentNames.OPTIMIZATION_EXPERT
            state["messages"].append(AIMessage(content=question))
            
            return state
        
        # ==================== 新增：识别用户选择要优化的方案（支持多方案）====================
        available_count = get_available_scheme_count(design_params)
        print(f"[OptimizationExpert] 可用方案数量: {available_count}")
        
        # 【关键改进】使用精确任务方案提取函数
        # 在多任务序列中，这样可以区分不同任务各自对应哪些方案
        llm = create_llm()
        selected_schemes = extract_task_specific_schemes(
            user_input=input_for_extraction,
            task_type="optimize",  # 指定当前任务类型
            available_count=available_count,
            llm=llm
        )
        
        # 如果有多个方案但用户未指定，询问
        if not selected_schemes:
            if available_count > 1:
                question = f"📋 **检测到 {available_count} 个设计方案**\n\n"
                question += "请指定要优化的方案：\n"
                question += "• 输入「优化第2个方案」优化单个方案\n"
                question += "• 输入「优化方案1和3」优化多个方案\n"
                question += "• 输入「优化所有方案」优化全部\n"
                question += "• 或直接回复「继续」优化第1个方案\n"
                
                state["waiting_for_user"] = True
                state["user_question"] = question
                state["pending_agent"] = AgentNames.OPTIMIZATION_EXPERT
                state["messages"].append(AIMessage(content=question))
                return state
            else:
                selected_schemes = [1]
        
        print(f"[OptimizationExpert] 选中优化方案: {selected_schemes}")
        state["_optimization_schemes"] = selected_schemes  # 保存选中的方案编号列表
        
        param_keys = [
            "root_Angle_in（叶根进口金属角[°]）", "root_Angle_out（叶根出口金属角[°]）",
            "root_Chord（叶根弦长[m]）", "root_THmax_CH（叶根最大相对厚度[%]）",
            "root_THmaxP（叶根最大相对厚度位置[-]）", "root_SWA（叶根最大相对厚度位置[m]）",
            "root_BOWA（叶根掠[m]）",
            "mid_Angle_in（叶中进口金属角[°]）", "mid_Angle_out（叶中出口金属角[°]）",
            "mid_Chord（叶中弦长[m]）", "mid_THmax_CH（叶中最大相对厚度[%]）",
            "mid_THmaxP（叶中最大相对厚度位置[-]）", "mid_SWA（叶中最大相对厚度位置[m]）",
            "mid_BOWA（叶中掠[m]）",
            "tip_Angle_in（叶顶进口金属角[°]）", "tip_Angle_out（叶顶出口金属角[°]）",
            "tip_Chord（叶顶弦长[m]）", "tip_THmax_CH（叶顶最大相对厚度[%]）",
            "tip_THmaxP（叶顶最大相对厚度位置[-]）", "tip_SWA（叶顶最大相对厚度位置[m]）",
            "tip_BOWA（叶顶掠[m]）"
        ]
        
        # ==================== 验证所有选中方案是否存在 ====================
        valid_schemes = []
        for scheme_idx in selected_schemes:
            scheme_design = design_params.get(f"第{scheme_idx}个设计结果", {})
            if scheme_design:
                valid_schemes.append(scheme_idx)
            else:
                print(f"[OptimizationExpert] 警告：未找到方案{scheme_idx}的设计数据，跳过")
        
        if not valid_schemes:
            state["error_message"] = "未找到任何有效的设计方案"
            state["final_result"] = "❌ 优化失败: 未找到任何有效的设计方案"
            state["messages"].append(AIMessage(content=state["final_result"]))
            state["next_action"] = EdgeConditions.TO_END
            return state
        
        # 导入进度回调函数
        from Tools.blade_optimization import _report_optimization_progress
        from langgraph_tools import blade_performance_evaluation_tool
        
        # 获取任务结果回调（用于推送每个方案的优化结果）
        from langgraph_multi_agent_workflow import get_task_result_callback
        task_result_callback = get_task_result_callback()
        
        # ==================== 4. 构建优化配置（使用本节点提取的参数）====================
        primary_objective = opt_config.get("primary_objective", "efficiency")  # 默认优化效率
        target_value = opt_config.get("target_value")  # 具体目标值
        target_type = opt_config.get("target_type", "maximize")  # maximize 或 equal
        max_generations = opt_config.get("max_generations") or 10  # 默认10次
        
        # 指标名称映射
        metric_names = {"flow": "流量", "efficiency": "效率", "pressure_ratio": "压比"}
        objective_name = metric_names.get(primary_objective, primary_objective)
        
        print(f"[OptimizationExpert] 优化目标: {objective_name}")
        print(f"[OptimizationExpert] 迭代次数: {max_generations}")
        print(f"[OptimizationExpert] 待优化方案数: {len(valid_schemes)}")
        
        # ==================== 5. 多方案优化循环 ====================
        all_optimization_results = []
        all_response_msgs = []
        
        for scheme_idx_in_loop, scheme_idx in enumerate(valid_schemes):
            print(f"\n{'='*70}")
            print(f"[OptimizationExpert] 开始优化方案 {scheme_idx} ({scheme_idx_in_loop + 1}/{len(valid_schemes)})")
            print(f"{'='*70}")
            
            # ✅ 发送进度重置信号（每个方案优化前重置进度条）
            _report_optimization_progress(
                'optimization_reset', 
                0, 
                f'开始优化方案 {scheme_idx} ({scheme_idx_in_loop + 1}/{len(valid_schemes)})',
                scheme_index=scheme_idx,
                current_scheme=scheme_idx_in_loop + 1,
                total_schemes=len(valid_schemes)
            )
            
            # 获取当前方案的设计参数
            scheme_design = design_params.get(f"第{scheme_idx}个设计结果", {})
            design_21d = [scheme_design.get(key, 0) for key in param_keys]
            
            # 评估当前方案的初始性能
            print(f"[OptimizationExpert] 评估方案{scheme_idx}的初始性能...")
            eval_result = blade_performance_evaluation_tool.invoke({
                "design_params": [design_21d]
            })
            
            if "error" in eval_result:
                print(f"[OptimizationExpert] 方案{scheme_idx}评估失败: {eval_result['error']}")
                all_optimization_results.append({
                    "scheme_idx": scheme_idx,
                    "success": False,
                    "error": eval_result["error"]
                })
                continue
            
            first_eval = eval_result.get("第1个设计性能结果", {})
            initial_performance = {
                "flow": first_eval.get("mass flow rate（预测流量[kg/s]）", 15.0),
                "efficiency": first_eval.get("isentropic efficiency（预测等熵效率[-]）", 0.85),
                "pressure_ratio": first_eval.get("total pressure ratio（预测总压比[-]）", 1.5)
            }
            
            print(f"[OptimizationExpert] 方案{scheme_idx}初始性能: 流量={initial_performance['flow']:.3f}, 效率={initial_performance['efficiency']:.4f}, 压比={initial_performance['pressure_ratio']:.3f}")
            
            # 构建约束配置
            constraints = {}
            if target_type == "equal" and target_value is not None:
                constraints[primary_objective] = {"type": "equal", "value": target_value}
                other_metrics = {"flow", "efficiency", "pressure_ratio"} - {primary_objective}
                for metric in other_metrics:
                    constraints[metric] = {"type": "min", "value": initial_performance[metric] * 0.95}
            else:
                if primary_objective == "efficiency":
                    constraints["flow"] = {"type": "min", "value": initial_performance["flow"] * 0.98}
                    constraints["pressure_ratio"] = {"type": "min", "value": initial_performance["pressure_ratio"] * 0.98}
                elif primary_objective == "flow":
                    constraints["efficiency"] = {"type": "min", "value": initial_performance["efficiency"] * 0.98}
                    constraints["pressure_ratio"] = {"type": "min", "value": initial_performance["pressure_ratio"] * 0.98}
                elif primary_objective == "pressure_ratio":
                    constraints["flow"] = {"type": "min", "value": initial_performance["flow"] * 0.98}
                    constraints["efficiency"] = {"type": "min", "value": initial_performance["efficiency"] * 0.98}
            
            optimization_config = {
                "primary_objective": primary_objective,
                "constraints": constraints,
                "target_value": target_value,
                "target_type": target_type
            }
            
            # 执行优化
            print(f"[OptimizationExpert] 开始方案{scheme_idx}的LLM驱动优化...")
            
            opt_result = blade_optimization_tool.invoke({
                "initial_design": design_21d,
                "initial_performance": initial_performance,
                "optimization_config": optimization_config,
                "max_generations": max_generations,
                "population_size": 5
            })
            
            if not opt_result.get("success", False):
                print(f"[OptimizationExpert] 方案{scheme_idx}优化失败: {opt_result.get('error', '未知错误')}")
                all_optimization_results.append({
                    "scheme_idx": scheme_idx,
                    "success": False,
                    "error": opt_result.get("error", "优化失败")
                })
                continue
            
            # 保存结果
            opt_result["scheme_idx"] = scheme_idx
            all_optimization_results.append(opt_result)
            
            # ==================== 构建单个方案的结果消息 ====================
            improvement = opt_result.get("improvement", {})
            opt_perf = opt_result.get("optimized_performance", {})
            optimized_design = opt_result.get("optimized_design", [])
            
            scheme_msg = f"✅ **方案{scheme_idx}优化已完成** ({scheme_idx_in_loop + 1}/{len(valid_schemes)})\n\n"
            scheme_msg += f"**📈 优化进程**: 共进行了 {opt_result.get('total_generations', 0)} 代迭代（设定: {max_generations}代）\n\n"
            
            if target_type == "equal" and target_value is not None:
                scheme_msg += f"**🎯 优化目标**: {objective_name} → {target_value}\n\n"
            else:
                scheme_msg += f"**🎯 优化目标**: 最大化{objective_name}\n\n"
            
            scheme_msg += "**📊 性能对比：**\n"
            scheme_msg += "| 指标 | 优化前 | 优化后 | 变化 |\n"
            scheme_msg += "| --- | --- | --- | --- |\n"
            for metric in ["flow", "efficiency", "pressure_ratio"]:
                if metric in improvement:
                    imp = improvement[metric]
                    metric_name = {"flow": "流量", "efficiency": "效率", "pressure_ratio": "压比"}[metric]
                    scheme_msg += f"| {metric_name} | {imp['before']:.4f} | {imp['after']:.4f} | {imp['percent']:+.2f}% |\n"
            
            if optimized_design and len(optimized_design) == 21:
                scheme_msg += "\n**🆕 优化后的21维参数：**\n"
                scheme_msg += f"`{[round(v, 4) for v in optimized_design]}`\n"
            
            scheme_msg += f"\n**💡 结果说明**: {opt_result.get('message', '')}"
            
            all_response_msgs.append(scheme_msg)
            
            # ✅ 立即推送当前方案的优化结果到前端
            if task_result_callback:
                try:
                    task_result_callback({
                        'stage': 'optimization_scheme_complete',
                        'task': 'optimize',
                        'result': scheme_msg,
                        'scheme_index': scheme_idx,
                        'current_scheme': scheme_idx_in_loop + 1,
                        'total_schemes': len(valid_schemes),
                        'progress': (scheme_idx_in_loop + 1) / len(valid_schemes) * 100
                    })
                    print(f"[OptimizationExpert] ✅ 已推送方案{scheme_idx}的优化结果到前端")
                except Exception as e:
                    print(f"[OptimizationExpert] 推送结果失败: {e}")
            
            print(f"[OptimizationExpert] ✅ 方案{scheme_idx}优化完成")
        
        # ==================== 6. 汇总所有结果 ====================
        success_count = sum(1 for r in all_optimization_results if r.get("success", False))
        
        # 保存最后一个成功优化的结果到state
        if all_optimization_results:
            last_success = next((r for r in reversed(all_optimization_results) if r.get("success", False)), None)
            if last_success:
                state["optimization_result"] = last_success
                store_task_result(state, "optimize")
                
                # 同步最后一个优化结果到design_params
                optimized_design = last_success.get("optimized_design", [])
                if optimized_design and len(optimized_design) == 21:
                    optimized_design_dict = {
                        "第1个设计结果": {key: val for key, val in zip(param_keys, optimized_design)},
                        "_source": "optimization",
                        "_optimized_from_scheme": last_success.get("scheme_idx", 1)
                    }
                    state["design_params"] = optimized_design_dict
                    if "task_results" not in state or state["task_results"] is None:
                        state["task_results"] = {}
                    state["task_results"]["design"] = optimized_design_dict
        
        # 构建最终响应消息
        if len(valid_schemes) == 1:
            # 单方案优化：直接使用方案消息
            response_msg = all_response_msgs[0] if all_response_msgs else "❌ 优化失败"
        else:
            # 多方案优化：汇总消息
            response_msg = f"✅ **批量优化已完成**\n\n"
            response_msg += f"**📊 优化概览**: 共优化 {len(valid_schemes)} 个方案，成功 {success_count} 个\n\n"
            response_msg += "---\n\n"
            response_msg += "\n\n---\n\n".join(all_response_msgs)
        
        state["final_result"] = response_msg
        
        # 日志记录
        log_collaboration(
            state,
            from_agent=AgentNames.OPTIMIZATION_EXPERT,
            to_agent="user",
            action="返回优化结果",
            details=f"优化了 {success_count}/{len(valid_schemes)} 个方案"
        )
        
        # ==================== 7. 检查是否有后续任务 ====================
        if state.get("task_mode") == "sequential" and len(state.get("task_queue", [])) > 1:
            return _handle_next_task(state, "optimize", response_msg)
        
        # 非多任务序列，正常添加消息
        state["messages"].append(AIMessage(content=response_msg))
        state["next_action"] = EdgeConditions.TO_END
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[OptimizationExpert] 错误: {str(e)}")
        state["error_message"] = f"优化专家执行失败: {str(e)}"
        state["final_result"] = f"❌ 优化失败: {str(e)}"
        state["messages"].append(AIMessage(content=state["final_result"]))
        state["next_action"] = EdgeConditions.TO_END
    
    return state


# ==================== 错误处理节点 ====================

def error_handler_node(state: MultiAgentState) -> MultiAgentState:
    """错误处理节点"""
    
    print(f"\n{'='*60}")
    print(f"[ErrorHandler] 处理错误...")
    print(f"{'='*60}")
    
    error_msg = state.get("error_message", "未知错误")
    
    state["final_result"] = f"❌ 执行出错：{error_msg}"
    state["messages"].append(AIMessage(content=state["final_result"]))
    state["next_action"] = EdgeConditions.TO_END
    
    return state

