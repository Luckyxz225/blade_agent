"""
LangGraph 工作流节点定义
========================

修改目的：
1. 将业务逻辑拆分为独立的节点函数
2. 每个节点负责单一职责，便于测试和维护
3. 通过状态传递实现节点间的数据流转

为什么需要节点化：
- 原有架构：LLM自由决策工具调用顺序，逻辑不透明
- 优化后：明确的节点 → 边 → 节点流程，可控可追踪
- 关键优势：支持条件分支、循环、并行执行
"""

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph_config import DesignState, NodeNames, EdgeConditions
from langgraph_tools import (
    blade_design_tool,
    blade_performance_evaluation_tool,
    blade_visualization_tool
)
import json
import re


def create_llm(model_name: str = None, temperature: float = 0.5):
    """
    创建LLM实例
    
    **修改说明**：
    - 使用LangChain标准LLM接口
    - 支持切换不同模型（OpenAI、硅基流动等）
    - 自动从环境变量读取配置
    """
    import os
    
    # 从环境变量读取配置
    model = model_name or os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo')
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_API_BASE') or os.getenv('OPENAI_BASE_URL')
    
    # 构建参数
    kwargs = {
        'model': model,
        'temperature': temperature
    }
    
    # 只在有配置时传入（避免覆盖默认值）
    if api_key:
        kwargs['api_key'] = api_key
    if base_url:
        kwargs['base_url'] = base_url
    
    return ChatOpenAI(**kwargs)


# ==================== 节点1: 意图识别 ====================
def intent_recognition_node(state: DesignState) -> DesignState:
    """
    意图识别节点
    
    **功能**：分析用户输入，识别意图类型并提取参数
    
    **为什么需要这个节点**：
    - 原有系统：LLM直接决策调用哪个工具，可能遗漏或错误理解
    - 优化后：专门的节点负责意图理解，确保后续流程正确
    
    **意图类型**：
    - design: 用户想要生成新设计（需要性能目标）
    - evaluate: 用户想要评估已有设计（需要21维参数）
    - visualize: 用户想要可视化（需要参数或使用默认值）
    - optimize: 用户想要迭代优化（需要目标和约束）
    
    **输入状态**：
    - user_input: 用户原始输入
    
    **输出状态**：
    - intent: 识别的意图
    - performance_targets: 提取的性能目标（如果有）
    - design_params: 提取的设计参数（如果有）
    """
    
    user_input = state["user_input"]
    
    # 构建意图识别提示词
    intent_prompt = f"""你是一个航空发动机设计专家，负责理解用户需求并识别意图。

用户输入：{user_input}

请分析用户意图，从以下类型中选择一个：
1. design - 用户想要生成新的叶片设计（通常包含性能目标：流量、效率、压比）
2. evaluate - 用户想要评估已有设计的性能（通常提供21维设计参数）
3. visualize - 用户想要查看叶片的3D几何形状
4. optimize - 用户想要优化设计（需要迭代改进）
5. unknown - 无法识别的意图（如闲聊、问候等）

同时，尽可能从用户输入中提取以下参数：
- 流量 (flow_rate): kg/s
- 效率 (efficiency): 0-1之间的小数
- 压比 (pressure_ratio): >1的数值
- 21维设计参数（如果用户提供了）

**重要**：即使用户只提供了部分参数（如只说"流量15kg/s"），也请提取已知参数，其他设为null。系统会自动补充默认值。

请以JSON格式返回：
{{
    "intent": "design/evaluate/visualize/optimize/unknown",
    "confidence": 0.0-1.0,
    "performance_targets": {{
        "flow_rate": 数值或null,
        "efficiency": 数值或null,
        "pressure_ratio": 数值或null
    }},
    "design_params": null或参数字典,
    "reasoning": "你的判断理由"
}}
"""
    
    try:
        print(f"[意图识别] 开始调用LLM...")
        llm = create_llm()
        response = llm.invoke([HumanMessage(content=intent_prompt)])
        print(f"[意图识别] LLM响应成功")
        
        # 解析LLM返回的JSON
        response_text = response.content
        
        # 提取JSON部分（LLM可能返回带解释的内容）
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # 如果解析失败，使用默认值
            result = {
                "intent": "unknown",
                "confidence": 0.0,
                "performance_targets": None,
                "design_params": None,
                "reasoning": "无法解析用户意图"
            }
        
        # 更新状态
        state["intent"] = result["intent"]
        
        # 智能填充缺失的性能参数（使用典型值）
        if result["performance_targets"]:
            targets = result["performance_targets"]
            state["performance_targets"] = {
                "flow_rate": targets.get("flow_rate") or 15.2,
                "efficiency": targets.get("efficiency") or 0.83,
                "pressure_ratio": targets.get("pressure_ratio") or 1.55
            }
        else:
            state["performance_targets"] = result["performance_targets"]
        
        state["design_params"] = result["design_params"]
        
        # 添加消息到历史
        state["messages"].append(
            AIMessage(content=f"我理解您的需求是：{result['reasoning']}")
        )
        
    except Exception as e:
        state["intent"] = "unknown"
        state["error_message"] = f"意图识别失败: {str(e)}"
    
    return state


# ==================== 节点2: 设计生成 ====================
def design_generation_node(state: DesignState) -> DesignState:
    """
    设计生成节点
    
    **功能**：根据性能目标调用blade_design工具生成叶片设计
    
    **为什么独立成节点**：
    - 设计生成是核心功能，需要专门处理
    - 可以在此节点添加前置验证、后置检查逻辑
    - 便于记录设计历史和版本管理
    
    **输入状态**：
    - performance_targets: 性能目标字典
    
    **输出状态**：
    - design_params: 生成的21维设计参数
    """
    
    performance_targets = state.get("performance_targets")
    
    if not performance_targets:
        state["error_message"] = "缺少性能目标参数"
        state["next_action"] = EdgeConditions.TO_ERROR
        return state
    
    try:
        # 调用设计工具
        print(f"[设计生成] 开始调用blade_design_tool...")
        print(f"[设计生成] 参数: flow_rate={performance_targets.get('flow_rate')}, "
              f"efficiency={performance_targets.get('efficiency')}, "
              f"pressure_ratio={performance_targets.get('pressure_ratio')}")
        design_result = blade_design_tool.invoke({
            "flow_rate": performance_targets.get("flow_rate", 15.0),
            "efficiency": performance_targets.get("efficiency", 0.85),
            "pressure_ratio": performance_targets.get("pressure_ratio", 1.5),
            "top_k": 1
        })
        
        print(f"[设计生成] blade_design_tool调用完成")
        
        # 检查是否有错误
        if "error" in design_result:
            print(f"[设计生成] 工具返回错误: {design_result['error']}")
            state["error_message"] = design_result["error"]
            state["next_action"] = EdgeConditions.TO_ERROR
            return state
        
        print(f"[设计生成] 设计生成成功，开始保存结果...")
        # 保存设计结果
        state["design_params"] = design_result
        
        # 添加成功消息
        state["messages"].append(
            AIMessage(content=f"设计生成完成！生成了 {len(design_result)} 个设计方案。")
        )
        
        state["next_action"] = EdgeConditions.TO_END
        
    except Exception as e:
        state["error_message"] = f"设计生成失败: {str(e)}"
        state["next_action"] = EdgeConditions.TO_ERROR
    
    return state


# ==================== 节点3: 性能评估 ====================
def performance_evaluation_node(state: DesignState) -> DesignState:
    """
    性能评估节点
    
    **功能**：评估给定设计参数的气动性能
    
    **为什么独立成节点**：
    - 评估是独立的分析任务
    - 可能需要多次调用（批量评估多个设计）
    - 便于添加性能对比和分析逻辑
    
    **输入状态**：
    - design_params: 21维设计参数
    
    **输出状态**：
    - evaluation_results: 性能评估结果
    """
    
    design_params = state.get("design_params")
    
    if not design_params:
        state["error_message"] = "缺少设计参数"
        state["next_action"] = EdgeConditions.TO_ERROR
        return state
    
    try:
        # 提取21维参数（需要根据实际数据结构调整）
        # 假设design_params是字典格式，需要转换为列表
        params_list = extract_21d_params(design_params)
        
        # 调用评估工具
        eval_result = blade_performance_evaluation_tool.invoke({
            "design_params": params_list
        })
        
        # 检查错误
        if "error" in eval_result:
            state["error_message"] = eval_result["error"]
            state["next_action"] = EdgeConditions.TO_ERROR
            return state
        
        # 保存评估结果
        state["evaluation_results"] = eval_result
        
        # 添加成功消息
        state["messages"].append(
            AIMessage(content="性能评估完成！")
        )
        
        state["next_action"] = EdgeConditions.TO_END
        
    except Exception as e:
        state["error_message"] = f"性能评估失败: {str(e)}"
        state["next_action"] = EdgeConditions.TO_ERROR
    
    return state


# ==================== 节点4: 可视化 ====================
def visualization_node(state: DesignState) -> DesignState:
    """
    可视化节点
    
    **功能**：生成叶片3D几何可视化图
    
    **为什么独立成节点**：
    - 可视化是耗时操作，独立执行便于异步处理
    - 可以支持多种可视化类型（3D、剖面图等）
    - 便于缓存和复用图像
    
    **输入状态**：
    - design_params: 21维设计参数（可选，使用默认值）
    
    **输出状态**：
    - visualization_path: 生成的图像路径
    """
    
    design_params = state.get("design_params", {})
    
    try:
        # 提取可视化所需的参数
        viz_params = extract_visualization_params(design_params)
        
        # 调用可视化工具
        viz_result = blade_visualization_tool.invoke(viz_params)
        
        # 检查错误
        if "error" in viz_result:
            state["error_message"] = viz_result["error"]
            state["next_action"] = EdgeConditions.TO_ERROR
            return state
        
        # 保存可视化路径
        state["visualization_path"] = viz_result.get("image_path", "")
        
        # 添加成功消息
        state["messages"].append(
            AIMessage(content=f"可视化完成！图像路径: {viz_result.get('image_path')}")
        )
        
        state["next_action"] = EdgeConditions.TO_END
        
    except Exception as e:
        state["error_message"] = f"可视化失败: {str(e)}"
        state["next_action"] = EdgeConditions.TO_ERROR
    
    return state


# ==================== 节点5: 优化循环 ====================
def optimization_loop_node(state: DesignState) -> DesignState:
    """
    优化循环节点
    
    **功能**：迭代优化设计，直到满足性能目标
    
    **为什么需要这个节点**：
    - 这是LangGraph的核心优势：支持循环流程
    - 原有系统很难实现自动化迭代优化
    - 优化后：可以设置收敛条件、最大迭代次数等
    
    **优化流程**：
    1. 生成初始设计
    2. 评估性能
    3. 判断是否满足目标
    4. 如果不满足，调整参数并重新设计
    5. 重复直到满足或达到最大迭代次数
    
    **输入状态**：
    - performance_targets: 目标性能
    - iteration_count: 当前迭代次数
    - max_iterations: 最大迭代次数
    
    **输出状态**：
    - design_params: 优化后的设计
    - optimization_history: 优化历史
    - next_action: continue（继续优化）或 finish（完成）
    """
    
    performance_targets = state.get("performance_targets")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    
    if iteration_count >= max_iterations:
        state["messages"].append(
            AIMessage(content=f"已达到最大迭代次数 {max_iterations}，优化结束。")
        )
        state["next_action"] = EdgeConditions.FINISH_OPTIMIZATION
        return state
    
    try:
        # 1. 生成设计
        design_result = blade_design_tool.invoke({
            "flow_rate": performance_targets.get("flow_rate", 15.0),
            "efficiency": performance_targets.get("efficiency", 0.85),
            "pressure_ratio": performance_targets.get("pressure_ratio", 1.5),
            "top_k": 1
        })
        
        # 2. 评估性能
        params_list = extract_21d_params(design_result)
        eval_result = blade_performance_evaluation_tool.invoke({
            "design_params": params_list
        })
        
        # 3. 判断是否满足目标
        if check_performance_meets_target(eval_result, performance_targets):
            state["design_params"] = design_result
            state["evaluation_results"] = eval_result
            state["messages"].append(
                AIMessage(content=f"优化成功！在第 {iteration_count + 1} 次迭代找到满足要求的设计。")
            )
            state["next_action"] = EdgeConditions.FINISH_OPTIMIZATION
        else:
            # 4. 继续优化
            state["iteration_count"] = iteration_count + 1
            state["optimization_history"].append({
                "iteration": iteration_count + 1,
                "design": design_result,
                "performance": eval_result
            })
            state["next_action"] = EdgeConditions.CONTINUE_OPTIMIZATION
        
    except Exception as e:
        state["error_message"] = f"优化失败: {str(e)}"
        state["next_action"] = EdgeConditions.TO_ERROR
    
    return state


# ==================== 节点6: 结果整合 ====================
def result_synthesis_node(state: DesignState) -> DesignState:
    """
    结果整合节点
    
    **功能**：整合所有执行结果，生成用户友好的输出
    
    **为什么需要这个节点**：
    - 各个节点返回的是原始数据
    - 需要整合为结构化、易读的最终报告
    - 可以添加专业分析和建议
    
    **输入状态**：
    - 所有执行节点的结果
    
    **输出状态**：
    - messages: 包含完整报告的消息
    """
    
    print(f"[结果整合] 开始整合结果...")
    
    intent = state.get("intent", "unknown")
    design_params = state.get("design_params")
    eval_results = state.get("evaluation_results")
    viz_path = state.get("visualization_path")
    
    print(f"[结果整合] intent={intent}, 有设计参数={bool(design_params)}, "
          f"有评估结果={bool(eval_results)}, 有可视化={bool(viz_path)}")
    
    # 构建结构化报告（不依赖LLM，避免超时）
    report_parts = []
    
    if intent == "design" and design_params:
        report_parts.append("✅ **叶片设计已完成**\n")
        report_parts.append(f"生成了 {len(design_params)} 组设计参数\n")
        
        # 显示第一组设计的关键参数
        if "第1个设计结果" in design_params:
            first_design = design_params["第1个设计结果"]
            report_parts.append("\n**设计参数摘要：**")
            report_parts.append(f"- 叶根进口角：{first_design.get('root_Angle_in（叶根进口金属角[°]）', 'N/A')}°")
            report_parts.append(f"- 叶根弦长：{first_design.get('root_Chord（叶根弦长[m]）', 'N/A')} m")
            report_parts.append(f"- 叶尖进口角：{first_design.get('tip_Angle_in（叶尖进口金属角[°]）', 'N/A')}°")
    
    elif intent == "evaluate" and eval_results:
        report_parts.append("✅ **性能评估已完成**\n")
        report_parts.append("\n**评估结果：**")
        report_parts.append(f"- 流量：{eval_results.get('flow_rate', 'N/A')} kg/s")
        report_parts.append(f"- 效率：{eval_results.get('efficiency', 'N/A')}")
        report_parts.append(f"- 压比：{eval_results.get('pressure_ratio', 'N/A')}")
    
    elif intent == "visualize" and viz_path:
        report_parts.append("✅ **3D可视化已生成**\n")
        report_parts.append(f"\n📊 可视化路径：{viz_path}")
    
    # 如果有数据，可以尝试调用LLM生成更详细的分析（带超时）
    if design_params or eval_results:
        try:
            llm = create_llm(temperature=0.3)
            
            synthesis_prompt = f"""简要总结以下航发叶片设计结果（50字以内）：

意图：{intent}
设计参数：{'有' if design_params else '无'}
评估结果：{'有' if eval_results else '无'}

只需要一句话总结。"""
            
            response = llm.invoke([HumanMessage(content=synthesis_prompt)])
            report_parts.insert(0, f"**AI分析：** {response.content}\n")
        except Exception as e:
            # LLM调用失败不影响整体结果
            pass
    
    # 合并报告
    final_report = "\n".join(report_parts) if report_parts else "任务已完成。"
    
    print(f"[结果整合] 报告生成完成，长度={len(final_report)}字符")
    
    state["messages"].append(AIMessage(content=final_report))
    
    print(f"[结果整合] 结果整合完成")
    
    return state


# ==================== 节点7: 错误处理 ====================
def error_handler_node(state: DesignState) -> DesignState:
    """
    错误处理节点
    
    **功能**：统一处理所有节点的错误
    
    **为什么需要这个节点**：
    - 集中式错误处理，便于日志记录和监控
    - 可以提供用户友好的错误提示
    - 支持错误恢复和重试机制
    - 处理闲聊和意图不明的情况
    """
    
    error_message = state.get("error_message", "")
    intent = state.get("intent", "unknown")
    user_input = state.get("user_input", "")
    
    # 特殊处理：如果是意图不明（可能是闲聊）
    if intent == "unknown" and not error_message:
        # 用LLM智能判断用户问题类型
        print(f"[Unknown处理] 用户输入: {user_input}")
        
        try:
            llm = create_llm(temperature=0.3)  # 低温度，判断更准确
            
            # 让LLM判断问题类型
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
   
2. "普通闲聊" - 用户问的是与系统功能无关的其他话题。例如：
   - "今天天气怎么样"
   - "讲个笑话"
   - "1+1等于几"
   - "推荐一部电影"

请只回复一个词：
- 如果是功能询问，回复：功能询问
- 如果是普通闲聊，回复：普通闲聊"""

            classification_response = llm.invoke([HumanMessage(content=classification_prompt)])
            question_type = classification_response.content.strip()
            
            print(f"[Unknown处理] LLM判断结果: {question_type}")
            
            # 根据判断结果决定回复方式
            is_system_inquiry = "功能询问" in question_type
            
        except Exception as e:
            # LLM判断失败时，降级到关键词匹配
            print(f"[Unknown处理] LLM判断失败，降级到关键词匹配: {e}")
            user_input_lower = user_input.lower().strip()
            system_inquiry_keywords = [
                "你是谁", "你是什么", "能做什么", "有什么功能",
                "怎么用", "帮助", "help", "你好", "您好", "hi", "hello"
            ]
            is_system_inquiry = any(keyword in user_input_lower for keyword in system_inquiry_keywords)
        
        if is_system_inquiry:
            # 返回系统功能介绍
            response = """您好！我是航空发动机叶片设计助手。😊

我可以帮您：
1. 🔧 **设计叶片** - 告诉我流量、效率、压比等性能目标
   示例："设计一个流量15kg/s、效率0.85、压比1.5的叶片"

2. 📊 **评估性能** - 提供21维设计参数，我来评估性能

3. 🎨 **可视化** - 生成叶片的3D几何形状

4. 🔄 **优化设计** - 迭代改进现有设计

请问您需要什么帮助？"""
        else:
            # 对于普通闲聊，调用LLM生成自然回复
            print(f"[闲聊模式] 用户输入: {user_input}")
            try:
                llm = create_llm(temperature=0.7)  # 稍高温度，更自然
                
                # 构建prompt，让LLM知道自己是叶片设计助手
                chat_prompt = f"""你是一个航空发动机叶片设计助手。用户问了一个与叶片设计无关的问题。

用户问题：{user_input}

请你：
1. 简短、自然地回应用户的问题（1-2句话）
2. 然后礼貌地提醒用户你的主要功能是叶片设计
3. 语气要友好、自然，不要生硬

示例风格：
- 用户："今天天气怎么样？"
  回复："我这边看不到实时天气呢😅 不过我对叶片设计很在行！如果您有叶片设计的需求，随时告诉我。"

- 用户："讲个笑话"
  回复："哈哈，我对笑话不太擅长，但我对叶片设计很擅长😄 需要帮您设计叶片或评估性能吗？"

现在请回复用户的问题："""

                llm_response = llm.invoke([HumanMessage(content=chat_prompt)])
                response = llm_response.content
                print(f"[闲聊模式] LLM回复生成成功")
                
            except Exception as e:
                # LLM调用失败时的降级回复
                print(f"[闲聊模式] LLM调用失败: {e}")
                response = f"""我是叶片设计助手，对"{user_input}"这类问题不太擅长呢😅

不过，我在叶片设计方面可是专家！如果您需要：
• 设计新叶片（提供性能目标）
• 评估设计性能
• 可视化叶片形状

随时告诉我！"""
    elif error_message:
        response = f"""执行过程中遇到问题：

❌ {error_message}

💡 建议：
- 检查输入参数是否完整
- 确保性能目标在合理范围内
- 如需帮助，请提供更详细的信息"""
    else:
        response = "抱歉，我没能理解您的需求。请提供更详细的信息。"
    
    state["messages"].append(AIMessage(content=response))
    
    return state


# ==================== 辅助函数 ====================

def extract_21d_params(design_result: Dict) -> List[List[float]]:
    """
    从设计结果中提取21维参数列表
    
    **为什么需要这个函数**：
    - 设计工具返回的是字典格式
    - 评估工具需要的是列表格式
    - 需要转换函数进行适配
    """
    # TODO: 根据实际的design_result结构实现
    # 示例实现：
    params = []
    if "第1个设计结果" in design_result:
        design = design_result["第1个设计结果"]
        # 提取21个参数
        param_list = [
            design.get(f"root_Angle_in（叶根进口金属角[°]）", 0),
            # ... 其他20个参数
        ]
        params.append(param_list)
    return params


def extract_visualization_params(design_params: Dict) -> Dict:
    """
    从设计参数中提取可视化所需的参数
    """
    # TODO: 根据实际结构实现
    return {}


def check_performance_meets_target(
    eval_result: Dict,
    targets: Dict,
    tolerance: float = 0.05
) -> bool:
    """
    检查性能是否满足目标
    
    **参数**：
    - eval_result: 评估结果
    - targets: 目标值
    - tolerance: 容差（默认5%）
    
    **返回**：
    True表示满足，False表示不满足
    """
    # TODO: 实现具体的判断逻辑
    return False

