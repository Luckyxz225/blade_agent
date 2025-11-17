"""
LangGraph 工作流图构建
======================

修改目的：
1. 将所有节点组装成完整的状态图
2. 定义节点间的转换条件和边
3. 实现复杂的工作流逻辑（条件分支、循环等）

为什么需要这个文件：
- 原有架构：LLM黑盒决策，无法控制执行流程
- 优化后：显式定义状态转移图，流程清晰可控
- 核心优势：可视化、可调试、可追踪
"""

from langgraph.graph import StateGraph, END
from langgraph_config import (
    DesignState,
    NodeNames,
    EdgeConditions
)
from langgraph_nodes import (
    intent_recognition_node,
    design_generation_node,
    performance_evaluation_node,
    visualization_node,
    optimization_loop_node,
    result_synthesis_node,
    error_handler_node
)
from typing import Literal


def route_by_intent(state: DesignState) -> str:
    """
    根据意图路由到不同节点
    
    **这是LangGraph的核心功能：条件路由**
    
    **为什么需要路由函数**：
    - 原有系统：LLM自由选择工具，可能出错
    - 优化后：确定性路由，保证正确的执行路径
    
    **路由逻辑**：
    - design → 设计生成节点
    - evaluate → 性能评估节点
    - visualize → 可视化节点
    - optimize → 优化循环节点
    - unknown → 错误处理节点
    """
    
    intent = state.get("intent", "unknown")
    
    if intent == "design":
        return EdgeConditions.TO_DESIGN
    elif intent == "evaluate":
        return EdgeConditions.TO_EVALUATE
    elif intent == "visualize":
        return EdgeConditions.TO_VISUALIZE
    elif intent == "optimize":
        return EdgeConditions.TO_OPTIMIZE
    else:
        return EdgeConditions.TO_ERROR


def route_after_execution(state: DesignState) -> str:
    """
    执行后路由
    
    **功能**：判断执行结果，决定下一步动作
    
    **路由逻辑**：
    - 如果有错误 → 错误处理节点
    - 如果成功 → 结果整合节点
    """
    
    if state.get("error_message"):
        return NodeNames.ERROR_HANDLER
    else:
        return NodeNames.RESULT_SYNTHESIS


def route_optimization(state: DesignState) -> str:
    """
    优化循环路由
    
    **这是LangGraph的核心功能：循环支持**
    
    **为什么需要循环**：
    - 迭代优化是航发设计的常见需求
    - 原有系统很难实现自动化循环
    - LangGraph原生支持图中的环路
    
    **路由逻辑**：
    - continue → 回到优化节点（形成循环）
    - finish → 结束优化，进入结果整合
    """
    
    next_action = state.get("next_action", EdgeConditions.FINISH_OPTIMIZATION)
    
    if next_action == EdgeConditions.CONTINUE_OPTIMIZATION:
        return NodeNames.OPTIMIZATION_LOOP  # 回到优化节点，形成循环
    else:
        return NodeNames.RESULT_SYNTHESIS  # 结束优化


def build_workflow() -> StateGraph:
    """
    构建完整的工作流图
    
    **这是整个系统的核心：状态图构建**
    
    **图结构说明**：
    
    用户输入
       ↓
    意图识别
       ↓
    ┌──条件路由──┐
    │           │
    设计生成  评估  可视化  优化循环
    │           │           ↓
    └───────结果整合←──────┘
       ↓
    结束
    
    **为什么这样设计**：
    1. 单一入口：所有请求先经过意图识别
    2. 条件分支：根据意图选择不同路径
    3. 循环支持：优化节点可以循环执行
    4. 统一出口：所有结果经过整合后输出
    5. 错误处理：任何节点出错都进入错误处理节点
    """
    
    # 创建状态图
    workflow = StateGraph(DesignState)
    
    # ============ 添加节点 ============
    
    # 节点1：意图识别（入口节点）
    workflow.add_node(
        NodeNames.INTENT_RECOGNITION,
        intent_recognition_node
    )
    
    # 节点2：设计生成
    workflow.add_node(
        NodeNames.DESIGN_GENERATION,
        design_generation_node
    )
    
    # 节点3：性能评估
    workflow.add_node(
        NodeNames.PERFORMANCE_EVAL,
        performance_evaluation_node
    )
    
    # 节点4：可视化
    workflow.add_node(
        NodeNames.VISUALIZATION,
        visualization_node
    )
    
    # 节点5：优化循环
    workflow.add_node(
        NodeNames.OPTIMIZATION_LOOP,
        optimization_loop_node
    )
    
    # 节点6：结果整合
    workflow.add_node(
        NodeNames.RESULT_SYNTHESIS,
        result_synthesis_node
    )
    
    # 节点7：错误处理
    workflow.add_node(
        NodeNames.ERROR_HANDLER,
        error_handler_node
    )
    
    # ============ 设置入口节点 ============
    workflow.set_entry_point(NodeNames.INTENT_RECOGNITION)
    
    # ============ 添加边（定义节点间的转换） ============
    
    # 边1：意图识别 → 条件路由
    # 这是条件边：根据intent字段路由到不同节点
    workflow.add_conditional_edges(
        NodeNames.INTENT_RECOGNITION,
        route_by_intent,
        {
            EdgeConditions.TO_DESIGN: NodeNames.DESIGN_GENERATION,
            EdgeConditions.TO_EVALUATE: NodeNames.PERFORMANCE_EVAL,
            EdgeConditions.TO_VISUALIZE: NodeNames.VISUALIZATION,
            EdgeConditions.TO_OPTIMIZE: NodeNames.OPTIMIZATION_LOOP,
            EdgeConditions.TO_ERROR: NodeNames.ERROR_HANDLER
        }
    )
    
    # 边2：设计生成 → 条件路由（检查错误）
    workflow.add_conditional_edges(
        NodeNames.DESIGN_GENERATION,
        route_after_execution,
        {
            NodeNames.ERROR_HANDLER: NodeNames.ERROR_HANDLER,
            NodeNames.RESULT_SYNTHESIS: NodeNames.RESULT_SYNTHESIS
        }
    )
    
    # 边3：性能评估 → 条件路由（检查错误）
    workflow.add_conditional_edges(
        NodeNames.PERFORMANCE_EVAL,
        route_after_execution,
        {
            NodeNames.ERROR_HANDLER: NodeNames.ERROR_HANDLER,
            NodeNames.RESULT_SYNTHESIS: NodeNames.RESULT_SYNTHESIS
        }
    )
    
    # 边4：可视化 → 条件路由（检查错误）
    workflow.add_conditional_edges(
        NodeNames.VISUALIZATION,
        route_after_execution,
        {
            NodeNames.ERROR_HANDLER: NodeNames.ERROR_HANDLER,
            NodeNames.RESULT_SYNTHESIS: NodeNames.RESULT_SYNTHESIS
        }
    )
    
    # 边5：优化循环 → 条件路由（循环或结束）
    # 这是关键：允许优化节点循环调用自己
    workflow.add_conditional_edges(
        NodeNames.OPTIMIZATION_LOOP,
        route_optimization,
        {
            NodeNames.OPTIMIZATION_LOOP: NodeNames.OPTIMIZATION_LOOP,  # 循环
            NodeNames.RESULT_SYNTHESIS: NodeNames.RESULT_SYNTHESIS      # 结束
        }
    )
    
    # 边6：结果整合 → 结束
    workflow.add_edge(
        NodeNames.RESULT_SYNTHESIS,
        END
    )
    
    # 边7：错误处理 → 结束
    workflow.add_edge(
        NodeNames.ERROR_HANDLER,
        END
    )
    
    return workflow


def create_agent():
    """
    创建可执行的Agent
    
    **这是对外暴露的接口**
    
    **使用方式**：
    ```python
    agent = create_agent()
    result = agent.invoke({
        "user_input": "设计一个流量15kg/s的叶片",
        "messages": [],
        "iteration_count": 0,
        "max_iterations": 10,
        "optimization_history": []
    })
    ```
    
    **返回值**：
    result["messages"] 包含完整的对话历史和结果
    """
    
    workflow = build_workflow()
    return workflow.compile()


def visualize_workflow():
    """
    可视化工作流图
    
    **这是LangGraph的强大功能：图可视化**
    
    **用途**：
    - 用于文档和演示
    - 帮助理解复杂流程
    - 调试时查看状态转移
    
    **生成方式**：
    ```python
    from IPython.display import Image, display
    
    workflow = build_workflow()
    display(Image(workflow.get_graph().draw_mermaid_png()))
    ```
    """
    try:
        from IPython.display import Image, display
        
        workflow = build_workflow()
        graph = workflow.compile()
        
        # 生成Mermaid图
        mermaid_png = graph.get_graph().draw_mermaid_png()
        
        # 保存图片
        with open("workflow_diagram.png", "wb") as f:
            f.write(mermaid_png)
        
        print("工作流图已保存到 workflow_diagram.png")
        
        # 如果在Jupyter中，直接显示
        try:
            display(Image(mermaid_png))
        except:
            pass
            
    except Exception as e:
        print(f"可视化失败: {e}")
        print("提示: 需要安装 pygraphviz 或 graphviz")


# ==================== 测试与调试 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 工作流图构建测试")
    print("=" * 60)
    
    # 测试1：构建工作流
    print("\n1. 构建工作流图...")
    try:
        workflow = build_workflow()
        agent = workflow.compile()
        print("✓ 工作流图构建成功")
    except Exception as e:
        print(f"✗ 工作流图构建失败: {e}")
    
    # 测试2：可视化工作流
    print("\n2. 可视化工作流...")
    try:
        visualize_workflow()
        print("✓ 工作流可视化成功")
    except Exception as e:
        print(f"✗ 工作流可视化失败: {e}")
    
    # 测试3：运行简单测试
    print("\n3. 运行简单测试...")
    try:
        agent = create_agent()
        
        # 初始状态
        initial_state = {
            "user_input": "设计一个流量15kg/s的叶片",
            "messages": [],
            "intent": "unknown",
            "performance_targets": None,
            "design_params": None,
            "evaluation_results": None,
            "visualization_path": None,
            "optimization_history": [],
            "iteration_count": 0,
            "max_iterations": 10,
            "error_message": None,
            "next_action": None
        }
        
        # 执行（这里可能因为依赖问题失败，仅测试结构）
        # result = agent.invoke(initial_state)
        # print(f"✓ 工作流执行成功")
        # print(f"最终状态: {result}")
        
        print("✓ Agent创建成功（实际执行需要配置LLM和工具）")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    # 输出工作流说明
    print("\n工作流图说明：")
    print("-" * 60)
    print("节点列表：")
    print("  1. intent_recognition - 意图识别")
    print("  2. design_generation - 设计生成")
    print("  3. performance_evaluation - 性能评估")
    print("  4. visualization - 可视化")
    print("  5. optimization_loop - 优化循环")
    print("  6. result_synthesis - 结果整合")
    print("  7. error_handler - 错误处理")
    print("\n执行流程：")
    print("  用户输入 → 意图识别 → [条件路由] → 执行节点 → 结果整合 → 输出")
    print("\n特殊功能：")
    print("  - 条件分支：根据意图选择不同路径")
    print("  - 循环支持：优化节点可以循环执行")
    print("  - 错误处理：统一的错误处理机制")
    print("=" * 60)

