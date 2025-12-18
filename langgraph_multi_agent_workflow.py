"""
LangGraph 多智能体工作流构建（增强版）
======================================

构建多Agent协作的状态图
新增：任务调度器节点，支持多任务序列执行
"""

from langgraph.graph import StateGraph, END
from langgraph_multi_agent_config import (
    MultiAgentState, AgentNames, EdgeConditions
)
from langgraph_multi_agent_nodes import (
    coordinator_node,
    design_expert_node,
    evaluation_expert_node,   # 性能评估专家
    visualization_expert_node,  # 可视化专家
    optimization_expert_node,  # 优化专家
    chat_expert_node,
    task_scheduler_node,  # 任务调度器
    error_handler_node,
    set_task_result_callback,  # 任务结果回调设置
    _task_result_callback  # 任务结果回调变量
)


def get_task_result_callback():
    """获取任务结果回调函数"""
    from langgraph_multi_agent_nodes import _task_result_callback
    return _task_result_callback


# ==================== 路由函数（增强版）====================

def route_from_coordinator(state: MultiAgentState) -> str:
    """
    从协调者路由（增强版）
    
    决策逻辑：
    1. 如果等待用户输入 → 结束（等待用户回答）
    2. 如果需要调度器 → 任务调度器
    3. 如果有错误 → 错误处理节点
    4. 如果指定了目标Agent → 路由到对应Agent
    5. 否则 → 结束
    """
    
    # 检查是否等待用户输入
    if state.get("waiting_for_user"):
        print(f"[Router] 协调者 → 结束（等待用户输入）")
        return END
    
    # 新增：检查是否需要调度器
    if state.get("next_action") == "to_scheduler":
        print(f"[Router] 协调者 → 任务调度器")
        return "to_scheduler"
    
    # 检查错误
    if state.get("error_message"):
        print(f"[Router] 协调者 → 错误处理")
        return EdgeConditions.TO_ERROR
    
    # 路由到目标Agent
    target_agent = state.get("target_agent")
    
    if target_agent == AgentNames.DESIGN_EXPERT:
        print(f"[Router] 协调者 → 设计专家")
        return EdgeConditions.TO_DESIGN
    
    elif target_agent == AgentNames.EVALUATION_EXPERT:
        print(f"[Router] 协调者 → 评估专家")
        return EdgeConditions.TO_EVALUATION
    
    elif target_agent == AgentNames.VISUALIZATION_EXPERT:
        print(f"[Router] 协调者 → 可视化专家")
        return EdgeConditions.TO_VISUALIZATION
    
    elif target_agent == AgentNames.OPTIMIZATION_EXPERT:
        print(f"[Router] 协调者 → 优化专家")
        return EdgeConditions.TO_OPTIMIZATION
    
    elif target_agent == AgentNames.CHAT_EXPERT:
        print(f"[Router] 协调者 → 闲聊专家")
        return EdgeConditions.TO_CHAT
    
    else:
        # 默认结束
        print(f"[Router] 协调者 → 结束")
        return END


def route_from_scheduler(state: MultiAgentState) -> str:
    """
    从任务调度器路由
    
    决策逻辑：
    1. 如果等待用户输入 → 结束
    2. 如果所有任务完成 → 结束
    3. 根据target_agent路由到对应专家
    """
    
    # 检查是否等待用户输入
    if state.get("waiting_for_user"):
        print(f"[Router] 调度器 → 结束（等待用户输入）")
        return END
    
    # 检查是否所有任务完成
    if state.get("next_action") == EdgeConditions.TO_END:
        print(f"[Router] 调度器 → 结束（所有任务完成）")
        return END
    
    # 检查错误
    if state.get("error_message"):
        print(f"[Router] 调度器 → 错误处理")
        return EdgeConditions.TO_ERROR
    
    # 路由到目标Agent
    target_agent = state.get("target_agent")
    
    if target_agent == AgentNames.DESIGN_EXPERT:
        print(f"[Router] 调度器 → 设计专家")
        return EdgeConditions.TO_DESIGN
    
    elif target_agent == AgentNames.EVALUATION_EXPERT:
        print(f"[Router] 调度器 → 评估专家")
        return EdgeConditions.TO_EVALUATION
    
    elif target_agent == AgentNames.VISUALIZATION_EXPERT:
        print(f"[Router] 调度器 → 可视化专家")
        return EdgeConditions.TO_VISUALIZATION
    
    elif target_agent == AgentNames.OPTIMIZATION_EXPERT:
        print(f"[Router] 调度器 → 优化专家")
        return EdgeConditions.TO_OPTIMIZATION
    
    elif target_agent == AgentNames.CHAT_EXPERT:
        print(f"[Router] 调度器 → 闲聊专家")
        return EdgeConditions.TO_CHAT
    
    else:
        print(f"[Router] 调度器 → 结束（未知目标）")
        return END


def route_from_expert(state: MultiAgentState) -> str:
    """
    从专家Agent路由（增强版）
    
    决策逻辑：
    1. 如果等待用户输入 → 结束（等待用户回答）
    2. 如果需要调度器（多任务序列）→ 任务调度器
    3. 如果next_action是TO_END → 结束
    4. 如果有错误 → 错误处理
    5. 否则 → 结束（专家完成即结束）
    """
    
    current_agent = state.get("current_agent", "unknown")
    
    # 检查是否等待用户输入
    if state.get("waiting_for_user"):
        print(f"[Router] {current_agent} → 结束（等待用户输入）")
        return END
    
    # 新增：检查是否需要调度器（多任务序列）
    if state.get("next_action") == "to_scheduler":
        print(f"[Router] {current_agent} → 任务调度器（下一个任务）")
        return "to_scheduler"
    
    # 检查next_action
    next_action = state.get("next_action")
    if next_action == EdgeConditions.TO_END or state.get("final_result"):
        print(f"[Router] {current_agent} → 结束")
        return END
    
    # 检查错误
    if state.get("error_message"):
        print(f"[Router] {current_agent} → 错误处理")
        return EdgeConditions.TO_ERROR
    
    # 默认结束
    print(f"[Router] {current_agent} → 结束")
    return END


# ==================== 工作流构建（增强版）====================

def build_multi_agent_workflow() -> StateGraph:
    """
    构建多智能体工作流（增强版 - 含任务调度器）
    
    工作流结构：
    
    START → Coordinator
              ↓
         {条件路由}
      ↙   ↓   ↓   ↘   ↘
   Design Eval Viz Chat Scheduler
      ↓   ↓   ↓    ↓      ↓
      ↘   ↓   ↓    ↓    ↙ (可回到专家)
          END END END
    
    特点：
    1. Coordinator识别意图并分派
    2. 专家执行后可返回调度器（多任务序列）
    3. 调度器控制多任务执行顺序
    4. 未知意图由ChatExpert处理
    5. 支持智能判断是否使用已有结果
    """
    
    # 创建状态图
    workflow = StateGraph(MultiAgentState)
    
    # ============ 添加节点 ============
    
    # 主节点
    workflow.add_node(AgentNames.COORDINATOR, coordinator_node)
    
    # 任务调度器节点（新增）
    workflow.add_node("task_scheduler", task_scheduler_node)
    
    # 专家节点
    workflow.add_node(AgentNames.DESIGN_EXPERT, design_expert_node)
    workflow.add_node(AgentNames.EVALUATION_EXPERT, evaluation_expert_node)
    workflow.add_node(AgentNames.VISUALIZATION_EXPERT, visualization_expert_node)
    workflow.add_node(AgentNames.OPTIMIZATION_EXPERT, optimization_expert_node)  # 新增：优化专家
    workflow.add_node(AgentNames.CHAT_EXPERT, chat_expert_node)
    
    # 错误处理节点
    workflow.add_node("error_handler", error_handler_node)
    
    # ============ 设置入口 ============
    
    workflow.set_entry_point(AgentNames.COORDINATOR)
    
    # ============ 添加边 ============
    
    # 1. 协调者的条件路由（新增调度器路由和优化专家）
    workflow.add_conditional_edges(
        AgentNames.COORDINATOR,
        route_from_coordinator,
        {
            EdgeConditions.TO_DESIGN: AgentNames.DESIGN_EXPERT,
            EdgeConditions.TO_EVALUATION: AgentNames.EVALUATION_EXPERT,
            EdgeConditions.TO_VISUALIZATION: AgentNames.VISUALIZATION_EXPERT,
            EdgeConditions.TO_OPTIMIZATION: AgentNames.OPTIMIZATION_EXPERT,  # 新增：优化专家
            EdgeConditions.TO_CHAT: AgentNames.CHAT_EXPERT,
            EdgeConditions.TO_ERROR: "error_handler",
            "to_scheduler": "task_scheduler",
            END: END
        }
    )
    
    # 2. 任务调度器的条件路由（含优化专家和闲聊专家）
    workflow.add_conditional_edges(
        "task_scheduler",
        route_from_scheduler,
        {
            EdgeConditions.TO_DESIGN: AgentNames.DESIGN_EXPERT,
            EdgeConditions.TO_EVALUATION: AgentNames.EVALUATION_EXPERT,
            EdgeConditions.TO_VISUALIZATION: AgentNames.VISUALIZATION_EXPERT,
            EdgeConditions.TO_OPTIMIZATION: AgentNames.OPTIMIZATION_EXPERT,
            EdgeConditions.TO_CHAT: AgentNames.CHAT_EXPERT,  # 新增：闲聊专家
            EdgeConditions.TO_ERROR: "error_handler",
            END: END
        }
    )
    
    # 3. 设计专家 → END 或 调度器
    workflow.add_conditional_edges(
        AgentNames.DESIGN_EXPERT,
        route_from_expert,
        {
            EdgeConditions.TO_ERROR: "error_handler",
            "to_scheduler": "task_scheduler",  # 新增：返回调度器执行下一个任务
            END: END
        }
    )
    
    # 4. 评估专家 → END 或 调度器
    workflow.add_conditional_edges(
        AgentNames.EVALUATION_EXPERT,
        route_from_expert,
        {
            EdgeConditions.TO_ERROR: "error_handler",
            "to_scheduler": "task_scheduler",  # 新增
            END: END
        }
    )
    
    # 5. 可视化专家 → END 或 调度器
    workflow.add_conditional_edges(
        AgentNames.VISUALIZATION_EXPERT,
        route_from_expert,
        {
            EdgeConditions.TO_ERROR: "error_handler",
            "to_scheduler": "task_scheduler",
            END: END
        }
    )
    
    # 6. 优化专家 → END 或 调度器
    workflow.add_conditional_edges(
        AgentNames.OPTIMIZATION_EXPERT,
        route_from_expert,
        {
            EdgeConditions.TO_ERROR: "error_handler",
            "to_scheduler": "task_scheduler",
            END: END
        }
    )
    
    # 7. 闲聊专家 → END 或 调度器
    workflow.add_conditional_edges(
        AgentNames.CHAT_EXPERT,
        route_from_expert,
        {
            EdgeConditions.TO_ERROR: "error_handler",
            "to_scheduler": "task_scheduler",  # 新增：支持多任务序列
            END: END
        }
    )
    
    # 8. 错误处理 → END
    workflow.add_edge("error_handler", END)
    
    return workflow


def create_multi_agent_app():
    """
    创建可执行的多智能体应用
    
    返回：
        编译后的工作流
    """
    
    workflow = build_multi_agent_workflow()
    app = workflow.compile()
    
    return app


def visualize_multi_agent_workflow():
    """
    可视化多智能体工作流
    
    生成 Mermaid 图或 PNG 图
    """
    try:
        workflow = build_multi_agent_workflow()
        graph = workflow.compile()
        
        # 尝试生成图片
        try:
            from IPython.display import Image
            mermaid_png = graph.get_graph().draw_mermaid_png()
            
            with open("multi_agent_workflow.png", "wb") as f:
                f.write(mermaid_png)
            
            print("✅ 工作流图已保存到: multi_agent_workflow.png")
            
            return mermaid_png
        except:
            # 如果无法生成图片，打印文本描述
            print("工作流结构：")
            print("  START → Coordinator")
            print("            ↓")
            print("       {条件路由}")
            print("       ↙  ↓  ↘  ↘")
            print("  Design Eval Viz User")
            print("     ↓    ↓   ↓    ↓")
            print("  Coordinator ← ← ←")
            print("     ↓")
            print("    END")
            
    except Exception as e:
        print(f"可视化失败: {e}")


# ==================== 测试与调试 ====================

if __name__ == "__main__":
    print("="*60)
    print("LangGraph 多智能体工作流构建测试")
    print("="*60)
    
    # 测试1: 构建工作流
    print("\n1. 构建工作流图...")
    try:
        workflow = build_multi_agent_workflow()
        app = workflow.compile()
        print("✓ 工作流图构建成功")
    except Exception as e:
        print(f"✗ 工作流图构建失败: {e}")
    
    # 测试2: 可视化
    print("\n2. 可视化工作流...")
    visualize_multi_agent_workflow()
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    
    print("\n工作流说明：")
    print("-" * 60)
    print("节点列表：")
    print("  1. coordinator - 协调者（中心节点）")
    print("  2. design_expert - 设计专家")
    print("  3. evaluation_expert - 评估专家")
    print("  4. visualization_expert - 可视化专家")
    print("  5. user_interface - 用户交互接口")
    print("  6. error_handler - 错误处理")
    print("\n执行流程：")
    print("  用户输入 → 协调者分析 → 分派专家 → 专家执行 → 返回协调者 → 整合结果")
    print("\n特殊功能：")
    print("  - 反问用户：任何Agent可以触发user_interface节点")
    print("  - 暂停/恢复：用户回答后恢复到原Agent")
    print("  - Agent协作：通过协调者中转消息")
    print("="*60)

