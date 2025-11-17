"""
LangGraph 版本的 Web 应用
===========================

修改目的：
1. 将原有Flask应用迁移到LangGraph架构
2. 保持Web接口不变，仅更换后端引擎
3. 增强可观测性和流程控制能力

为什么这样修改：
- 原有web_app_v3.py使用agents-py框架
- 优化后使用LangGraph，提供更强大的工作流能力
- 保持前端兼容性，最小化改动成本

主要改动：
1. 移除agents-py相关导入，改用LangGraph
2. 将Agent替换为LangGraph工作流
3. 增强状态追踪和可视化功能
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import asyncio
from typing import Union
from datetime import datetime
import functools
import threading
from collections import deque
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 导入LangGraph组件
from langgraph_workflow import create_agent
from langgraph_config import DesignState

load_dotenv()

# ========================= 初始化 Flask 应用 ========================= #
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')


# ========================= 全局变量 ========================= #
conversation_history = deque(maxlen=1000)  # 对话历史
current_workflow = None  # 当前LangGraph工作流
thread_local_data = threading.local()  # 线程本地存储


# ========================= 静态文件路由 ========================= #
@app.route('/static/images/<path:filename>')
def serve_image(filename):
    """
    静态图像服务
    
    **无需修改**：这是标准的Flask静态文件服务
    """
    return send_from_directory('static/images', filename)


# ========================= 异步装饰器 ========================= #
def async_route(f):
    """
    支持Flask路由异步处理的装饰器
    
    **无需修改**：这是通用的异步支持函数
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(f(*args, **kwargs))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(f(*args, **kwargs))
    return wrapper


# ========================= 路由定义 ========================= #

@app.route('/')
def index():
    """
    首页路由
    
    **修改说明**：
    - 保持不变，仍然渲染index_v3.html
    - 前端无需任何改动
    """
    env_config = {
        'api_key': os.getenv('API_KEY', ''),
        'base_url': os.getenv('BASE_URL', ''),
        'model_name': os.getenv('MODEL_NAME', '')
    }
    return render_template('index_V3.html', env_config=env_config)


@app.route('/api/config', methods=['POST'])
@async_route
async def update_config():
    """
    更新API配置
    
    **修改说明**：
    - 原有：创建agents-py的Agent
    - 优化后：创建LangGraph工作流
    
    **为什么这样改**：
    - LangGraph使用标准的LangChain LLM接口
    - 配置过程更简单，无需复杂的Agent初始化
    - 工作流是无状态的，配置后即可使用
    """
    global current_workflow
    
    data = request.json
    api_key = data.get('api_key') or os.getenv('API_KEY')
    base_url = data.get('base_url') or os.getenv('BASE_URL')
    model_name = data.get('model_name') or os.getenv('MODEL_NAME')
    
    if not all([api_key, base_url, model_name]):
        return jsonify({
            "success": False,
            "error": "需通过前端输入或.env文件配置API_KEY/BASE_URL/MODEL_NAME"
        })
    
    if not base_url.startswith(('http://', 'https://')):
        return jsonify({"success": False, "error": "基础URL必须以http://或https://开头"})
    
    try:
        # 设置环境变量（LangChain会自动读取）
        os.environ['OPENAI_API_KEY'] = api_key
        os.environ['OPENAI_BASE_URL'] = base_url  # 使用标准键名
        os.environ['OPENAI_API_BASE'] = base_url  # 兼容旧键名
        os.environ['OPENAI_MODEL_NAME'] = model_name
        
        # 测试连接
        test_llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0
        )
        test_result = await test_llm.ainvoke([HumanMessage(content="测试连接")])
        
        if not test_result or not test_result.content:
            raise Exception("API测试调用失败：未能获取有效响应")
        
        # 创建LangGraph工作流（重点改动）
        current_workflow = create_agent()
        
        return jsonify({
            "success": True,
            "message": f"配置更新成功！已连接到模型: {model_name}",
            "framework": "LangGraph"  # 标识使用的框架
        })
        
    except Exception as e:
        current_workflow = None
        return jsonify({"success": False, "error": f"配置失败: {str(e)}"})


@app.route('/api/chat', methods=['POST'])
@app.route('/api/chat/stream', methods=['POST'])  # 添加兼容路由
@async_route
async def chat():
    """
    聊天接口（支持 /api/chat 和 /api/chat/stream）
    
    **修改说明**：
    - 原有：调用agents-py的Runner.run
    - 优化后：调用LangGraph工作流的invoke
    - 添加 /api/chat/stream 路由以兼容前端
    
    **为什么这样改**：
    - LangGraph的invoke方法是标准接口
    - 输入是DesignState字典，输出也是DesignState
    - 更容易追踪状态变化和中间结果
    
    **关键改动**：
    1. 构建初始状态（而非简单的消息字符串）
    2. 调用workflow.invoke（而非Runner.run）
    3. 从返回状态中提取结果（而非result.final_output）
    """
    if not current_workflow:
        return jsonify({"error": "请先配置API设置"})
    
    data = request.json
    user_message = data.get('message')
    
    if not user_message or not user_message.strip():
        return jsonify({"error": "请输入有效的问题"})
    
    if len(user_message) > 10000:
        return jsonify({"error": "消息长度过长，请控制在10000字符以内"})
    
    try:
        print(f"\n{'='*60}")
        print(f"[Web] 收到用户请求: {user_message[:50]}...")
        print(f"{'='*60}")
        
        # 构建初始状态（重点改动）
        initial_state: DesignState = {
            "user_input": user_message,
            "messages": [HumanMessage(content=user_message)],  # 使用 LangChain 消息对象
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
        
        # 执行工作流（重点改动）
        print(f"[Web] 开始执行LangGraph工作流...")
        final_state = await asyncio.to_thread(
            current_workflow.invoke,
            initial_state
        )
        print(f"[Web] 工作流执行完成")
        
        # 提取结果（重点改动）
        response_data = {
            "message": extract_final_message(final_state),
            "intent": final_state.get("intent", "unknown"),
            "design_params": final_state.get("design_params"),
            "evaluation_results": final_state.get("evaluation_results"),
            "visualization_path": final_state.get("visualization_path"),
            "optimization_history": final_state.get("optimization_history", []),
            "error": final_state.get("error_message"),
            "timestamp": datetime.now().isoformat(),
            "workflow_status": "completed"
        }
        
        # 保存对话历史
        conversation_history.append({
            "user_message": user_message,
            "assistant_response": response_data["message"],
            "timestamp": datetime.now().isoformat(),
            "intent": final_state.get("intent")
        })
        
        return jsonify(response_data)
        
    except Exception as e:
        msg = str(e).lower()
        if "authentication" in msg or "api key" in msg:
            return jsonify({"error": "API密钥无效或已过期，请检查配置"})
        elif "connection" in msg or "network" in msg:
            return jsonify({"error": "网络连接失败，请检查网络设置和API地址"})
        elif "model" in msg and ("not found" in msg or "not exist" in msg):
            return jsonify({"error": "模型不存在，请检查模型名称是否正确"})
        elif "timeout" in msg:
            return jsonify({"error": "请求超时，请稍后重试"})
        else:
            return jsonify({"error": f"处理请求时出错: {str(e)}"})


@app.route('/api/workflow/status', methods=['GET'])
def get_workflow_status():
    """
    获取工作流状态
    
    **新增功能**：
    - 原有系统没有这个接口
    - LangGraph提供了强大的状态追踪能力
    - 可以查看工作流执行到哪个节点
    
    **应用场景**：
    - 调试时查看执行路径
    - 监控长时间运行的优化任务
    - 用于前端显示进度条
    """
    if not current_workflow:
        return jsonify({"error": "工作流未初始化"})
    
    try:
        # 获取工作流图信息
        graph = current_workflow.get_graph()
        
        return jsonify({
            "status": "active",
            "nodes": list(graph.nodes.keys()),
            "edges": len(graph.edges),
            "framework": "LangGraph",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": f"获取状态失败: {str(e)}"})


@app.route('/api/workflow/visualize', methods=['GET'])
def visualize_workflow():
    """
    可视化工作流图
    
    **新增功能**：
    - LangGraph支持将工作流图导出为图片
    - 可以生成Mermaid格式的流程图
    - 用于文档和演示
    
    **应用场景**：
    - 向用户展示系统能力
    - 帮助开发者理解流程
    - 生成技术文档
    """
    if not current_workflow:
        return jsonify({"error": "工作流未初始化"})
    
    try:
        graph = current_workflow.get_graph()
        
        # 生成Mermaid格式的流程图
        mermaid_code = graph.draw_mermaid()
        
        return jsonify({
            "mermaid": mermaid_code,
            "message": "工作流图生成成功",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": f"可视化失败: {str(e)}"})


@app.route('/api/tools', methods=['GET'])
def get_tools():
    """
    获取工具列表
    
    **修改说明**：
    - 保持接口不变
    - 工具定义在langgraph_tools.py中
    """
    from langgraph_tools import LANGGRAPH_TOOLS
    
    tools = []
    for tool in LANGGRAPH_TOOLS:
        tools.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": str(tool.args_schema.schema() if hasattr(tool, 'args_schema') else {})
        })
    
    return jsonify(tools)


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    健康检查
    
    **修改说明**：
    - 添加LangGraph特有的信息
    - 显示工作流状态
    """
    return jsonify({
        "status": "healthy",
        "workflow_configured": current_workflow is not None,
        "conversation_count": len(conversation_history),
        "framework": "LangGraph",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/clear-history', methods=['POST'])
def clear_conversation_history():
    """
    清除对话历史
    
    **无需修改**：这是通用功能
    """
    conversation_history.clear()
    return jsonify({
        "success": True,
        "message": "对话历史已清除",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/history', methods=['GET'])
def get_conversation_history():
    """
    获取对话历史
    
    **无需修改**：这是通用功能
    """
    limit = min(request.args.get('limit', 20, type=int), 100)
    recent = list(conversation_history)[-limit:]
    
    history = []
    for item in recent:
        if item.get('user_message'):
            history.append({
                "role": "user",
                "content": item['user_message'],
                "timestamp": item.get('timestamp')
            })
        if item.get('assistant_response'):
            history.append({
                "role": "assistant",
                "content": item['assistant_response'],
                "timestamp": item.get('timestamp')
            })
    
    return jsonify({
        "history": history,
        "total_count": len(conversation_history),
        "returned_count": len(history)
    })


# ========================= 辅助函数 ========================= #

def extract_final_message(state: DesignState) -> str:
    """
    从状态中提取最终消息
    
    **新增函数**：
    - LangGraph返回的是状态字典
    - 从messages列表中提取最后的AI消息
    
    **为什么需要这个函数**：
    - 原有系统的result.final_output是字符串
    - 新系统需要从messages列表中提取
    - 保持与前端的兼容性
    """
    messages = state.get("messages", [])
    
    # 从后往前找最后一条AI消息
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'ai':
            return msg.content
        elif hasattr(msg, 'content'):  # 兼容处理
            return msg.content
    
    # 如果没有消息，返回默认值
    return "处理完成，但没有生成回复"


# ========================= 启动应用 ========================= #

if __name__ == '__main__':
    print("=" * 60)
    print("LangGraph 版本的航发设计智能体 Web 应用")
    print("=" * 60)
    print(f"启动时间: {datetime.now().isoformat()}")
    print(f"框架: LangGraph + LangChain + Flask")
    print(f"端口: 4000")
    print("=" * 60)
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=4000
    )

