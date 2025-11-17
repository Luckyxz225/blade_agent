# 引入Flask框架相关模块，用于构建Web应用
from flask import Flask, render_template, request, jsonify, send_from_directory
# 引入系统模块
import os
import asyncio
from typing import Union
from datetime import datetime
import functools
import threading
from collections import deque
import weakref

# 引入自定义agent系统相关类与工具
from agents import Agent, RunContextWrapper, RunHooks, Runner, ModelSettings, Tool
from agents.extensions.models.litellm_model import LitellmModel
from pydantic import BaseModel
from dotenv import load_dotenv
# 引入三个叶片设计相关的工具函数
from tools import plot_blade_profile
from generative_design.compressor_design import blade_design
from generative_design.blade_performance_evaluation import blade_performance_evaluation

load_dotenv()  # 加载环境变量
# ------------------------- 初始化 Flask 应用 ------------------------- #

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')  # 设置会话密钥

# 添加静态文件路由
@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('../myAgent-main/static/images', filename)

# ------------------------- 全局与线程变量初始化 ------------------------- #
thread_local_data = threading.local()  # 用于线程本地存储，避免多线程数据混乱
conversation_history = deque(maxlen=1000)  # 存储对话历史，最多保存1000条
_agent_cache = weakref.WeakValueDictionary()  # 使用弱引用缓存代理对象
_calculation_cache = {}  # 通用计算缓存，用于缓存计算结果避免重复
_cache_max_size = 1000  # 缓存最大容量限制
current_agent = None  # 当前激活的Agent代理对象

# ------------------------- 辅助工具函数定义 ------------------------- #
def safe_float_conversion(value: Union[str, int, float]) -> float:
    """
    安全地将输入值转换为 float 类型。
    支持字符串、整数和浮点数，自动处理异常和非法输入。
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("空字符串无法转换为数字")
        try:
            return float(value)
        except ValueError as e:
            raise ValueError(f"无法将 '{value}' 转换为数字: {str(e)}")
    raise ValueError(f"不支持的类型: {type(value).__name__}")

def _cached_calculation(key: str, calc_func):
    """
    缓存计算函数的结果，用于避免重复计算。
    超过最大容量后会清除最早的缓存项。
    """
    if key in _calculation_cache:
        return _calculation_cache[key]
    result = calc_func()
    if len(_calculation_cache) >= _cache_max_size:
        del _calculation_cache[next(iter(_calculation_cache))]
    _calculation_cache[key] = result
    return result

def _build_context_message(current_user_message: str, max_history_pairs: int = 5) -> str:
    """
    构造带有上下文的用户消息，便于模型理解当前问题。
    从对话历史中最多选取指定条数拼接。
    """
    if not conversation_history:
        return current_user_message
    context_parts = ["=== 历史对话 ==="]
    for idx, history_item in enumerate(list(conversation_history)[-max_history_pairs:], 1):
        context_parts.extend([
            f"对话{idx}:",
            f"用户: {history_item.get('user_message', '')}",
            f"助手: {history_item.get('assistant_response', '')}", ""
        ])
    context_parts.append("=== 当前问题 ===")
    context_parts.append(current_user_message)
    return "\n".join(context_parts)

# ------------------------- 异步函数装饰器 ------------------------- #
def async_route(f):
    """
    用于支持 Flask 路由异步处理的装饰器。
    自动处理事件循环的创建与运行。
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

# ------------------------- 自定义钩子类，用于追踪工具调用过程 ------------------------- #
class MyRunHooks(RunHooks):
    def __init__(self):
        self.event_counter = 0
        self.current_tool_call = None
        self.current_tool_arguments = None
        self.task_plan = []
        self.current_step = 0

    @property
    def tool_calls(self):
        # 使用线程本地存储保存工具调用信息
        if not hasattr(thread_local_data, 'tool_calls'):
            thread_local_data.tool_calls = []
        return thread_local_data.tool_calls

    def clear_tool_calls(self):
        thread_local_data.tool_calls = []

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool):
        # 工具调用开始事件
        self.event_counter += 1
        self.current_tool_call = {"tool": tool.name, "arguments": {}, "output": None, "agent": agent.name}
        
        # 将工具调用计划发送到前端任务区域 - 显示"我将调用...工具"
        if hasattr(thread_local_data, 'tool_planning'):
            thread_local_data.tool_planning.append({
                'type': 'tool_plan',
                'tool': tool.name,
                'status': 'planning',
                'message': f'我将调用 {tool.name} 工具',
                'timestamp': datetime.now().isoformat()
            })
        
        # 更新状态为"正在调用"
        if hasattr(thread_local_data, 'tool_planning'):
            thread_local_data.tool_planning.append({
                'type': 'tool_executing',
                'tool': tool.name,
                'status': 'executing',
                'message': f'{tool.name} 工具正在调用',
                'timestamp': datetime.now().isoformat()
            })

    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Tool, output):
        # 工具调用结束事件，记录结果
        self.event_counter += 1
        if self.current_tool_call:
            output_str = str(output)
            if len(output_str) > 10000:
                output_str = output_str[:10000] + "... (truncated)"
            self.current_tool_call.update({
                "output": output_str,
                "arguments": self.current_tool_arguments or {}
            })
            self.tool_calls.append(self.current_tool_call)
            self.current_tool_call = None
            self.current_tool_arguments = None
            
            # 将工具结果发送到前端结果区域
            if hasattr(thread_local_data, 'tool_results'):
                thread_local_data.tool_results.append({
                    'type': 'tool_result',
                    'tool': tool.name,
                    'output': output_str,
                    'timestamp': datetime.now().isoformat()
                })

    def set_tool_arguments(self, arguments):
        self.current_tool_arguments = arguments
        
    def add_task_step(self, step_description):
        """添加任务步骤到规划中"""
        self.task_plan.append(step_description)
        
    def update_current_step(self, step_index):
        """更新当前执行步骤"""
        self.current_step = step_index

# ------------------------- 路由定义 ------------------------- #
@app.route('/')
def index():
    """
    首页路由，渲染index.html页面，并传递环境变量配置
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
    更新API配置，包括 API 密钥、基础 URL 和模型名称。
    成功后初始化当前Agent对象并测试连接。
    """
    global current_agent
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
        # 创建 LLM 模型和测试 Agent
        llm = LitellmModel(model=model_name, api_key=api_key, base_url=base_url)
        test_agent = Agent(
            name="测试代理",
            instructions="你是一个测试助手，请回复\"连接成功\"",
            model=llm,
            model_settings=ModelSettings(temperature=0.0),
            tools=[]
        )
        test_result = await Runner.run(test_agent, "测试连接")
        if not test_result or not test_result.final_output:
            raise Exception("API测试调用失败：未能获取有效响应")

        # 创建实际使用的代理
        instructions = """
        你是来自中国科学院工程热物理研究所数字孪生研究中心的工程师,名字是IET-Agent，你的专业是航空发动机压气机设计。你的输出应该与用户的语言保持一致。

        工具使用指导：
        1. 对于基础的叶片设计需求，使用blade_design生成基础叶片参数,并以表格形式给出所有的设计参数以及预估性能，如果有多个设计尽量将结果放在一个表格，并从专业设计者角度简要分析不同以及可能造成的性能差异
        2. 对于基础的叶片性能评估，使用 blade_performance_evaluation 评估叶片气动性能
        3. 对于明确需要优化的设计需求，使用 blade_optimization 工具进行自动化迭代优化
        4. 对于需要可视化的叶片造型，使用 plot_blade_profile 工具进行可视化
        4. 始终提供专业的技术分析和工程建议

        注意事项：
        - 在回答过程中，你应该首先展示你的任务规划过程，如：针对用户提出de...,我将...进行解决
        - 在进行叶片设计和优化时，需充分考虑气动性能、结构强度和制造工艺等因素。
        - 对于复杂的设计问题，建议分解为多个子问题，逐步求解。
        - 在使用优化工具时，需明确初始方案和性能目标，以便获得最佳设计方案。
        - 调用工具函数时，确保输入输出参数类型正确，避免类型转换错误。
        """

        current_agent = Agent(
            name="工程热物理研究所数字孪生研究中心工程师",
            instructions=instructions,
            model=llm,
            model_settings=ModelSettings(temperature=0.5),
            tools=[blade_design, blade_performance_evaluation, plot_blade_profile]
        )
        return jsonify({"success": True, "message": f"配置更新成功！已连接到模型: {model_name}"})
    except Exception as e:
        current_agent = None
        return jsonify({"success": False, "error": f"配置失败: {str(e)}"})

@app.route('/api/chat', methods=['POST'])
@async_route
async def chat():
    """
    聊天接口，接受用户消息，结合上下文历史与Agent进行交互并返回模型输出。
    """
    if not current_agent:
        return jsonify({"error": "请先配置API设置"})
    data = request.json
    user_message = data.get('message')
    if not user_message or not user_message.strip():
        return jsonify({"error": "请输入有效的问题"})
    if len(user_message) > 10000:
        return jsonify({"error": "消息长度过长，请控制在10000字符以内"})

    try:
        hooks = MyRunHooks()
        hooks.clear_tool_calls()
        thread_local_data.current_hooks = hooks
        # 初始化工具结果存储
        if not hasattr(thread_local_data, 'tool_results'):
            thread_local_data.tool_results = []

        # 构造上下文消息并发送至Agent
        context_message = _build_context_message(user_message)
        result = await Runner.run(current_agent, context_message, hooks=hooks)
        response_data = {
            "message": result.final_output,
            "tool_calls": hooks.tool_calls,
            "tool_results": getattr(thread_local_data, 'tool_results', []),
            "task_plan": {
                "steps": hooks.task_plan,
                "current_step": hooks.current_step,
                "total_steps": len(hooks.task_plan)
            },
            "timestamp": datetime.now().isoformat()
        }

        # 保存对话历史
        conversation_history.append({
            "user_message": user_message,
            "assistant_response": result.final_output,
            "timestamp": datetime.now().isoformat(),
            "tool_calls_count": len(hooks.tool_calls)
        })
        return jsonify(response_data)
    except Exception as e:
        # 针对常见错误类型提供特定错误信息
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
    finally:
        # 清除线程中的hook信息和工具结果
        if hasattr(thread_local_data, 'current_hooks'):
            thread_local_data.current_hooks = None
        if hasattr(thread_local_data, 'tool_results'):
            thread_local_data.tool_results = []

@app.route('/api/chat/stream', methods=['POST'])
@async_route
async def chat_stream():
    """
    流式聊天接口，支持实时显示任务规划和执行过程
    """
    if not current_agent:
        return jsonify({"error": "请先配置API设置"})
    data = request.json
    user_message = data.get('message')
    if not user_message or not user_message.strip():
        return jsonify({"error": "请输入有效的问题"})
    if len(user_message) > 10000:
        return jsonify({"error": "消息长度过长，请控制在10000字符以内"})

    try:
        hooks = MyRunHooks()
        hooks.clear_tool_calls()
        thread_local_data.current_hooks = hooks
        # 初始化工具结果和规划存储
        if not hasattr(thread_local_data, 'tool_results'):
            thread_local_data.tool_results = []
        if not hasattr(thread_local_data, 'tool_planning'):
            thread_local_data.tool_planning = []

        # 构造上下文消息
        context_message = _build_context_message(user_message)
        
        # 这里需要修改为支持流式处理
        # 由于agents库可能不支持原生流式，我们先模拟流式效果
        result = await Runner.run(current_agent, context_message, hooks=hooks)
        
        # 模拟流式响应
        response_data = {
            "message": result.final_output,
            "tool_calls": hooks.tool_calls,
            "tool_results": getattr(thread_local_data, 'tool_results', []),
            "tool_planning": getattr(thread_local_data, 'tool_planning', []),
            "task_plan": {
                "steps": hooks.task_plan,
                "current_step": hooks.current_step,
                "total_steps": len(hooks.task_plan)
            },
            "timestamp": datetime.now().isoformat()
        }

        # 保存对话历史
        conversation_history.append({
            "user_message": user_message,
            "assistant_response": result.final_output,
            "timestamp": datetime.now().isoformat(),
            "tool_calls_count": len(hooks.tool_calls)
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
    finally:
        if hasattr(thread_local_data, 'current_hooks'):
            thread_local_data.current_hooks = None
        if hasattr(thread_local_data, 'tool_results'):
            thread_local_data.tool_results = []

@app.route('/api/tools', methods=['GET'])
def get_tools():
    """
    获取当前注册的所有工具函数的名称和参数描述
    """
    tools = [
        {"name": "压气机几何设计", "description": "根据设计目标设计叶片几何参数", "parameters": {"flow_rate": "目标流量", "pressure_ratio": "目标压比", "efficiency": "目标效率"}},
        {"name": "压气机性能评估", "description": "评估叶片性能", "parameters": {"blade_height": "叶片高度", "blade_chord": "叶片弦长", "blade_angle": "叶片角度"}},
        {"name": "压气机性能优化", "description": "优化叶片设计", "parameters": {"target_flow_rate": "目标流量", "target_pressure_ratio": "目标压比", "target_efficiency": "目标效率", "initial_blade_height": "初始高度", "initial_blade_chord": "初始弦长", "initial_blade_angle": "初始角度"}},
        {"name": "叶片形状可视化","description": "绘制叶片轮廓图", "parameters": {"blade_height": "叶片高度", "blade_chord": "叶片弦长", "blade_angle": "叶片角度"}},
        {"name": "创建UG几何文件", "description": "构建实体几何", "parameters": {"blade_height": "叶片高度", "blade_chord": "叶片弦长", "blade_angle": "叶片角度"}},
        {"name": "前处理", "description": "构建网格", "parameters": {"file_path": "几何文件"}},
        {"name": "CFX求解", "description": "进行CFD求解","parameters": {"file_path": "def文件"}},
        {"name": "后处理", "description": "采用CFX POST后处理","parameters": {"file_path": "res文件"}}
    ]
    return jsonify(tools)

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    应用运行状态检查，返回当前Agent状态、缓存大小等信息
    """
    return jsonify({
        "status": "healthy",
        "agent_configured": current_agent is not None,
        "conversation_count": len(conversation_history),
        "cache_size": len(_calculation_cache),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    获取统计信息，例如对话历史数量和缓存使用情况
    """
    return jsonify({
        "conversation_history_size": len(conversation_history),
        "calculation_cache_size": len(_calculation_cache),
        "agent_configured": current_agent is not None
    })

@app.route('/api/clear-history', methods=['POST'])
def clear_conversation_history():
    """
    清除对话历史记录
    """
    conversation_history.clear()
    return jsonify({"success": True, "message": "对话历史已清除", "timestamp": datetime.now().isoformat()})

@app.route('/api/history', methods=['GET'])
def get_conversation_history():
    """
    获取最近的对话历史记录（最多100条）
    """
    limit = min(request.args.get('limit', 20, type=int), 100)
    recent = list(conversation_history)[-limit:]
    history = []
    for item in recent:
        if item.get('user_message'):
            history.append({"role": "user", "content": item['user_message'], "timestamp": item.get('timestamp')})
        if item.get('assistant_response'):
            history.append({"role": "assistant", "content": item['assistant_response'], "timestamp": item.get('timestamp')})
    return jsonify({"history": history, "total_count": len(conversation_history), "returned_count": len(history)})

# ------------------------- 启动 Flask 应用 ------------------------- #
if __name__ == '__main__':
    # host设为0.0.0.0，允许外部访问；debug=False表示生产环境运行
    print("Starting Flask on port 5000...")  # 确认执行到这里
    app.run(debug=True, host='0.0.0.0', port=4000)