"""
LangGraph 多智能体 Web 应用
===========================

基于 Flask 的 Web 接口，支持：
1. 多Agent协作
2. 反问用户（暂停/恢复）
3. Agent协作日志查看
"""

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_multi_agent_config import MultiAgentState, AgentNames
from langgraph_multi_agent_workflow import create_multi_agent_app
from langgraph_multi_agent_nodes import set_task_result_callback  # 新增：任务结果回调
from dotenv import load_dotenv
import os
import json
import queue
import threading

# 加载环境变量
load_dotenv()

app = Flask(__name__)
CORS(app)

# ==================== 全局变量 ====================

# 当前工作流
current_workflow = None

# API配置状态（是否已初始化）
api_initialized = False


def init_api_from_env():
    """从环境变量初始化API配置"""
    global api_initialized, current_workflow
    
    # 从.env读取配置
    api_key = os.getenv('API_KEY') or os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('BASE_URL') or os.getenv('OPENAI_BASE_URL')
    model_name = os.getenv('MODEL_NAME') or os.getenv('OPENAI_MODEL_NAME')
    
    if not all([api_key, base_url, model_name]):
        print("[Web] ⚠️ .env 中未找到完整的API配置，需要前端配置")
        return False
    
    # 设置环境变量（供 LangChain 使用）
    os.environ['OPENAI_API_KEY'] = api_key
    os.environ['OPENAI_BASE_URL'] = base_url
    os.environ['OPENAI_API_BASE'] = base_url
    os.environ['OPENAI_MODEL_NAME'] = model_name
    
    print(f"[Web] ✅ 从 .env 加载 API 配置：")
    print(f"  - API Key: {api_key[:20]}...{api_key[-4:]}")
    print(f"  - Base URL: {base_url}")
    print(f"  - Model: {model_name}")
    
    # 初始化工作流
    try:
        current_workflow = create_multi_agent_app()
        api_initialized = True
        print("[Web] ✅ 工作流初始化成功，可以直接对话")
        return True
    except Exception as e:
        print(f"[Web] ❌ 工作流初始化失败: {e}")
        return False

# 会话管理（简化版，实际应用应使用Redis等）
# 格式：{session_id: {"state": MultiAgentState, "workflow": app}}
sessions = {}

# 对话历史（用于展示）
conversation_history = []

# 进度队列管理（用于SSE推送）
# 格式：{session_id: queue.Queue()}
progress_queues = {}

# 进度队列锁
progress_lock = threading.Lock()


# ==================== 工具函数 ====================

def get_progress_queue(session_id: str) -> queue.Queue:
    """获取或创建进度队列"""
    with progress_lock:
        if session_id not in progress_queues:
            progress_queues[session_id] = queue.Queue()
        return progress_queues[session_id]


def send_progress(session_id: str, progress_data: dict):
    """发送进度更新到指定会话"""
    try:
        q = get_progress_queue(session_id)
        q.put(json.dumps(progress_data))
    except Exception as e:
        print(f"[进度推送] 失败: {e}")


def clear_progress_queue(session_id: str):
    """清除进度队列（清空而不是删除，保持引用有效）"""
    with progress_lock:
        if session_id in progress_queues:
            # 清空队列而不是删除，这样SSE端点仍然可以使用同一个队列
            try:
                while not progress_queues[session_id].empty():
                    progress_queues[session_id].get_nowait()
            except:
                pass


def extract_final_message(state: MultiAgentState) -> str:
    """从状态中提取最终消息"""
    
    if state.get("final_result"):
        return state["final_result"]
    
    # 只提取AI的消息，不要提取用户的消息
    messages = state.get("messages", [])
    if messages:
        # 从后往前找，找到第一个AIMessage
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message.content
    
    return "处理完成"


def get_or_create_session(session_id: str) -> dict:
    """获取或创建会话"""
    if session_id not in sessions:
        sessions[session_id] = {
            "workflow": create_multi_agent_app(),
            "state": None
        }
    return sessions[session_id]


# ==================== API 路由 ====================

@app.route('/')
def index():
    """首页"""
    return send_from_directory('templates', 'index_multi_agent.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """静态文件服务"""
    return send_from_directory('static', filename)


@app.route('/api/config', methods=['POST'])
def update_config():
    """
    配置API（与单智能体逻辑一致）
    
    前端发送API配置后，初始化工作流
    """
    global current_workflow
    
    try:
        data = request.json
        
        # 获取配置（与单智能体一致）
        api_key = data.get('api_key') or os.getenv('API_KEY')
        base_url = data.get('base_url') or os.getenv('BASE_URL')
        model_name = data.get('model_name') or os.getenv('MODEL_NAME')
        
        # 验证必填项
        if not all([api_key, base_url, model_name]):
            return jsonify({
                "success": False,
                "error": "需通过前端输入或.env文件配置API_KEY/BASE_URL/MODEL_NAME"
            })
        
        # 验证URL格式
        if not base_url.startswith(('http://', 'https://')):
            return jsonify({"success": False, "error": "基础URL必须以http://或https://开头"})
        
        # 设置环境变量（与单智能体完全一致）
        os.environ['OPENAI_API_KEY'] = api_key
        os.environ['OPENAI_BASE_URL'] = base_url
        os.environ['OPENAI_API_BASE'] = base_url  # 兼容旧键名
        os.environ['OPENAI_MODEL_NAME'] = model_name
        
        print(f"[Web] 配置信息：")
        print(f"  - API Key: {api_key[:20]}...{api_key[-4:]}")
        print(f"  - Base URL: {base_url}")
        print(f"  - Model: {model_name}")
        
        # 测试连接（与单智能体一致）
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        test_llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0
        )
        print(f"[Web] 测试API连接...")
        test_result = test_llm.invoke([HumanMessage(content="测试连接")])
        
        if not test_result or not test_result.content:
            raise Exception("API测试调用失败：未能获取有效响应")
        
        print(f"[Web] API连接测试成功！")
        
        # 创建工作流
        current_workflow = create_multi_agent_app()
        
        # 清空所有会话
        sessions.clear()
        
        print(f"[Web] 多智能体工作流已初始化")
        
        return jsonify({
            "success": True,
            "message": f"配置更新成功！已连接到模型: {model_name}",
            "framework": "LangGraph-MultiAgent"
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        current_workflow = None
        return jsonify({
            "success": False,
            "error": f"配置失败: {str(e)}"
        })


@app.route('/api/chat', methods=['POST'])
@app.route('/api/chat/stream', methods=['POST'])
def chat():
    """
    多智能体聊天接口
    
    支持：
    1. 正常对话
    2. 反问用户（返回question，等待用户回答）
    3. 恢复执行（用户回答后继续）
    """
    
    if not current_workflow:
        return jsonify({"error": "请先配置API设置"})
    
    data = request.json
    user_message = data.get('message')
    session_id = data.get('session_id', 'default')
    
    if not user_message or not user_message.strip():
        return jsonify({"error": "请输入有效的问题"})
    
    try:
        # 清除旧的进度队列，确保每次都是新的开始
        print(f"[Web] 清除旧的进度队列 (session: {session_id})")
        clear_progress_queue(session_id)
        
        # 设置进度回调（用于实时推送进度）
        from Tools.blade_design import set_progress_callback
        from Tools.blade_optimization import set_optimization_progress_callback
        from Tools.blade_visualization import set_visualization_progress_callback
        
        def progress_callback(progress_data):
            """进度回调函数：推送进度到前端"""
            print(f"[进度推送] {progress_data}")
            
            # ✅ 根据stage判断消息类型
            # 注意：design_complete 和 optimization_complete 是子任务完成，不应关闭 SSE
            # 只有 all_complete 才是整个请求完成
            stage = progress_data.get('stage', '')
            if stage == 'all_complete':
                msg_type = 'complete'  # 只有整体完成才发送 complete 类型
            elif stage == 'error':
                msg_type = 'error'
            else:
                msg_type = 'progress'  # 子任务完成也是 progress 类型
            
            # 构建消息（包含所有字段）
            msg = {
                'type': msg_type,
                'stage': stage,
                'progress': progress_data.get('progress', 0),
                'message': progress_data.get('message', '')
            }
            
            # ✅ 如果是设计中间图片，添加 image_path 和 label
            if stage == 'design_image':
                msg['image_path'] = progress_data.get('image_path', '')
                msg['label'] = progress_data.get('label', '')
            
            # ✅ 如果是优化中间图片，添加 image_path 和 label
            if stage == 'optimization_image':
                msg['image_path'] = progress_data.get('image_path', '')
                msg['label'] = progress_data.get('label', '')
            
            # ✅ 如果是可视化图片，添加 image_path、label 和 json_path
            if stage == 'visualization_image':
                msg['image_path'] = progress_data.get('image_path', '')
                msg['label'] = progress_data.get('label', '')
                msg['json_path'] = progress_data.get('json_path', '')
            
            # ✅ 【新增】如果是任务结果（多任务中间结果），添加所有相关字段
            if stage == 'task_result':
                msg['task'] = progress_data.get('task', '')
                msg['result'] = progress_data.get('result', '')
                msg['task_index'] = progress_data.get('task_index', 0)
                msg['total_tasks'] = progress_data.get('total_tasks', 0)
                if progress_data.get('image_path'):
                    msg['image_path'] = progress_data.get('image_path', '')
                if progress_data.get('json_path'):
                    msg['json_path'] = progress_data.get('json_path', '')
            
            send_progress(session_id, msg)
        
        set_progress_callback(progress_callback)
        set_optimization_progress_callback(progress_callback)  # 同时设置优化进度回调
        set_visualization_progress_callback(progress_callback)  # 同时设置可视化进度回调
        set_task_result_callback(progress_callback)  # 新增：任务结果回调（多任务实时输出）
        
        # 发送初始进度（测试SSE连接）
        send_progress(session_id, {
            'type': 'progress',
            'stage': 'init',
            'progress': 1,
            'message': '开始处理请求...'
        })
        print(f"[Web] 已发送初始进度信号")
        
        print(f"\n{'='*60}")
        print(f"[Web] 收到用户请求: {user_message[:50]}...")
        print(f"[Web] Session ID: {session_id}")
        api_key = os.getenv('OPENAI_API_KEY', 'None')
        if api_key != 'None':
            print(f"[Web] 当前 API Key: {api_key[:20]}...{api_key[-4:]}")
        else:
            print(f"[Web] 当前 API Key: 未设置")
        print(f"[Web] 当前 Base URL: {os.getenv('OPENAI_BASE_URL', 'None')}")
        print(f"[Web] 当前 Model: {os.getenv('OPENAI_MODEL_NAME', 'None')}")
        print(f"{'='*60}")
        
        # 使用全局的 current_workflow
        workflow = current_workflow
        
        # 获取会话（用于保存状态和支持恢复执行）
        if session_id not in sessions:
            sessions[session_id] = {"state": None}
        session = sessions[session_id]
        previous_state = session.get("state")
        
        # 判断是新对话还是恢复对话
        if previous_state and previous_state.get("waiting_for_user"):
            # 恢复对话：用户回答了问题
            print(f"[Web] 恢复执行（用户已回答）")
            print(f"[Web] 之前等待的 Agent: {previous_state.get('pending_agent')}")
            
            # 将用户回答添加到消息历史
            previous_state["messages"].append(HumanMessage(content=user_message))
            
            # 清除等待状态
            previous_state["waiting_for_user"] = False
            previous_state["user_question"] = None
            
            # 更新 user_input 为用户的回答（让 Agent 重新解析）
            previous_state["user_input"] = user_message
            
            # 继续执行工作流
            final_state = workflow.invoke(previous_state)
        else:
            # 新对话：构建初始状态
            print(f"[Web] 开始新对话")
            
            # ==================== 关键修复：保留之前的任务结果 ====================
            # 从previous_state中保留关键结果数据，实现跨对话的结果传递
            preserved_task_results = {}
            preserved_design_params = None
            preserved_evaluation_results = None
            preserved_last_task = None
            preserved_last_task_time = None
            
            if previous_state:
                preserved_task_results = previous_state.get("task_results", {})
                preserved_design_params = previous_state.get("design_params")
                preserved_evaluation_results = previous_state.get("evaluation_results")
                preserved_last_task = previous_state.get("last_task")
                preserved_last_task_time = previous_state.get("last_task_time")
                
                print(f"[Web] 保留之前的任务结果:")
                print(f"  - task_results: {list(preserved_task_results.keys()) if preserved_task_results else '空'}")
                print(f"  - design_params: {'有' if preserved_design_params else '无'}")
                print(f"  - last_task: {preserved_last_task}")
            
            initial_state: MultiAgentState = {
                "user_input": user_message,
                "messages": [HumanMessage(content=user_message)],
                
                # 用户交互控制
                "waiting_for_user": False,
                "user_question": None,
                "pending_agent": None,
                
                # Agent协作
                "current_agent": AgentNames.COORDINATOR,
                "target_agent": None,
                "agent_messages": [],
                
                # 意图与参数
                "intent": "unknown",
                "performance_targets": {},  # 设计性能目标（每次新对话清空）
                "design_params": preserved_design_params,  # ✅ 保留设计结果
                "partial_blade_params": {},  # 累积的21维参数（每次新对话清空）
                
                # 执行结果
                "evaluation_results": preserved_evaluation_results,  # ✅ 保留评估结果
                "visualization_path": None,
                "visualization_json_path": None,  # 3D数据JSON路径
                
                # 优化相关
                "optimization_history": [],
                "iteration_count": 0,
                "max_iterations": 5,
                
                # 协作历史
                "collaboration_log": [],
                
                # 错误与流程控制
                "error_message": None,
                "next_action": None,
                "final_result": None,
                
                # ==================== 多任务编排 ====================
                "task_mode": "single",  # 每次新对话重置
                "task_queue": [],
                "current_task_index": 0,
                
                # ==================== 结果存储与传递（保留！）====================
                "task_results": preserved_task_results,  # ✅ 关键：保留任务结果
                "last_task": preserved_last_task,  # ✅ 保留上次任务
                "last_task_time": preserved_last_task_time,  # ✅ 保留时间戳
                
                # ==================== 意图细化 ====================
                "intent_type": "new",
                "use_existing_result": None,
                "awaiting_confirmation": False,
                
                # ==================== 多任务中间结果累积 ====================
                "accumulated_results": []
            }
            
            # 执行工作流（同步调用）
            final_state = workflow.invoke(initial_state)
        
        print(f"[Web] 工作流执行完成")
        
        # 保存会话状态（用于恢复执行）
        session["state"] = final_state
        
        # 检查是否需要用户输入
        if final_state.get("waiting_for_user"):
            print(f"[Web] 需要用户输入")
            print(f"[Web] 问题: {final_state.get('user_question', '')[:100]}...")
            
            response_data = {
                "message": final_state.get("user_question", "请提供更多信息"),
                "requires_user_input": True,
                "type": "question",
                "intent": final_state.get("intent", "unknown"),
                "pending_agent": final_state.get("pending_agent"),  # ✅ 告诉前端哪个agent在等待
                "timestamp": datetime.now().isoformat(),
                "workflow_status": "waiting_for_user",
                "collaboration_log": final_state.get("collaboration_log", [])
            }
            
            # 记录对话历史
            conversation_history.append({
                "user_message": user_message,
                "assistant_response": response_data["message"],
                "timestamp": datetime.now().isoformat(),
                "type": "question"
            })
            
            return jsonify(response_data)
        
        # 发送整体完成信号
        send_progress(session_id, {
            'type': 'complete',
            'stage': 'all_complete',
            'progress': 100,
            'message': '任务完成！'
        })
        
        # 正常完成：提取结果
        response_data = {
            "message": extract_final_message(final_state),
            "requires_user_input": False,
            "type": "response",
            "intent": final_state.get("intent", "unknown"),
            "design_params": final_state.get("design_params"),
            "evaluation_results": final_state.get("evaluation_results"),
            "visualization_path": final_state.get("visualization_path"),
            "visualization_json_path": final_state.get("visualization_json_path"),  # 3D数据路径
            "visualization_all_schemes": final_state.get("_visualization_all_schemes"),  # 多方案可视化路径列表
            "collaboration_log": final_state.get("collaboration_log", []),
            "agent_messages": final_state.get("agent_messages", []),
            "error": final_state.get("error_message"),
            "timestamp": datetime.now().isoformat(),
            "workflow_status": "completed"
        }
        
        # 发送整体完成信号
        print(f"[Web] 任务完成，发送all_complete信号")
        send_progress(session_id, {
            'type': 'complete',
            'stage': 'all_complete',
            'progress': 100,
            'message': '任务已完成'
        })
        
        # 记录对话历史
        conversation_history.append({
            "user_message": user_message,
            "assistant_response": response_data["message"],
            "timestamp": datetime.now().isoformat(),
            "intent": final_state.get("intent"),
            "type": "response"
        })
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # 发送错误信号
        send_progress(session_id, {
            'type': 'error',
            'stage': 'error',
            'progress': 0,
            'message': f'错误: {str(e)}'
        })
        
        return jsonify({
            "error": f"处理请求时出错: {str(e)}",
            "type": "error"
        })


@app.route('/api/progress/<session_id>')
def progress_stream(session_id):
    """SSE端点：实时推送任务进度"""
    def generate():
        q = get_progress_queue(session_id)
        try:
            while True:
                try:
                    # 等待进度更新，超时5秒发送心跳
                    progress_data = q.get(timeout=5)
                    yield f"data: {progress_data}\n\n"
                    
                    # 检查是否是完成信号
                    data = json.loads(progress_data)
                    if data.get('type') == 'complete' or data.get('type') == 'error':
                        break
                except queue.Empty:
                    # 发送心跳保持连接
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            # 清理队列
            clear_progress_queue(session_id)
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/api/history', methods=['GET'])
def get_history():
    """获取对话历史"""
    return jsonify({
        "history": conversation_history
    })


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """清空对话历史"""
    global conversation_history, sessions
    conversation_history = []
    sessions = {}
    return jsonify({"success": True, "message": "历史已清空"})


@app.route('/api/collaboration_log', methods=['GET'])
def get_collaboration_log():
    """
    获取Agent协作日志
    
    用于调试和可视化Agent协作过程
    """
    session_id = request.args.get('session_id', 'default')
    
    if session_id in sessions and sessions[session_id]["state"]:
        state = sessions[session_id]["state"]
        return jsonify({
            "collaboration_log": state.get("collaboration_log", []),
            "agent_messages": state.get("agent_messages", [])
        })
    
    return jsonify({
        "collaboration_log": [],
        "agent_messages": []
    })


# ==================== API状态检查端点 ====================

@app.route('/api/status', methods=['GET'])
def get_api_status():
    """检查API是否已初始化"""
    global api_initialized
    return jsonify({
        "initialized": api_initialized,
        "model": os.getenv('OPENAI_MODEL_NAME', 'unknown')
    })


# ==================== 主函数 ====================

if __name__ == '__main__':
    print("="*60)
    print("🚀 LangGraph 多智能体 Web 应用")
    print("="*60)
    
    # 尝试从.env自动初始化API
    init_api_from_env()
    
    print(f"访问地址: http://localhost:5002")
    print(f"特性: 多Agent协作 + 反问用户 + 协作日志")
    print("="*60)
    
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=True,
        threaded=True
    )

