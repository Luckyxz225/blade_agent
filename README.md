# 使用说明

## 一、环境配置

### 1.1 依赖安装

**方式一：一键安装（推荐）**
```bash
# 安装所有依赖
pip3 install -r requirements_langgraph.txt
pip3 install -r requirements_original_modules.txt
```

**方式二：Conda环境**
```bash
# 创建环境（Python 3.10）
conda create -n blade_agent python=3.10
conda activate blade_agent

# 安装依赖
pip install -r requirements_langgraph.txt
pip install -r requirements_original_modules.txt
```

### 1.2 API配置

**环境变量方式**（推荐）:
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 或其他兼容API
export MODEL_NAME="gpt-4"  # 或 deepseek-chat 等
```

**前端配置方式**:
1. 启动服务后访问前端
2. 在配置面板输入API Key和Base URL
3. 点击"测试连接"确认配置

## 二、启动服务

### 2.1 启动命令
```bash
# 确保在项目根目录
cd /Users/liuyuze/Desktop/myAgent-main_V2_副本/myAgent-main副本

# 激活环境（如使用conda）
conda activate blade_agent

# 启动Web服务
python web_app_langgraph.py
```

### 2.2 访问地址
- **Web界面**: http://localhost:4000
- **API端点**: http://localhost:4000/api/chat/stream

### 2.3 停止服务
```bash
# 查找进程
lsof -i:4000

# 强制终止
kill -9 <PID>
```

## 三、使用示例

### 3.1 叶片设计
**用户输入**:
```
设计一个流量15kg/s、效率0.85、压比1.5的叶片
```

**系统输出**:
- 21维设计参数
- 预测性能指标
- 3D可视化图像

**支持的输入形式**:
- "流量15、效率0.85、压比1.5的叶片"
- "我需要一个压比1.5的设计"（自动补充其他参数）
- "设计一个高效率叶片"（全部使用默认值）

### 3.2 性能评估
**用户输入**:
```
评估这个设计的性能：参数1=xxx, 参数2=xxx, ...（共21个）
```

**系统输出**:
- 流量预测值
- 效率预测值
- 压比预测值

**注意**: 如果参数不足21个，系统会自动补充默认值

### 3.3 可视化
**用户输入**:
```
显示叶片形状
```
或在设计/评估后追问：
```
画出这个叶片
```

**系统输出**:
- 3D几何图像
- 图像访问链接

### 3.4 功能询问
**用户输入**:
```
你能做什么？
```
或：
```
介绍一下你自己
```

**系统输出**:
- 系统功能列表
- 使用示例

### 3.5 闲聊对话
**用户输入**:
```
今天天气怎么样？
```

**系统输出**:
- 简短回应
- 引导回到叶片设计主题

## 四、参数说明

### 4.1 性能目标参数
| 参数名 | 单位 | 默认值 | 说明 |
|--------|------|--------|------|
| 流量 | kg/s | 15 | 空气质量流量 |
| 效率 | - | 0.85 | 绝热效率（0-1） |
| 压比 | - | 1.5 | 总压比 |

### 4.2 设计参数
21维向量，包括：
- 叶根、叶中、叶尖截面参数
- 进出口角度
- 厚度、弦长等几何参数

**注意**: 用户无需了解21维参数的具体含义，系统会自动处理

## 五、常见问题

### 5.1 依赖冲突
**问题**: `langchain-core version incompatible`

**解决**:
```bash
pip uninstall langchain langchain-core langchain-openai langgraph -y
pip install -r requirements_langgraph.txt --force-reinstall
```

### 5.2 PyTorch CUDA错误
**问题**: `Torch not compiled with CUDA enabled`

**解决**: 系统已自动配置CPU模式，无需手动处理

### 5.3 端口占用
**问题**: `Port 4000 is in use`

**解决**:
```bash
lsof -i:4000
kill -9 <PID>
```

### 5.4 模型文件未找到
**问题**: `No such file or directory: .../prior_parameters_guide_15200.pth`

**解决**: 确保 `all_file_path/` 目录包含以下文件：
- `prior_parameters_guide_15200.pth`
- `pre_guide_15200.pth`
- `best_model_in_testdataset.pth`
- `parameter_geometry_5200.npy`
- `label_5200.npy`

### 5.5 API超时
**问题**: LLM调用超时或无响应

**解决**:
1. 检查网络连接和API密钥
2. 检查Base URL是否正确
3. 尝试更换模型（如deepseek-chat更快）
4. 查看终端日志定位问题节点

### 5.6 返回结果为空
**问题**: 用户输入后系统无响应

**解决**:
1. 查看终端日志中的错误信息
2. 确认API配置正确
3. 检查模型文件完整性
4. 尝试简化输入（如只输入"设计叶片"）

## 六、API接口

### 6.1 配置API
**端点**: `POST /api/config`

**请求体**:
```json
{
  "api_key": "your-api-key",
  "base_url": "https://api.openai.com/v1",
  "model_name": "gpt-4"
}
```

**响应**:
```json
{
  "status": "success",
  "message": "API configured successfully"
}
```

### 6.2 对话API
**端点**: `POST /api/chat/stream`

**请求体**:
```json
{
  "message": "设计一个流量15kg/s的叶片"
}
```

**响应**:
```json
{
  "response": "...",
  "image_path": "/static/images/blade3d_xxx.png"
}
```

## 七、日志与调试

### 7.1 查看日志
系统在终端输出详细日志，关键日志标签：

| 标签 | 说明 |
|------|------|
| `[意图识别]` | 用户意图判断过程 |
| `[设计生成]` | 叶片设计执行状态 |
| `[性能评估]` | 性能预测执行状态 |
| `[可视化]` | 图像生成状态 |
| `[Unknown处理]` | 闲聊/功能询问处理 |
| `[Error]` | 错误信息 |

### 7.2 调试技巧
1. **启用详细日志**: 终端会自动输出每个节点的执行状态
2. **查看状态**: 检查日志中的 `state` 内容
3. **单独测试工具**: 使用 `test_tools_simple.py` 测试模型推理
4. **API测试**: 使用curl测试API端点

### 7.3 测试脚本
```bash
# 测试工具函数
python test_tools_simple.py

# 测试LangGraph工作流
python langgraph_tools.py
```

## 八、性能优化

### 8.1 加速建议
1. **使用更快的模型**: DeepSeek速度优于GPT-4
2. **启用GPU**: 如有NVIDIA GPU，安装CUDA版PyTorch
3. **减少迭代次数**: 修改 `langgraph_config.py` 中的 `MAX_OPTIMIZATION_ITERATIONS`
4. **简化可视化**: 跳过图像生成步骤

### 8.2 资源要求
| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 4核 | 8核+ |
| 内存 | 8GB | 16GB+ |
| 磁盘 | 2GB | 5GB+ |
| GPU | 无 | NVIDIA 4GB+ |

## 九、开发指南

### 9.1 添加新功能节点
1. 在 `langgraph_nodes.py` 定义节点函数
2. 在 `langgraph_workflow.py` 添加节点和边
3. 在 `langgraph_config.py` 添加必要的状态字段

### 9.2 修改默认参数
编辑 `langgraph_config.py` 中的常量：
```python
DEFAULT_PERFORMANCE_TARGETS = {
    "mass_flow": 15.0,  # 修改默认流量
    "efficiency": 0.85,
    "pressure_ratio": 1.5
}
```

### 9.3 更换LLM模型
修改环境变量或在 `langgraph_nodes.py` 中调整 `create_llm()` 函数的默认值

## 十、技术支持

### 10.1 常见错误代码
| 错误 | 原因 | 解决方法 |
|------|------|---------|
| 400 | API请求错误 | 检查API Key和模型名称 |
| 401 | 认证失败 | 检查API Key |
| 429 | 请求过多 | 等待后重试 |
| 500 | 服务器错误 | 查看终端日志 |

### 10.2 系统要求
- Python: 3.8 - 3.11（推荐3.10）
- 操作系统: macOS, Linux, Windows
- 网络: 需访问OpenAI兼容API

