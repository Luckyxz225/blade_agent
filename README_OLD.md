# 智能Agent叶片设计系统

## 项目简介

基于大语言模型的智能Agent系统，专门用于压气机叶片设计、性能评估和优化。提供Web界面和命令行两种交互方式。

## 功能特点

- **Web界面** - Flask + Bootstrap的可视化界面
- **AI对话** - 支持多轮对话和工具调用
- **叶片设计** - 自动设计、性能评估、优化工具
- **多Agent** - 支持单Agent和多Agent协作模式

## 技术栈

- Flask + agents-py + litellm
- Bootstrap + JavaScript
- 支持OpenAI、硅基流动等LLM

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量（创建.env文件）
SILICON_API_KEY2=your_api_key
SILICON_BASE_URL=https://api.siliconflow.cn/v1

# 启动Web应用
python web_app.py
# 访问 http://localhost:5000

# 或运行Demo
python demo/demo_FuncCallAgent.py
```

## 主要文件

- `web_app.py` - Flask主应用，包含叶片设计工具
- `demo/demo_FuncCallAgent.py` - 单Agent工具调用示例  
- `demo/demo_MultiAgent.py` - 多Agent协作示例
- `templates/index.html` - Web界面
- `requirements.txt` - 依赖包列表

## 使用示例

```
用户: 请设计一个流量10kg/s的叶片
系统: [调用 blade_design 工具] → 返回几何参数

用户: 评估这个叶片的性能
系统: [调用 blade_performance_evaluation 工具] → 返回性能指标
```