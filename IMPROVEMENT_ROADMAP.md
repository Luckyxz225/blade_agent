# 项目改进路线图 (Project Improvement Roadmap)

**版本:** 1.0  
**生成日期:** 2025-11-17  
**项目:** blade_agent

---

## 📋 总览 (Overview)

本文档规划了 blade_agent 项目的技术改进方向，涵盖代码质量、性能优化、测试覆盖、文档完善等方面。

### 当前状态评分 (Current Status Score)

| 维度 | 评分 | 目标 |
|------|------|------|
| 代码质量 | 7/10 | 9/10 |
| 安全性 | 5/10 | 9/10 |
| 性能 | 6/10 | 8/10 |
| 测试覆盖 | 2/10 | 8/10 |
| 文档 | 7/10 | 9/10 |
| 可维护性 | 7/10 | 9/10 |

---

## 🎯 第一阶段：基础改进 (Phase 1: Foundation - Month 1-2)

### 1.1 代码质量提升 (Code Quality)

#### 1.1.1 降低代码复杂度

**目标:** 将圈复杂度 > 10 的函数降至 < 8

**任务清单:**
- [ ] 重构 `result_synthesis_node` (langgraph_nodes.py:431)
  - 当前复杂度: 14 (C)
  - 拆分为 3 个子函数
  - 估算工作量: 3 小时
  
- [ ] 重构 `chat_stream` (web_app_v3.py:338)
  - 当前复杂度: 18 (C)
  - 使用策略模式简化条件分支
  - 估算工作量: 4 小时
  
- [ ] 重构 `chat` 函数 (web_app_v3.py:272)
  - 当前复杂度: 17 (C)
  - 提取错误处理为独立函数
  - 估算工作量: 3 小时

**期望成果:**
- 平均圈复杂度从 5.2 降至 4.0
- 代码可读性提升 30%

#### 1.1.2 消除代码重复

**识别的重复代码:**
```python
# geometry_generate.py 和 interp_all_shape.py
# 重复的 token 处理逻辑 (~20 行)
```

**重构方案:**
```python
# 新建 geometry_generate/utils.py
def normalize_tokens(line: str, max_widths: list) -> list:
    """标准化数值 token"""
    numbers = []
    for i, token in enumerate(line.split()):
        if token in ('None', 'nan'):
            token = ' '
        else:
            try:
                num = float(token)
                token = str(int(num)) if num.is_integer() else f"{num:.7f}"
            except ValueError:
                pass
        max_widths[i] = max(max_widths[i], len(token))
        numbers.append(token)
    return numbers
```

**估算工作量:** 2 小时

#### 1.1.3 添加类型注解

**当前状态:** ~60% 函数有类型注解

**目标:** 100% 核心函数有完整类型注解

**任务:**
- [ ] 添加类型注解到所有公共函数
- [ ] 使用 `mypy` 进行类型检查
- [ ] 配置 pre-commit hook 自动检查

**估算工作量:** 8 小时

---

### 1.2 安全加固 (Security Hardening)

详见 [SECURITY_FIXES.md](./SECURITY_FIXES.md)

**关键任务:**
- [x] 修复 PyTorch 不安全加载 (P0)
- [x] 移除硬编码密钥 (P0)
- [ ] 修复命令注入风险 (P1)
- [ ] 添加 CSRF 保护 (P1)
- [ ] 实施 API 速率限制 (P2)

**总估算工作量:** 15-20 小时

---

### 1.3 测试框架建立 (Testing Framework)

#### 1.3.1 单元测试

**目标:** 核心模块测试覆盖率 > 70%

**测试结构:**
```
tests/
├── unit/
│   ├── test_langgraph_nodes.py
│   ├── test_langgraph_workflow.py
│   ├── test_tools.py
│   ├── test_models.py
│   └── test_geometry.py
├── integration/
│   ├── test_api_endpoints.py
│   ├── test_workflow_e2e.py
│   └── test_model_inference.py
├── fixtures/
│   ├── mock_models.py
│   └── sample_data.json
└── conftest.py
```

**示例测试:**
```python
# tests/unit/test_langgraph_nodes.py
import pytest
from unittest.mock import Mock, patch
from langgraph_nodes import intent_recognition_node

def test_intent_recognition_design():
    """测试设计意图识别"""
    state = {
        "user_input": "设计一个流量15kg/s的叶片",
        "messages": []
    }
    
    with patch('langgraph_nodes.create_llm') as mock_llm:
        mock_llm.return_value.invoke.return_value.content = '{"intent": "design", ...}'
        result = intent_recognition_node(state)
    
    assert result["intent"] == "design"
    assert result["performance_targets"]["mass_flow"] == 15.0

def test_intent_recognition_invalid_input():
    """测试无效输入处理"""
    state = {
        "user_input": "随机文本",
        "messages": []
    }
    
    result = intent_recognition_node(state)
    assert result["intent"] == "unknown"
```

**估算工作量:** 30-40 小时

#### 1.3.2 集成测试

**API 端点测试:**
```python
# tests/integration/test_api_endpoints.py
import pytest
from web_app_langgraph import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """测试健康检查端点"""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_chat_endpoint(client):
    """测试对话端点"""
    response = client.post('/api/chat/stream', json={
        "message": "设计一个叶片"
    })
    assert response.status_code == 200
```

**估算工作量:** 15-20 小时

#### 1.3.3 性能测试

**基准测试:**
```python
# tests/performance/test_model_inference.py
import pytest
import time
from generative_design.compressor_design import blade_design

def test_design_generation_performance():
    """测试设计生成性能"""
    start = time.time()
    result = blade_design(
        flow_rate=15.0,
        efficiency=0.85,
        pressure_ratio=1.5
    )
    duration = time.time() - start
    
    assert duration < 5.0, f"设计生成耗时过长: {duration}s"
    assert result['status'] == 'success'

@pytest.mark.benchmark
def test_concurrent_requests(benchmark):
    """测试并发性能"""
    def run_design():
        return blade_design(15.0, 0.85, 1.5)
    
    result = benchmark(run_design)
    assert result['status'] == 'success'
```

**估算工作量:** 10 小时

---

## 🚀 第二阶段：性能优化 (Phase 2: Performance - Month 3-4)

### 2.1 模型推理优化

#### 2.1.1 模型缓存

**当前问题:** 每次请求都重新加载模型

**优化方案:**
```python
# 创建 generative_design/model_cache.py
from functools import lru_cache
import torch

class ModelCache:
    """全局模型缓存单例"""
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_or_load(self, model_name: str, loader_func):
        """获取或加载模型"""
        if model_name not in self._models:
            print(f"[缓存] 首次加载模型: {model_name}")
            self._models[model_name] = loader_func()
        return self._models[model_name]
    
    def clear(self):
        """清空缓存"""
        self._models.clear()
        torch.cuda.empty_cache()

# 使用
cache = ModelCache()
model = cache.get_or_load('unet', load_unet_model)
```

**期望提升:** 首次加载后响应时间减少 80%

**估算工作量:** 4 小时

#### 2.1.2 批量推理

**优化方案:**
```python
def batch_inference(models: list, batch_size: int = 8):
    """批量推理优化"""
    results = []
    for i in range(0, len(models), batch_size):
        batch = models[i:i+batch_size]
        # GPU 批量推理
        batch_results = model.predict(torch.stack(batch))
        results.extend(batch_results)
    return results
```

**期望提升:** 吞吐量提升 3-5x

**估算工作量:** 6 小时

#### 2.1.3 模型量化

**方案:**
```python
import torch.quantization as quantization

# 动态量化（易于实施）
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 或静态量化（更高性能）
model.qconfig = quantization.get_default_qconfig('fbgemm')
quantization.prepare(model, inplace=True)
# 校准...
quantization.convert(model, inplace=True)
```

**期望提升:** 推理速度提升 2-4x, 模型大小减少 75%

**估算工作量:** 12 小时

---

### 2.2 缓存策略

#### 2.2.1 Redis 结果缓存

**架构:**
```
用户请求 → 检查缓存 → 缓存命中？
                          ↓ 否
                       模型推理 → 存入缓存 → 返回结果
                          ↓ 是
                       返回缓存结果
```

**实现:**
```python
import redis
import hashlib
import json

class ResultCache:
    def __init__(self, redis_url='redis://localhost:6379'):
        self.redis = redis.from_url(redis_url)
    
    def get_cache_key(self, params: dict) -> str:
        """生成缓存键"""
        param_str = json.dumps(params, sort_keys=True)
        return f"design:{hashlib.md5(param_str.encode()).hexdigest()}"
    
    def get(self, params: dict):
        """获取缓存"""
        key = self.get_cache_key(params)
        cached = self.redis.get(key)
        return json.loads(cached) if cached else None
    
    def set(self, params: dict, result: dict, ttl=3600):
        """设置缓存"""
        key = self.get_cache_key(params)
        self.redis.setex(key, ttl, json.dumps(result))

# 使用
cache = ResultCache()
result = cache.get(params)
if not result:
    result = blade_design(**params)
    cache.set(params, result)
```

**期望提升:** 重复请求响应时间减少 95%

**估算工作量:** 6 小时

---

### 2.3 异步处理

#### 2.3.1 异步 API

**改造为异步:**
```python
from flask import Flask
from flask_async import async_route

@app.route('/api/chat/stream', methods=['POST'])
@async_route
async def chat_stream():
    """异步对话端点"""
    message = request.json['message']
    
    # 异步调用
    result = await asyncio.to_thread(
        process_message, message
    )
    
    return jsonify(result)
```

**期望提升:** 并发处理能力提升 200%

**估算工作量:** 10 小时

---

## 📚 第三阶段：文档与工程化 (Phase 3: Documentation - Month 5)

### 3.1 API 文档

#### 3.1.1 OpenAPI/Swagger

**任务:**
- [ ] 添加 flask-swagger-ui
- [ ] 编写 OpenAPI 规范
- [ ] 生成交互式文档

**示例:**
```python
from flask_swagger_ui import get_swaggerui_blueprint

SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Blade Agent API"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
```

**估算工作量:** 8 小时

---

### 3.2 开发者指南

**需要创建的文档:**
- [ ] CONTRIBUTING.md - 贡献指南
- [ ] DEVELOPMENT.md - 开发环境搭建
- [ ] DEPLOYMENT.md - 部署指南
- [ ] CHANGELOG.md - 变更日志

**估算工作量:** 12 小时

---

### 3.3 CI/CD 流程

#### 3.3.1 GitHub Actions

**工作流配置:**
```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.10
      
      - name: Install dependencies
        run: |
          pip install -r requirements_langgraph.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ --cov=./ --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
  
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run linters
        run: |
          pip install black flake8 mypy
          black --check .
          flake8 .
          mypy .
  
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run security scan
        run: |
          pip install bandit safety
          bandit -r .
          safety check
```

**估算工作量:** 6 hours

---

## 🔧 第四阶段：工具与监控 (Phase 4: Tooling - Month 6)

### 4.1 监控系统

#### 4.1.1 Prometheus 指标

**实现:**
```python
from prometheus_client import Counter, Histogram, generate_latest

# 定义指标
request_count = Counter('blade_agent_requests_total', 'Total requests')
request_duration = Histogram('blade_agent_request_duration_seconds', 'Request duration')
model_inference_time = Histogram('blade_agent_model_inference_seconds', 'Model inference time')

@app.route('/metrics')
def metrics():
    return generate_latest()

# 使用
@request_duration.time()
def process_request():
    request_count.inc()
    # ...
```

**估算工作量:** 8 小时

#### 4.1.2 日志聚合

**ELK Stack 集成:**
```python
import logging
from pythonjsonlogger import jsonlogger

# 配置 JSON 日志
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
```

**估算工作量:** 6 小时

---

### 4.2 容器化

#### 4.2.1 Docker 镜像

**多阶段构建:**
```dockerfile
# Dockerfile
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements*.txt ./
RUN pip install --user --no-cache-dir -r requirements_langgraph.txt

FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
EXPOSE 4000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:4000", "web_app_langgraph:app"]
```

**估算工作量:** 4 小时

#### 4.2.2 Docker Compose

**本地开发环境:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "4000:4000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

**估算工作量:** 3 小时

---

## 📊 进度跟踪 (Progress Tracking)

### 里程碑 (Milestones)

| 里程碑 | 目标日期 | 状态 | 完成度 |
|--------|---------|------|--------|
| M1: 基础改进 | Month 2 | 🟡 进行中 | 10% |
| M2: 性能优化 | Month 4 | ⚪ 未开始 | 0% |
| M3: 文档完善 | Month 5 | ⚪ 未开始 | 0% |
| M4: 工具监控 | Month 6 | ⚪ 未开始 | 0% |

### 工作量统计 (Effort Estimation)

| 阶段 | 任务数 | 总工作量 | 优先级 |
|------|--------|---------|--------|
| Phase 1 | 15 | 75-95h | P0 |
| Phase 2 | 8 | 45-60h | P1 |
| Phase 3 | 6 | 26h | P2 |
| Phase 4 | 5 | 21h | P3 |
| **总计** | **34** | **167-202h** | - |

---

## ✅ 验收标准 (Acceptance Criteria)

### Phase 1 完成标准:
- [x] 所有 P0 安全问题已修复
- [ ] 核心模块测试覆盖率 > 70%
- [ ] 圈复杂度 > 10 的函数 < 5 个
- [ ] 所有公共函数有类型注解

### Phase 2 完成标准:
- [ ] 模型加载时间 < 1s（首次加载后）
- [ ] API 响应时间 < 2s（95th percentile）
- [ ] 支持 100 并发用户

### Phase 3 完成标准:
- [ ] API 文档完整且可访问
- [ ] CI/CD 流程自动化
- [ ] 代码覆盖率 badge 显示

### Phase 4 完成标准:
- [ ] Prometheus 指标可用
- [ ] Docker 镜像构建成功
- [ ] 日志可查询和分析

---

## 📞 联系与反馈 (Contact & Feedback)

如有问题或建议，请：
1. 创建 GitHub Issue
2. 联系项目维护者
3. 查看项目 Wiki

**最后更新:** 2025-11-17
