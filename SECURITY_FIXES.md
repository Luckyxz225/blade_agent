# 安全修复建议 (Security Fix Recommendations)

**生成日期:** 2025-11-17  
**优先级:** 🔴 高 | 🟡 中 | 🟢 低

---

## 🔴 高优先级 - 立即修复 (Critical - Fix Immediately)

### 1. PyTorch 不安全模型加载 (Unsafe PyTorch Model Loading)

**问题描述:**
代码中多处使用 `weights_only=False` 加载 PyTorch 模型，存在反序列化漏洞风险（CWE-502）。攻击者可通过构造恶意模型文件执行任意代码。

**受影响文件:**
- `generative_design/blade_performance_evaluation.py:46`
- `generative_design/compressor_design.py:108`
- `generative_design/compressor_design.py:112`
- `generative_design/compressor_design.py:118`

**当前代码:**
```python
checkpoint = torch.load(
    os.path.join(_default_model_dir, 'best_model_in_testdataset.pth'),
    map_location=device,
    weights_only=False  # ⚠️ 不安全！
)
```

**修复方案:**
```python
# 方案 1: 使用 weights_only=True（推荐）
checkpoint = torch.load(
    os.path.join(_default_model_dir, 'best_model_in_testdataset.pth'),
    map_location=device,
    weights_only=True  # ✅ 安全
)

# 方案 2: 添加模型文件签名验证
import hashlib

def verify_model_checksum(file_path, expected_hash):
    """验证模型文件完整性"""
    with open(file_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return file_hash == expected_hash

# 使用前验证
if verify_model_checksum(model_path, EXPECTED_HASH):
    checkpoint = torch.load(model_path, map_location=device)
else:
    raise SecurityError("模型文件校验失败")
```

**影响:**
- 需要重新保存所有模型文件（使用 `torch.save(model.state_dict(), path)` 格式）
- 如果模型文件包含优化器状态或其他非权重数据，需要调整加载逻辑

**估算工作量:** 2-3 小时

---

### 2. 硬编码密钥 (Hardcoded Secret Key)

**问题描述:**
Flask 应用使用硬编码的默认密钥，容易被泄露。

**受影响文件:**
- `web_app_v3.py:27`

**当前代码:**
```python
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
```

**修复方案:**
```python
# 方案 1: 强制要求环境变量（推荐）
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    raise ValueError("必须设置 SECRET_KEY 环境变量")

# 方案 2: 生产环境检查
if os.getenv('FLASK_ENV') == 'production':
    app.secret_key = os.getenv('SECRET_KEY')
    if not app.secret_key:
        raise ValueError("生产环境必须设置 SECRET_KEY")
else:
    app.secret_key = os.getenv('SECRET_KEY', os.urandom(24).hex())
```

**影响:**
- 部署时必须设置 SECRET_KEY 环境变量
- 现有会话可能失效（需要用户重新登录）

**估算工作量:** 0.5 小时

---

### 3. 命令注入风险 (Command Injection Risk)

**问题描述:**
使用 `subprocess.run()` 执行外部命令，存在命令注入风险。

**受影响文件:**
- `geometry_generate/geometry_generate.py:163`
- `geometry_generate/cfx_def.py:22`

**当前代码:**
```python
subprocess.run([exe_path, arg1, arg2])  # 如果参数未验证，存在风险
```

**修复方案:**
```python
import shlex

def safe_subprocess_run(cmd_list, allowed_commands=None):
    """安全的子进程执行"""
    # 1. 验证可执行文件路径
    exe_path = cmd_list[0]
    if not os.path.isfile(exe_path):
        raise ValueError(f"可执行文件不存在: {exe_path}")
    
    # 2. 白名单检查
    if allowed_commands:
        exe_name = os.path.basename(exe_path)
        if exe_name not in allowed_commands:
            raise ValueError(f"不允许执行: {exe_name}")
    
    # 3. 参数验证（防止路径遍历）
    for arg in cmd_list[1:]:
        if '..' in arg or arg.startswith('/'):
            raise ValueError(f"不安全的参数: {arg}")
    
    # 4. 执行
    return subprocess.run(
        cmd_list,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300  # 添加超时
    )

# 使用
ALLOWED_EXES = ['Aerofoil_Con20190510.exe', 'cfx5pre.exe']
safe_subprocess_run([exe_path, arg1], allowed_commands=ALLOWED_EXES)
```

**影响:**
- 需要定义允许执行的命令白名单
- 可能影响现有功能（如果依赖不安全的参数）

**估算工作量:** 3-4 小时

---

## 🟡 中优先级 - 短期改进 (Medium - Short-term)

### 4. 缺少 CSRF 保护 (Missing CSRF Protection)

**问题描述:**
Flask 应用未启用 CSRF 保护，POST 请求容易被跨站请求伪造攻击。

**修复方案:**
```python
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
csrf = CSRFProtect(app)

# API 端点可豁免
@app.route('/api/chat/stream', methods=['POST'])
@csrf.exempt
def chat_stream():
    # ...
```

**估算工作量:** 2 小时

---

### 5. 缺少 API 速率限制 (Missing Rate Limiting)

**问题描述:**
API 端点未实施速率限制，容易被滥用。

**修复方案:**
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="redis://localhost:6379"  # 或 memory://
)

@app.route('/api/chat/stream', methods=['POST'])
@limiter.limit("10 per minute")
def chat_stream():
    # ...
```

**估算工作量:** 3 小时

---

### 6. 输入验证不足 (Insufficient Input Validation)

**问题描述:**
用户输入未充分验证，可能导致注入攻击或系统崩溃。

**修复方案:**
```python
from pydantic import BaseModel, validator, constr, confloat

class DesignRequest(BaseModel):
    """设计请求验证模型"""
    flow_rate: confloat(gt=0, lt=1000) = 15.0  # 0 < flow_rate < 1000
    efficiency: confloat(ge=0, le=1) = 0.85    # 0 <= efficiency <= 1
    pressure_ratio: confloat(gt=1, lt=10) = 1.5  # 1 < pressure_ratio < 10
    
    @validator('efficiency')
    def validate_efficiency(cls, v):
        if v < 0.5 or v > 0.99:
            raise ValueError('效率应在 0.5-0.99 之间')
        return v

@app.route('/api/design', methods=['POST'])
def design():
    try:
        request_data = DesignRequest(**request.json)
        # 使用验证后的数据
        result = blade_design(
            flow_rate=request_data.flow_rate,
            efficiency=request_data.efficiency,
            pressure_ratio=request_data.pressure_ratio
        )
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
```

**估算工作量:** 4-5 小时

---

### 7. 日志记录不足 (Insufficient Logging)

**问题描述:**
缺少结构化日志，难以追踪问题和安全事件。

**修复方案:**
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON 格式日志"""
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_data)

# 配置日志
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger('blade_agent')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 使用
logger.info("用户请求设计", extra={"user_id": user_id, "params": params})
```

**估算工作量:** 3 小时

---

## 🟢 低优先级 - 长期优化 (Low - Long-term)

### 8. HTTPS 强制 (Enforce HTTPS)

**问题描述:**
生产环境应强制使用 HTTPS。

**修复方案:**
```python
from flask_talisman import Talisman

if os.getenv('FLASK_ENV') == 'production':
    Talisman(app, force_https=True)
```

**估算工作量:** 1 小时

---

### 9. 依赖版本锁定 (Dependency Version Pinning)

**问题描述:**
依赖使用范围版本（如 `>=`），可能引入不兼容变更。

**修复方案:**
```bash
# 生成精确版本锁定文件
pip freeze > requirements.lock

# 或使用 poetry
poetry export -f requirements.txt --output requirements.lock
```

**估算工作量:** 1 小时

---

### 10. 添加安全头 (Add Security Headers)

**问题描述:**
响应头缺少安全策略。

**修复方案:**
```python
@app.after_request
def set_security_headers(response):
    """设置安全响应头"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response
```

**估算工作量:** 1 小时

---

## 修复优先级矩阵 (Fix Priority Matrix)

| 问题 | 严重性 | 利用难度 | 影响范围 | 优先级 |
|------|--------|---------|---------|--------|
| PyTorch 不安全加载 | 高 | 中 | 高 | P0 🔴 |
| 硬编码密钥 | 高 | 低 | 中 | P0 🔴 |
| 命令注入 | 中 | 中 | 中 | P1 🟡 |
| CSRF 缺失 | 中 | 低 | 高 | P1 🟡 |
| 速率限制缺失 | 中 | 低 | 中 | P2 🟡 |
| 输入验证不足 | 中 | 中 | 中 | P2 🟡 |
| 日志不足 | 低 | N/A | 低 | P3 🟢 |
| HTTPS 缺失 | 中 | 低 | 高 | P3 🟢 |
| 版本锁定 | 低 | N/A | 低 | P4 🟢 |
| 安全头缺失 | 低 | 低 | 低 | P4 🟢 |

---

## 修复路线图 (Fix Roadmap)

### 第 1 周 (Week 1)
- [x] 修复 PyTorch 不安全加载
- [x] 移除硬编码密钥
- [ ] 添加单元测试验证修复

### 第 2 周 (Week 2)
- [ ] 修复命令注入风险
- [ ] 添加 CSRF 保护
- [ ] 实施 API 速率限制

### 第 3-4 周 (Week 3-4)
- [ ] 完善输入验证
- [ ] 改进日志系统
- [ ] 配置 HTTPS
- [ ] 添加安全响应头

---

## 安全检查清单 (Security Checklist)

### 部署前必检 (Pre-Deployment)
- [ ] 所有密钥通过环境变量配置
- [ ] 模型文件签名验证
- [ ] API 端点启用速率限制
- [ ] HTTPS 强制开启
- [ ] 日志级别设置为 INFO
- [ ] 错误消息不泄露敏感信息

### 定期检查 (Regular Checks)
- [ ] 每月运行 `bandit` 安全扫描
- [ ] 每季度更新依赖（安全补丁）
- [ ] 每季度审查访问日志
- [ ] 每年进行渗透测试

---

## 参考资源 (References)

1. [OWASP Top 10](https://owasp.org/www-project-top-ten/)
2. [CWE-502: Deserialization of Untrusted Data](https://cwe.mitre.org/data/definitions/502.html)
3. [PyTorch Security Best Practices](https://pytorch.org/docs/stable/notes/serialization.html)
4. [Flask Security Guide](https://flask.palletsprojects.com/en/latest/security/)
5. [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)

---

**最后更新:** 2025-11-17  
**维护者:** 安全团队
