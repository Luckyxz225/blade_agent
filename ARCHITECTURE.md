# LangGraph智能体架构与功能

## 一、核心架构

### 1.1 技术栈
- **框架**: LangChain + LangGraph
- **LLM**: 支持OpenAI兼容API（GPT、DeepSeek等）
- **深度学习**: PyTorch + 自定义UNet/ViT模型
- **Web服务**: Flask + RESTful API

### 1.2 状态管理
系统使用TypedDict定义全局状态（DesignState），包含：

| 状态字段 | 类型 | 说明 |
|---------|------|------|
| user_input | str | 用户原始输入 |
| intent | str | 意图类型（design_generation/performance_eval/visualization/unknown） |
| messages | List | 对话历史消息 |
| performance_targets | dict | 性能目标（流量/效率/压比） |
| design_params | dict | 设计参数（21维物理指标） |
| performance_metrics | dict | 性能指标（3维评估结果） |
| blade_profile_path | str | 叶片可视化图像路径 |
| optimization_iteration | int | 优化迭代次数 |
| final_result | str | 最终结果文本 |
| error_message | str | 错误信息 |

## 二、工作流节点

### 2.1 意图识别节点（Intent Recognition）
**功能**: 识别用户需求并提取参数

**支持意图**:
1. `design_generation` - 根据性能目标设计叶片
2. `performance_eval` - 评估现有设计性能
3. `visualization` - 可视化叶片几何形状
4. `unknown` - 闲聊或功能询问

**智能参数处理**:
- 支持部分参数输入，自动补充默认值
- 性能目标默认值：流量=15kg/s，效率=0.85，压比=1.5
- 设计参数默认值：从预训练模型先验分布采样

### 2.2 设计生成节点（Design Generation）
**功能**: 性能目标 → 设计参数（逆向设计）

**输入**: 3维性能目标（流量、效率、压比）
**输出**: 21维设计参数（几何参数）
**模型**: 先验引导UNet + ViT性能预测
**文件**: `generative_design/compressor_design.py`

### 2.3 性能评估节点（Performance Evaluation）
**功能**: 设计参数 → 性能指标（正向评估）

**输入**: 21维设计参数
**输出**: 3维性能指标
**模型**: ViT Transformer
**文件**: `generative_design/blade_performance_evaluation.py`

### 2.4 可视化节点（Visualization）
**功能**: 生成叶片3D几何图像

**输入**: 21维设计参数
**输出**: PNG图像文件
**方法**: 基于参数的数学建模绘图
**文件**: `tools.py::plot_blade_profile`

### 2.5 优化循环节点（Optimization Loop）
**功能**: 迭代优化设计

**流程**:
```
当前设计 → 性能评估 → 判断是否满足目标
    ↓ 不满足
微调目标 → 重新设计 → 重复（最多5次）
    ↓ 满足或达到上限
输出最优设计
```

### 2.6 结果整合节点（Result Synthesis）
**功能**: 汇总执行结果，生成结构化报告

**处理逻辑**:
1. 收集各节点执行结果（设计参数、性能指标、图像）
2. 生成结构化数据报告
3. 调用LLM生成简洁文字总结

### 2.7 错误处理节点（Error Handler）
**功能**: 统一异常处理和用户交互

**处理类型**:
1. **功能询问**（LLM智能判断）
   - 用户询问系统能力 → 返回功能介绍
   
2. **普通闲聊**（LLM智能判断）
   - 非工作相关话题 → LLM生成个性化回复，引导回主题
   
3. **执行错误**
   - 工具调用失败、参数异常 → 返回具体错误信息

**关键创新**: 使用LLM分类器判断问题类型，替代关键词匹配，提升灵活性

## 三、工作流路由

### 3.1 意图路由
```
START → Intent Recognition
         ↓
    intent判断
    /    |    \
design  perf  viz → 对应节点 → 条件路由
generation eval           ↓
                      有错误? → Error Handler
                          ↓ 无错误
                    Result Synthesis → END
```

### 3.2 条件路由
- **需要优化**: `DESIGN_GENERATION` → `OPTIMIZATION_LOOP`
- **无需优化**: 直接 → `RESULT_SYNTHESIS`
- **发生错误**: 任意节点 → `ERROR_HANDLER`

## 四、核心功能

### 4.1 叶片设计（Design Generation）
**场景**: 用户提供性能目标，系统自动设计叶片

**输入示例**:
```
"设计一个流量15kg/s、效率0.85、压比1.5的叶片"
```

**输出**:
- 21维设计参数（几何描述）
- 预测性能指标
- 3D可视化图像

### 4.2 性能评估（Performance Evaluation）
**场景**: 用户提供设计参数，系统评估性能

**输入示例**:
```
"评估这个设计的性能：[21个参数...]"
```

**输出**:
- 流量、效率、压比预测值
- 与目标的对比分析

### 4.3 可视化（Visualization）
**场景**: 用户需要查看叶片几何形状

**输入示例**:
```
"显示叶片的3D形状"
```

**输出**:
- 3D几何图像（PNG格式）
- 图像访问URL

### 4.4 迭代优化（Optimization）
**场景**: 自动迭代改进设计直至满足目标

**触发条件**:
- 性能与目标偏差 > 阈值
- 迭代次数 < 5次

**优化策略**:
- 根据当前性能与目标的差值调整下一次输入
- 保留历史最优结果

### 4.5 智能闲聊（Idle Chat）
**场景**: 用户询问系统能力或进行闲聊

**功能询问** → 返回系统功能介绍
**普通闲聊** → LLM生成自然回复并引导回主题

**分类方式**: LLM语义判断，而非关键词匹配

## 五、模型资源

### 5.1 模型文件
| 文件名 | 用途 | 大小 |
|--------|------|------|
| prior_parameters_guide_15200.pth | 先验参数生成模型 | ~100MB |
| pre_guide_15200.pth | 性能预测引导模型 | ~50MB |
| best_model_in_testdataset.pth | 最优性能评估模型 | ~80MB |

### 5.2 数据文件
| 文件名 | 用途 |
|--------|------|
| parameter_geometry_5200.npy | 几何参数数据集 |
| label_5200.npy | 标签数据集 |

## 六、系统优势

### 6.1 智能性
- LLM驱动的意图理解和参数提取
- 支持自然语言交互
- 智能补全缺失参数
- 灵活的闲聊处理（LLM分类）

### 6.2 鲁棒性
- 完善的错误处理机制
- 降级策略（LLM失败时回退到关键词匹配）
- 详细的日志输出便于调试

### 6.3 可扩展性
- 模块化节点设计
- 统一的状态管理
- 易于添加新功能节点

### 6.4 工程化
- 支持CPU/GPU自适应
- 动态路径解析（跨平台兼容）
- RESTful API接口
- Web前端集成

