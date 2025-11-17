#!/usr/bin/env python3
"""
简单测试脚本：测试各个工具是否能正常工作
"""

import sys
import os

print("=" * 60)
print("工具测试脚本")
print("=" * 60)

# 测试1：导入模块
print("\n【测试1】导入模块...")
try:
    from langgraph_tools import (
        blade_design_tool,
        blade_performance_evaluation_tool,
        blade_visualization_tool
    )
    print("✅ 工具导入成功")
except Exception as e:
    print(f"❌ 工具导入失败: {e}")
    sys.exit(1)

# 测试2：简单的意图识别（不调用深度学习模型）
print("\n【测试2】测试可视化工具（不依赖PyTorch）...")
try:
    result = blade_visualization_tool.invoke({
        "root_Angle_in": 57,
        "root_Angle_out": -33,
        "root_Chord": 0.17,
        "root_THmax_CH": 15.73,
        "root_THmaxP": 0.76,
        "root_SWA": 0.01,
        "root_BOWA": 0.01,
        "mid_Angle_in": 55,
        "mid_Angle_out": 25,
        "mid_Chord": 0.2,
        "mid_THmax_CH": 6.7,
        "mid_THmaxP": 0.6,
        "mid_SWA": 0.02,
        "mid_BOWA": 0.01,
        "tip_Angle_in": 68,
        "tip_Angle_out": 61,
        "tip_Chord": 0.13,
        "tip_THmax_CH": 6.3,
        "tip_THmaxP": 0.6,
        "tip_SWA": -0.01,
        "tip_BOWA": 0.02
    })
    if "error" not in result:
        print(f"✅ 可视化工具测试成功: {result.get('message', '')}")
    else:
        print(f"⚠️ 可视化工具返回错误: {result['error']}")
except Exception as e:
    print(f"❌ 可视化工具测试失败: {e}")

# 测试3：设计生成（会调用PyTorch）
print("\n【测试3】测试设计生成工具（依赖PyTorch - 这是关键测试）...")
print("⏳ 正在加载模型并生成设计...")
print("   （这可能需要30-60秒，如果卡住超过2分钟说明有问题）")

try:
    import time
    start_time = time.time()
    
    result = blade_design_tool.invoke({
        "flow_rate": 15.0,
        "efficiency": 0.85,
        "pressure_ratio": 1.5,
        "top_k": 1
    })
    
    elapsed = time.time() - start_time
    
    if "error" not in result:
        print(f"✅ 设计生成成功！耗时：{elapsed:.1f}秒")
        print(f"   生成了设计参数")
    else:
        print(f"⚠️ 设计生成返回错误: {result['error']}")
        
except KeyboardInterrupt:
    print("\n⚠️ 用户中断测试")
    sys.exit(1)
except Exception as e:
    print(f"❌ 设计生成失败: {type(e).__name__}: {e}")
    import traceback
    print("\n完整错误信息：")
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)

