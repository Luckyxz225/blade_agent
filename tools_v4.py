"""
基于 LangGraph ToolNode 的工具模块
将原有的自定义工具系统重构为 LangChain 兼容的工具
"""

from typing import Union, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import base64
from io import BytesIO
from pathlib import Path
import uuid
import random
from datetime import datetime

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# 使用非GUI后端，适合Web服务
matplotlib.use('Agg')

# 创建保存图像的目录
IMAGE_DIR = Path("static/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


class BladeDesignInput(BaseModel):
    """叶片设计工具输入参数"""
    flow_rate: float = Field(description="目标流量 (kg/s)", ge=0.1, le=100.0)
    pressure_ratio: float = Field(description="目标压比", ge=1.0, le=10.0)
    efficiency: float = Field(description="目标效率", ge=0.1, le=1.0)


class BladePerformanceInput(BaseModel):
    """叶片性能评估工具输入参数"""
    blade_height: float = Field(description="叶片高度 (mm)", ge=1.0, le=1000.0)
    blade_chord: float = Field(description="叶片弦长 (mm)", ge=1.0, le=1000.0)
    blade_angle: float = Field(description="叶片角度 (度)", ge=1.0, lt=90.0)


class BladeProfileInput(BaseModel):
    """叶片轮廓可视化工具输入参数"""
    # 叶根参数
    root_angle_in: float = Field(default=57, description="叶根进口金属角（度）")
    root_angle_out: float = Field(default=-33, description="叶根出口金属角（度）")
    root_chord: float = Field(default=0.17, description="叶根弦长（m）")
    root_thmax_ch: float = Field(default=15.73, description="叶根最大厚度与弦长比（%）")
    root_thmaxp: float = Field(default=0.76, description="叶根最大厚度位置（0-1）")
    root_swa: float = Field(default=0.01, description="叶根掠量（m）")
    root_bowa: float = Field(default=0.01, description="叶根弯量（m）")
    
    # 叶中参数
    mid_angle_in: float = Field(default=55, description="叶中进口金属角（度）")
    mid_angle_out: float = Field(default=25, description="叶中出口金属角（度）")
    mid_chord: float = Field(default=0.2, description="叶中弦长（m）")
    mid_thmax_ch: float = Field(default=6.7, description="叶中最大厚度与弦长比（%）")
    mid_thmaxp: float = Field(default=0.6, description="叶中最大厚度位置（0-1）")
    mid_swa: float = Field(default=0.02, description="叶中掠量（m）")
    mid_bowa: float = Field(default=0.01, description="叶中弯量（m）")
    
    # 叶尖参数
    tip_angle_in: float = Field(default=68, description="叶尖进口金属角（度）")
    tip_angle_out: float = Field(default=61, description="叶尖出口金属角（度）")
    tip_chord: float = Field(default=0.13, description="叶尖弦长（m）")
    tip_thmax_ch: float = Field(default=6.3, description="叶尖最大厚度与弦长比（%）")
    tip_thmaxp: float = Field(default=0.6, description="叶尖最大厚度位置（0-1）")
    tip_swa: float = Field(default=-0.01, description="叶尖掠量（m）")
    tip_bowa: float = Field(default=0.02, description="叶尖弯量（m）")


class BladeDesignTool(BaseTool):
    """压气机叶片设计工具"""
    
    name: str = "blade_design"
    description: str = """根据设计目标生成压气机叶片的几何参数。
    输入流量、压比和效率要求，输出优化的叶片设计参数。
    适用于初始设计阶段的参数确定。"""
    args_schema: type[BaseModel] = BladeDesignInput
    
    def _run(self, flow_rate: float, pressure_ratio: float, efficiency: float) -> Dict[str, Any]:
        """执行叶片设计计算"""
        try:
            # 基于设计目标计算叶片参数
            # 这里使用简化的设计公式，实际应用中会使用更复杂的设计理论
            
            # 叶片高度计算 (基于流量)
            blade_height = 20 + flow_rate * 2.5  # mm
            
            # 叶片弦长计算 (基于压比)
            blade_chord = 15 + pressure_ratio * 8  # mm
            
            # 叶片角度计算 (基于效率)
            blade_angle = 30 + efficiency * 40  # degrees
            
            # 计算预估性能
            estimated_efficiency = min(0.95, 0.7 + efficiency * 0.2)
            estimated_pressure_ratio = max(1.0, pressure_ratio * 0.95)
            estimated_flow_rate = flow_rate * 0.98
            
            # 附加设计参数
            design_params = {
                "叶片高度": round(blade_height, 2),
                "叶片弦长": round(blade_chord, 2),
                "叶片角度": round(blade_angle, 2),
                "转速": round(3000 + flow_rate * 50, 0),
                "马赫数": round(0.4 + efficiency * 0.3, 3),
                "雷诺数": round(50000 + flow_rate * 10000, 0)
            }
            
            result = {
                "设计参数": design_params,
                "预估性能": {
                    "流量": round(estimated_flow_rate, 3),
                    "压比": round(estimated_pressure_ratio, 3),
                    "效率": round(estimated_efficiency, 3)
                },
                "设计状态": "成功",
                "时间戳": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                "错误": f"叶片设计失败: {str(e)}",
                "设计状态": "失败",
                "时间戳": datetime.now().isoformat()
            }
    
    async def _arun(self, flow_rate: float, pressure_ratio: float, efficiency: float) -> Dict[str, Any]:
        """异步执行"""
        return self._run(flow_rate, pressure_ratio, efficiency)


class BladePerformanceEvaluationTool(BaseTool):
    """叶片性能评估工具"""
    
    name: str = "blade_performance_evaluation"
    description: str = """评估给定叶片几何参数的气动性能。
    输入叶片高度、弦长和角度，输出性能指标包括流量、压比和效率。
    用于验证设计方案的可行性。"""
    args_schema: type[BaseModel] = BladePerformanceInput
    
    def _run(self, blade_height: float, blade_chord: float, blade_angle: float) -> Dict[str, Any]:
        """执行性能评估计算"""
        try:
            # 验证输入参数合理性
            if blade_height <= 0 or blade_chord <= 0 or blade_angle <= 0 or blade_angle >= 90:
                raise ValueError("参数超出合理范围")
            
            # 基于几何参数计算性能指标
            # 这里使用简化的性能模型，实际应用中会使用CFD或经验关系式
            
            # 流量计算（基于通流面积）
            flow_area = blade_height * blade_chord * 0.001  # 转换为m²
            flow_rate = flow_area * 50 * np.sin(np.radians(blade_angle))
            
            # 压比计算（基于叶片几何和角度）
            pressure_ratio = 1.0 + (blade_angle / 90) * 2.5 * (blade_height / 100)
            
            # 效率计算（基于设计参数匹配度）
            aspect_ratio = blade_height / blade_chord
            optimal_aspect_ratio = 2.0
            aspect_penalty = abs(aspect_ratio - optimal_aspect_ratio) / optimal_aspect_ratio
            efficiency = 0.85 - aspect_penalty * 0.2
            
            # 调整角度影响
            if blade_angle < 20 or blade_angle > 70:
                efficiency *= 0.9
            
            # 确保结果在合理范围内
            efficiency = max(0.1, min(0.95, efficiency))
            pressure_ratio = max(1.0, pressure_ratio)
            flow_rate = max(0.1, flow_rate)
            
            # 附加分析结果
            analysis = {
                "几何特征": {
                    "展弦比": round(aspect_ratio, 3),
                    "稠度": round(blade_chord / (blade_height * np.sin(np.radians(blade_angle))), 3),
                    "通流面积": round(flow_area, 6)
                },
                "性能评级": self._get_performance_rating(efficiency),
                "设计建议": self._get_design_recommendations(blade_height, blade_chord, blade_angle, efficiency)
            }
            
            result = {
                "性能指标": {
                    "流量": round(float(flow_rate), 6),
                    "压比": round(float(pressure_ratio), 6),
                    "效率": round(float(efficiency), 6)
                },
                "分析结果": analysis,
                "评估状态": "成功",
                "时间戳": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                "错误": f"性能评估失败: {str(e)}",
                "性能指标": {
                    "流量": 0.0,
                    "压比": 1.0,
                    "效率": 0.0
                },
                "评估状态": "失败",
                "时间戳": datetime.now().isoformat()
            }
    
    def _get_performance_rating(self, efficiency: float) -> str:
        """获取性能评级"""
        if efficiency >= 0.85:
            return "优秀"
        elif efficiency >= 0.75:
            return "良好"
        elif efficiency >= 0.65:
            return "一般"
        else:
            return "需要改进"
    
    def _get_design_recommendations(self, height: float, chord: float, angle: float, efficiency: float) -> list[str]:
        """获取设计建议"""
        recommendations = []
        
        aspect_ratio = height / chord
        if aspect_ratio < 1.5:
            recommendations.append("建议增加叶片高度或减少弦长以提高展弦比")
        elif aspect_ratio > 3.0:
            recommendations.append("展弦比较高，需注意结构强度")
            
        if angle < 25:
            recommendations.append("叶片角度较小，可能导致失速")
        elif angle > 65:
            recommendations.append("叶片角度较大，损失可能增加")
            
        if efficiency < 0.7:
            recommendations.append("效率偏低，建议优化几何参数")
            
        return recommendations if recommendations else ["当前设计参数较为合理"]
    
    async def _arun(self, blade_height: float, blade_chord: float, blade_angle: float) -> Dict[str, Any]:
        """异步执行"""
        return self._run(blade_height, blade_chord, blade_angle)


class PlotBladeProfileTool(BaseTool):
    """叶片轮廓可视化工具"""
    
    name: str = "plot_blade_profile"
    description: str = """根据叶片设计参数生成3D叶片几何形状可视化图像。
    输入叶根、叶中、叶尖的设计参数，生成压气机叶片的三维可视化图。
    用于直观展示叶片几何特征。"""
    args_schema: type[BaseModel] = BladeProfileInput
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """执行叶片可视化"""
        try:
            # 生成唯一文件名
            filename = f"blade3d_{uuid.uuid4().hex}.png"
            filepath = IMAGE_DIR / filename
            
            # 模拟加载叶片数据（实际应用中会根据参数生成或加载几何数据）
            success = self._generate_blade_visualization(filepath, **kwargs)
            
            if success:
                return {
                    "image_path": f"/static/images/{filename}",
                    "description": f"压气机叶片3D几何形状，包含叶根、叶中和叶尖截面",
                    "参数信息": self._format_parameters(**kwargs),
                    "可视化状态": "成功",
                    "时间戳": datetime.now().isoformat()
                }
            else:
                return {
                    "错误": "可视化数据生成失败",
                    "image_path": "",
                    "description": "3D叶片可视化失败",
                    "可视化状态": "失败",
                    "时间戳": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "错误": str(e),
                "image_path": "",
                "description": "3D叶片可视化失败",
                "可视化状态": "失败",
                "时间戳": datetime.now().isoformat()
            }
    
    def _generate_blade_visualization(self, filepath: Path, **kwargs) -> bool:
        """生成叶片可视化图像"""
        try:
            # 尝试加载真实叶片数据
            i = random.randint(5237, 5345)
            data_file_path = f"E:/3stage_R0_Blade_gemoturbo_shape/folder_{i}/{i}_BladeIn.dat"
            
            try:
                # 尝试加载真实数据
                data = np.loadtxt(data_file_path, skiprows=2)
                
                if data.shape == (1414, 3):
                    return self._plot_real_blade_data(filepath, data)
                else:
                    # 数据格式不匹配，使用模拟数据
                    return self._plot_simulated_blade_data(filepath, **kwargs)
                    
            except (FileNotFoundError, OSError):
                # 文件不存在，使用模拟数据
                return self._plot_simulated_blade_data(filepath, **kwargs)
                
        except Exception:
            # 任何异常都使用模拟数据
            return self._plot_simulated_blade_data(filepath, **kwargs)
    
    def _plot_real_blade_data(self, filepath: Path, data: np.ndarray) -> bool:
        """绘制真实叶片数据"""
        try:
            num_curves = 7
            points_per_curve = 202
            curves = [data[i * points_per_curve:(i + 1) * points_per_curve] for i in range(num_curves)]
            curves = np.array(curves)
            
            x, y, z = curves[:, :, 0], curves[:, :, 1], curves[:, :, 2]
            
            # 对后半段点倒序排列
            x[:, 101:] = x[:, 101:][:, ::-1]
            y[:, 101:] = y[:, 101:][:, ::-1]
            z[:, 101:] = z[:, 101:][:, ::-1]
            
            # 创建3D图形
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制曲面
            surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
            
            # 设置图形属性
            ax.set_title("Compressor Blade 3D Geometry", fontsize=14)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
            # 添加颜色条
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Height')
            
            # 调整视角
            ax.view_init(elev=25, azim=45)
            
            # 保存图像
            plt.savefig(filepath, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            return True
            
        except Exception:
            return False
    
    def _plot_simulated_blade_data(self, filepath: Path, **kwargs) -> bool:
        """绘制模拟叶片数据"""
        try:
            # 从参数中提取关键信息
            root_chord = kwargs.get('root_chord', 0.17)
            mid_chord = kwargs.get('mid_chord', 0.2)
            tip_chord = kwargs.get('tip_chord', 0.13)
            
            # 生成模拟的叶片几何
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 创建叶片截面
            n_sections = 7
            n_points = 50
            
            # 径向位置
            r = np.linspace(0, 1, n_sections)
            
            # 弦长分布
            chord_dist = np.interp(r, [0, 0.5, 1], [root_chord, mid_chord, tip_chord])
            
            # 生成每个截面的轮廓
            x_sections = []
            y_sections = []
            z_sections = []
            
            for i, ri in enumerate(r):
                # 翼型轮廓（NACA 类似）
                x_profile = np.linspace(0, chord_dist[i], n_points)
                thickness = 0.12 * chord_dist[i] * (0.2969 * np.sqrt(x_profile/chord_dist[i]) 
                                                   - 0.1260 * (x_profile/chord_dist[i])
                                                   - 0.3516 * (x_profile/chord_dist[i])**2
                                                   + 0.2843 * (x_profile/chord_dist[i])**3
                                                   - 0.1015 * (x_profile/chord_dist[i])**4)
                
                # 上表面
                y_upper = thickness/2
                # 下表面
                y_lower = -thickness/2
                
                # Z坐标（径向位置）
                z_pos = ri * 0.3  # 叶片高度为0.3m
                
                x_sections.append(x_profile)
                y_sections.append(np.concatenate([y_upper, y_lower[::-1]]))
                z_sections.append(np.full(len(y_upper) + len(y_lower), z_pos))
            
            # 转换为网格数据
            X = np.array(x_sections)
            Y = np.array(y_sections)
            Z = np.array(z_sections)
            
            # 绘制线框
            for i in range(n_sections):
                ax.plot(X[i, :n_points], Y[i, :n_points], Z[i, :n_points], 'b-', alpha=0.7)
                ax.plot(X[i, n_points:], Y[i, n_points:], Z[i, n_points:], 'r-', alpha=0.7)
            
            # 绘制径向线
            for j in range(0, n_points*2, 10):
                ax.plot(X[:, j], Y[:, j], Z[:, j], 'g-', alpha=0.5)
            
            # 设置图形属性
            ax.set_title("Simulated Compressor Blade Geometry", fontsize=14)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
            # 调整视角
            ax.view_init(elev=25, azim=45)
            
            # 保存图像
            plt.savefig(filepath, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            return True
            
        except Exception:
            return False
    
    def _format_parameters(self, **kwargs) -> Dict[str, Any]:
        """格式化参数信息"""
        return {
            "叶根参数": {
                "进口角": kwargs.get('root_angle_in', 57),
                "出口角": kwargs.get('root_angle_out', -33),
                "弦长": kwargs.get('root_chord', 0.17)
            },
            "叶中参数": {
                "进口角": kwargs.get('mid_angle_in', 55),
                "出口角": kwargs.get('mid_angle_out', 25),
                "弦长": kwargs.get('mid_chord', 0.2)
            },
            "叶尖参数": {
                "进口角": kwargs.get('tip_angle_in', 68),
                "出口角": kwargs.get('tip_angle_out', 61),
                "弦长": kwargs.get('tip_chord', 0.13)
            }
        }
    
    async def _arun(self, **kwargs) -> Dict[str, Any]:
        """异步执行"""
        return self._run(**kwargs)


# 导出工具列表，供agent_v4.py使用
AVAILABLE_TOOLS = [
    BladeDesignTool(),
    BladePerformanceEvaluationTool(),
    PlotBladeProfileTool()
]


def get_tool_descriptions() -> list[Dict[str, str]]:
    """获取工具描述信息"""
    descriptions = []
    for tool in AVAILABLE_TOOLS:
        descriptions.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": str(tool.args_schema.schema())
        })
    return descriptions


if __name__ == "__main__":
    # 测试工具功能
    print("测试叶片设计工具...")
    design_tool = BladeDesignTool()
    design_result = design_tool._run(10.0, 2.5, 0.85)
    print(f"设计结果: {design_result}")
    
    print("\n测试性能评估工具...")
    perf_tool = BladePerformanceEvaluationTool()
    perf_result = perf_tool._run(50.0, 30.0, 45.0)
    print(f"性能结果: {perf_result}")
    
    print("\n测试可视化工具...")
    plot_tool = PlotBladeProfileTool()
    plot_result = plot_tool._run()
    print(f"可视化结果: {plot_result}")