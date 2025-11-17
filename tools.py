from typing import Union
# from agents import function_tool  # LangGraph版本不需要
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import matplotlib
from pathlib import Path
matplotlib.use('Agg')  # 使用非GUI后端，适合Web服务或后端处理
import random
import numpy as np
# @function_tool  # LangGraph版本不需要
def blade_performance_evaluation(blade_height: Union[float, int, str], blade_chord: Union[float, int, str],
                                 blade_angle: Union[float, int, str]) -> dict:
    """Evaluate blade performance based on geometry parameters."""
    # 记录工具参数
    arguments = {
        "blade_height": blade_height,
        "blade_chord": blade_chord,
        "blade_angle": blade_angle
    }


    try:
        blade_height = blade_height
        blade_chord = blade_chord
        blade_angle = blade_angle

        if blade_height <= 0 or blade_chord <= 0 or blade_angle <= 0 or blade_angle >= 90:
            raise ValueError("参数超出合理范围")

        flow_rate = 0.5
        pressure_ratio = 0.5
        efficiency = 0.5

        efficiency = max(0.1, min(0.95, efficiency))
        pressure_ratio = max(1.0, pressure_ratio)

        return {
            "flow_rate": round(float(flow_rate), 6),
            "pressure_ratio": round(float(pressure_ratio), 6),
            "efficiency": round(float(efficiency), 6)
        }
    except Exception as e:
        return {
            "error": f"性能评估失败: {str(e)}",
            "flow_rate": 0.0,
            "pressure_ratio": 1.0,
            "efficiency": 0.0
        }


# 创建保存图像的目录
IMAGE_DIR = Path("static/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# @function_tool
# def plot_blade_profile(
#         blade_height: float=2,
#         blade_chord: float=2,
#         blade_angle: float=40,
#         profile_style: str = "naca0012"
# ) -> dict:
#     """
#     绘制叶片轮廓图并保存为文件，返回图像路径和描述
#
#     **参数说明**:
#     - blade_height: 叶片高度(mm)
#     - blade_chord: 叶片弦长(mm)
#     - blade_angle: 叶片角度(度)
#     - profile_style: 轮廓类型 (naca0012|elliptic|parabolic)
#
#     **返回值** (dict):
#     必须为object,并且包含.image_path
#     {"image_path": "图像路径", "description": "轮廓描述"}
#     """
#     try:
#         # 生成唯一文件名
#         import uuid
#         filename = f"blade_{uuid.uuid4().hex}.png"
#         filepath = IMAGE_DIR / filename
#
#         # 创建叶片轮廓数据
#         fig, ax = plt.subplots(figsize=(8, 4))
#
#         # 根据不同类型生成轮廓
#         if profile_style == "naca0012":
#             x = [0, 0.25, 0.5, 0.75, 1.0]
#             y_upper = [0, 0.08, 0.05, 0.02, 0]
#             y_lower = [0, -0.08, -0.05, -0.02, 0]
#         elif profile_style == "elliptic":
#             x = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
#             y_upper = [0, 0.12, 0.15, 0.12, 0.06, 0]
#             y_lower = [0, -0.12, -0.15, -0.12, -0.06, 0]
#         else:  # parabolic
#             x = [0, 0.3, 0.6, 1.0]
#             y_upper = [0, 0.1, 0.07, 0]
#             y_lower = [0, -0.1, -0.07, 0]
#
#         # 缩放尺寸
#         x_scaled = [xi * blade_chord for xi in x]
#         y_upper_scaled = [yi * blade_chord + blade_height / 2 for yi in y_upper]
#         y_lower_scaled = [yi * blade_chord - blade_height / 2 for yi in y_lower]
#
#         # 绘制轮廓
#         ax.plot(x_scaled, y_upper_scaled, 'b-', label='Upper Surface')
#         ax.plot(x_scaled, y_lower_scaled, 'r-', label='Lower Surface')
#         ax.fill_between(x_scaled, y_upper_scaled, y_lower_scaled, color='gray', alpha=0.2)
#
#         # 设置图形属性
#         ax.set_title(f"Blade Profile ({profile_style})")
#         ax.set_xlabel('Chord Length (mm)')
#         ax.set_ylabel('Height (mm)')
#         ax.legend()
#         ax.grid(True)
#         ax.set_aspect('equal')
#
#         # 保存图像
#         plt.savefig(filepath, format='png', dpi=100)
#         plt.close(fig)
#
#         return {
#             "image_path": f"/static/images/{filename}",
#             "description": f"{profile_style}型轮廓，弦长{blade_chord}mm，高度{blade_height}mm"
#         }
#     except Exception as e:
#         return {
#             "error": str(e),
#             "image_path": "",
#             "description": "绘图失败"
#         }


# @function_tool  # LangGraph版本不需要
def plot_blade_profile(
        # —— 叶根（Hub）参数 ——
        root_Angle_in: float=57,  # 叶根进口金属角（单位：度）
        root_Angle_out: float=-33,  # 叶根出口金属角（单位：度）
        root_Chord: float=0.17,  # 叶根弦长（单位：m）
        root_THmax_CH: float=15.73,  # 叶根最大厚度与弦长比 THmax/CH（单位：%）
        root_THmaxP: float=0.76,  # 叶根最大厚度位置（0-1）
        root_SWA: float=0.01,  # 叶根掠量 SWA（单位：m）
        root_BOWA: float=0.01,  # 叶根弯量 BOWA（单位：m）

        # —— 叶中（Mid-span）参数 ——
        mid_Angle_in: float=55,  # 叶中进口金属角（单位：度）
        mid_Angle_out: float=25,  # 叶中出口金属角（单位：度）
        mid_Chord: float=0.2,  # 叶中弦长（单位：m）
        mid_THmax_CH: float=6.7,  # 叶中最大厚度与弦长比 THmax/CH（单位：%）
        mid_THmaxP: float=0.6,  # 叶中最大厚度位置（0-1）
        mid_SWA: float=0.02,  # 叶中掠量 SWA（单位：m）
        mid_BOWA: float=0.01,  # 叶中弯量 BOWA（单位：m）

        # —— 叶尖（Tip）参数 ——
        tip_Angle_in: float=68,  # 叶尖进口金属角（单位：度）
        tip_Angle_out: float=61,  # 叶尖出口金属角（单位：度）
        tip_Chord: float=0.13,  # 叶尖弦长（单位：m）
        tip_THmax_CH: float=6.3,  # 叶尖最大厚度与弦长比 THmax/CH（单位：%）
        tip_THmaxP: float=0.6,  # 叶尖最大厚度位置（0-1）
        tip_SWA: float=-0.01,  # 叶尖掠量 SWA（单位：m）
        tip_BOWA: float=0.02  # 叶尖弯量 BOWA（单位：m）
) -> dict:
    """
    根据21个设计参数可视化压气机叶片的几何形状并保存图片

    **返回值** (dict):
    必须为object,并且包含.image_path
    {"image_path": "图像路径", "description": "轮廓描述"}
    """
    try:
        # 生成唯一文件名
        import uuid
        filename = f"blade3d_{uuid.uuid4().hex}.png"
        filepath = IMAGE_DIR / filename

        # 基于输入参数生成简化的叶片几何可视化
        # 创建三个截面（叶根、叶中、叶尖）的简化表示
        num_curves = 7  # 沿叶高方向的截面数量
        points_per_curve = 202  # 每个截面的点数
        
        # 从叶根到叶尖线性插值
        hub_params = [root_Angle_in, root_Angle_out, root_Chord, root_THmax_CH, root_THmaxP, root_SWA, root_BOWA]
        mid_params = [mid_Angle_in, mid_Angle_out, mid_Chord, mid_THmax_CH, mid_THmaxP, mid_SWA, mid_BOWA]
        tip_params = [tip_Angle_in, tip_Angle_out, tip_Chord, tip_THmax_CH, tip_THmaxP, tip_SWA, tip_BOWA]
        
        # 生成简化的叶片表面点
        curves = []
        for i in range(num_curves):
            t = i / (num_curves - 1)  # 从0到1
            # 线性插值参数
            if t < 0.5:
                params = [(1-2*t)*h + 2*t*m for h, m in zip(hub_params, mid_params)]
            else:
                params = [(2-2*t)*m + (2*t-1)*p for m, p in zip(mid_params, tip_params)]
            
            chord = params[2]
            thickness = params[3] / 100 * chord  # THmax_CH 是百分比
            sweep = params[5]
            bow = params[6]
            
            # 生成翼型轮廓点（简化的椭圆形）
            theta = np.linspace(0, 2*np.pi, points_per_curve)
            x = chord * np.cos(theta) / 2 + sweep
            y = thickness * np.sin(theta) / 2 + bow
            z = np.full_like(x, t * 0.3)  # 叶高方向（假设叶高0.3m）
            
            curve = np.column_stack([x, y, z])
            curves.append(curve)
        
        curves = np.array(curves)

        x, y, z = curves[:, :, 0], curves[:, :, 1], curves[:, :, 2]

        # 对后半段点倒序排列
        x[:, 101:] = x[:, 101:][:, ::-1]
        y[:, 101:] = y[:, 101:][:, ::-1]
        z[:, 101:] = z[:, 101:][:, ::-1]

        # 创建3D图形
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制曲面
        surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)

        # 设置图形属性
        ax.set_title(f"Compressor Blade Surface")
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
        print(f"叶片绘制完毕{filename}")
        return {
            "image_path": f"/static/images/{filename}",
            "description": f"压气机叶片3D几何形状，包含叶根、叶中和叶尖截面"
        }
    except Exception as e:
        return {
            "error": str(e),
            "image_path": "",
            "description": "3D叶片可视化失败"
        }


# if __name__=="__main__":
    # 查看装饰后的工具元数据
    # tool = plot_blade_profile  # 装饰后的函数实际是 FunctionTool 实例
    # print(tool.name)  # 函数名（或 name_override）
    # print(tool.description)  # 从docstring提取的描述
    # print(tool.params_json_schema)  # 包含参数描述的JSON Schema
    # a=visualization_blade_geometry()