"""
叶片可视化工具模块
只包含必要的可视化函数
"""
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import numpy as np

matplotlib.use('Agg')  # 使用非GUI后端，适合Web服务或后端处理

# 创建保存图像的目录
IMAGE_DIR = Path("static/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
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


