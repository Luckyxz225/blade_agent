import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免多线程GUI问题
import matplotlib.pyplot as plt
import uuid
import json
from pathlib import Path

# 设置中文字体（按优先级尝试多个字体）
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS',  # macOS
    'PingFang SC',       # macOS
    'Heiti SC',          # macOS
    'Microsoft YaHei',   # Windows
    'SimHei',            # Windows/Linux
    'DejaVu Sans',       # 通用备选
    'sans-serif'         # 最终备选
]
plt.rcParams['axes.unicode_minus'] = False

# 图片保存目录
IMAGE_DIR = Path(__file__).parent.parent / "static" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# 全局进度回调函数（用于实时推送可视化结果到前端）
_visualization_progress_callback = None

def set_visualization_progress_callback(callback):
    """设置可视化进度回调函数"""
    global _visualization_progress_callback
    _visualization_progress_callback = callback

def _report_visualization_progress(stage: str, progress: float, message: str = "", **kwargs):
    """报告可视化进度"""
    if _visualization_progress_callback:
        try:
            data = {
                'stage': stage,
                'progress': progress,
                'message': message
            }
            data.update(kwargs)
            _visualization_progress_callback(data)
        except Exception as e:
            print(f"[可视化进度回调错误] {e}")


def visualize_blade(
        # —— 叶根（Hub）参数 ——
        root_Angle_in: float = 57,
        root_Angle_out: float = -33,
        root_Chord: float = 0.17,
        root_THmax_CH: float = 15.73,
        root_THmaxP: float = 0.76,
        root_SWA: float = 0.01,
        root_BOWA: float = 0.01,

        # —— 叶中（Mid-span）参数 ——
        mid_Angle_in: float = 55,
        mid_Angle_out: float = 25,
        mid_Chord: float = 0.2,
        mid_THmax_CH: float = 6.7,
        mid_THmaxP: float = 0.6,
        mid_SWA: float = 0.02,
        mid_BOWA: float = 0.01,

        # —— 叶尖（Tip）参数 ——
        tip_Angle_in: float = 68,
        tip_Angle_out: float = 61,
        tip_Chord: float = 0.13,
        tip_THmax_CH: float = 6.3,
        tip_THmaxP: float = 0.6,
        tip_SWA: float = -0.01,
        
        # —— 控制参数 ——
        send_sse: bool = True,  # 是否发送 SSE 事件到前端（优化任务内部调用时设为 False）
        tip_BOWA: float = 0.02,

        blade_height: float = 0.5,
        span_samples: int = 25,
        chord_samples: int = 80,
        save_image: bool = True,
) -> dict:
    """
    根据 21 个叶片参数可视化完整三维叶片
    
    参数说明：
    21个参数分为3组（叶根/叶中/叶尖），每组7个参数：
    - Angle_in/out: 进出口金属角（度）
    - Chord: 弦长（米）
    - THmax_CH: 最大相对厚度（%）
    - THmaxP: 最大厚度位置（0-1）
    - SWA: 掠量（米）
    - BOWA: 弯量（米）
    
    返回值：
    包含图像路径和描述信息的字典
    """

    span_nodes = np.array([0.0, 0.5, 1.0])
    span_grid = np.linspace(0, 1, span_samples)
    chord_grid = np.linspace(0, 1, chord_samples)

    def interp_span(root_v, mid_v, tip_v):
        return np.interp(span_grid, span_nodes, [root_v, mid_v, tip_v])

    angle_in = np.deg2rad(interp_span(root_Angle_in, mid_Angle_in, tip_Angle_in))
    angle_out = np.deg2rad(interp_span(root_Angle_out, mid_Angle_out, tip_Angle_out))
    chord = interp_span(root_Chord, mid_Chord, tip_Chord)
    thmax_ch = interp_span(root_THmax_CH, mid_THmax_CH, tip_THmax_CH) / 100.0
    thmax_p = interp_span(root_THmaxP, mid_THmaxP, tip_THmaxP)
    swa = interp_span(root_SWA, mid_SWA, tip_SWA)
    bowa = interp_span(root_BOWA, mid_BOWA, tip_BOWA)

    XU = np.zeros((span_samples, chord_samples))
    YU = np.zeros_like(XU)
    ZU = np.zeros_like(XU)
    XL = np.zeros_like(XU)
    YL = np.zeros_like(XU)
    ZL = np.zeros_like(XU)

    for i, s in enumerate(span_grid):
        c = chord[i]
        x_norm = chord_grid
        x = x_norm * c

        theta = angle_in[i] + (angle_out[i] - angle_in[i]) * x_norm
        dx = x[1] - x[0]
        y_c = np.zeros_like(x)
        y_c[1:] = np.cumsum(np.tan(theta[1:]) * dx)

        t_max = c * thmax_ch[i]
        peak = thmax_p[i]
        th = t_max * np.exp(-((x_norm - peak) / 0.18) ** 2)

        dy_dx = np.gradient(y_c, dx)
        normal = np.arctan(dy_dx) + np.pi / 2.0

        x_u = x + th * np.cos(normal) + swa[i] * s
        y_u = y_c + th * np.sin(normal) + bowa[i] * s
        x_l = x - th * np.cos(normal) + swa[i] * s
        y_l = y_c - th * np.sin(normal) + bowa[i] * s
        z = np.full_like(x, s * blade_height)

        XU[i], YU[i], ZU[i] = x_u, y_u, z
        XL[i], YL[i], ZL[i] = x_l, y_l, z

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(XU, YU, ZU, color='steelblue', alpha=0.65, linewidth=0, antialiased=True)
    ax.plot_surface(XL, YL, ZL, color='lightgreen', alpha=0.55, linewidth=0, antialiased=True)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Compressor Blade')

    max_range = np.array([
        XU.max() - XU.min(),
        YU.max() - YU.min(),
        ZU.max() - ZU.min()
    ]).max() / 2.0
    mid_x = (XU.max() + XU.min()) * 0.5
    mid_y = (YU.max() + YU.min()) * 0.5
    mid_z = (ZU.max() + ZU.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    # 保存图像
    if save_image:
        file_id = uuid.uuid4().hex
        filename = f"blade3d_{file_id}.png"
        filepath = IMAGE_DIR / filename
        plt.savefig(filepath, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[可视化] 叶片绘制完毕: {filename}")
        
        image_path = f"/static/images/{filename}"
        
        # 【新增】同时保存 3D 数据 JSON 文件，供前端 Three.js 交互渲染
        json_filename = f"blade3d_{file_id}.json"
        json_filepath = IMAGE_DIR / json_filename
        
        # 将 3D 曲面数据转换为 JSON 格式（降采样以减小文件大小）
        # 对数据进行适当采样，保持模型清晰度同时减小数据量
        step = max(1, span_samples // 20)  # 展向采样
        chord_step = max(1, chord_samples // 40)  # 弦向采样
        
        blade_3d_data = {
            "upper_surface": {
                "x": XU[::step, ::chord_step].tolist(),
                "y": YU[::step, ::chord_step].tolist(),
                "z": ZU[::step, ::chord_step].tolist()
            },
            "lower_surface": {
                "x": XL[::step, ::chord_step].tolist(),
                "y": YL[::step, ::chord_step].tolist(),
                "z": ZL[::step, ::chord_step].tolist()
            },
            "metadata": {
                "blade_height": blade_height,
                "span_samples": len(range(0, span_samples, step)),
                "chord_samples": len(range(0, chord_samples, chord_step)),
                "parameters": {
                    "root": {
                        "Angle_in": root_Angle_in, "Angle_out": root_Angle_out,
                        "Chord": root_Chord, "THmax_CH": root_THmax_CH,
                        "THmaxP": root_THmaxP, "SWA": root_SWA, "BOWA": root_BOWA
                    },
                    "mid": {
                        "Angle_in": mid_Angle_in, "Angle_out": mid_Angle_out,
                        "Chord": mid_Chord, "THmax_CH": mid_THmax_CH,
                        "THmaxP": mid_THmaxP, "SWA": mid_SWA, "BOWA": mid_BOWA
                    },
                    "tip": {
                        "Angle_in": tip_Angle_in, "Angle_out": tip_Angle_out,
                        "Chord": tip_Chord, "THmax_CH": tip_THmax_CH,
                        "THmaxP": tip_THmaxP, "SWA": tip_SWA, "BOWA": tip_BOWA
                    }
                }
            }
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(blade_3d_data, f)
        print(f"[可视化] 3D数据已保存: {json_filename}")
        
        json_path = f"/static/images/{json_filename}"
        
        # 通过 SSE 发送可视化图片到前端图库（仅当 send_sse=True 时）
        # 优化任务内部调用时设为 False，避免干扰优化图片的展示
        if send_sse:
            _report_visualization_progress(
                'visualization_image',
                100,
                '可视化完成',
                image_path=image_path,
                json_path=json_path,  # 新增：传递 3D 数据路径
                label='可视化结果'
            )
        
        return {
            "image_path": image_path,
            "json_path": json_path,  # 新增：返回 3D 数据路径
            "description": "压气机叶片3D几何形状，包含叶根、叶中和叶尖截面",
            "status": "success"
        }
    else:
        return {
            "status": "success",
            "message": "Blade visualization rendered (not saved)",
        }


if __name__ == "__main__":
    visualize_blade()
    visualize_blade(
    root_Angle_in=52, root_Angle_out=-28, root_Chord=0.18,
    root_THmax_CH=14.5, root_THmaxP=0.72, root_SWA=0.015, root_BOWA=0.012,
    mid_Angle_in=58, mid_Angle_out=20, mid_Chord=0.21,
    mid_THmax_CH=7.2, mid_THmaxP=0.62, mid_SWA=0.018, mid_BOWA=0.012,
    tip_Angle_in=72, tip_Angle_out=55, tip_Chord=0.125,
    tip_THmax_CH=6.8, tip_THmaxP=0.58, tip_SWA=-0.008, tip_BOWA=0.018,
    blade_height=0.55, span_samples=25, chord_samples=90
    )
    plt.show(block=True)