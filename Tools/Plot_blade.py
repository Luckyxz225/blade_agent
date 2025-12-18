import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.colors import LightSource, to_rgb
from typing import Tuple
import os
import sys

# 获取项目根目录并导入系统配置
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
sys.path.insert(0, _project_root)

# 使用系统配置设置 Matplotlib 后端（跨平台支持）
try:
    from system_config import SystemConfig
    # 交互式后端（用于显示图形窗口）
    SystemConfig.setup_matplotlib(interactive=True)
except ImportError:
    # 回退：尝试常见后端
    import platform
    system = platform.system().lower()
    backends_to_try = []
    if system == "darwin":  # macOS
        backends_to_try = ['MacOSX', 'TkAgg', 'Agg']
    elif system == "windows":
        backends_to_try = ['TkAgg', 'Qt5Agg', 'Agg']
    else:  # Linux
        backends_to_try = ['TkAgg', 'Qt5Agg', 'Agg']
    
    for backend in backends_to_try:
        try:
            matplotlib.use(backend)
            break
        except:
            continue

# ---------- 材质与绘图配置 ----------
STYLE_CONFIG = {
    "blade_material_color": "#BCC6CC",  # 金属基色（叶片）
    "hub_color": "#708090",  # 轮毂颜色
    "shroud_color": "#2C3E50",  # 机匣颜色（线框）
    "bg_color": "white",
    "grid_alpha": 0.1,
    "grid_style": ":",
    "light_azdeg": 220,
    "light_altdeg": 45,
    "vert_exag": 0.5,
    "blend_mode": "overlay",
}


# ---------- 公共工具函数 ----------

def set_equal_3d(ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """
    设置 3D 坐标轴为等比例显示，防止图形拉伸。

    Parameters
    ----------
    ax : plt.Axes
        3D 坐标轴对象。
    x, y, z : np.ndarray
        所有点的坐标数组。
    """
    x_flat = np.asarray(x).ravel()
    y_flat = np.asarray(y).ravel()
    z_flat = np.asarray(z).ravel()

    max_range = np.max([x_flat.max() - x_flat.min(),
                        y_flat.max() - y_flat.min(),
                        z_flat.max() - z_flat.min()]) / 2.0
    mid_x = (x_flat.max() + x_flat.min()) * 0.5
    mid_y = (y_flat.max() + y_flat.min()) * 0.5
    mid_z = (z_flat.max() + z_flat.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    try:
        # 去掉坐标轴面板填充
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
    except Exception:
        pass

    ax.grid(True, linestyle=STYLE_CONFIG["grid_style"], alpha=STYLE_CONFIG["grid_alpha"], color="black")


def smooth_blade(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 new_sections: int = 50, new_points_per_section: int = 202
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对叶片表面数据进行双线性插值，实现平滑。

    Parameters
    ----------
    x, y, z : np.ndarray
        原始叶片坐标数组，形状为 (sections, points_per_section)
    new_sections : int
        平滑后的截面数量。
    new_points_per_section : int
        每个截面上的点数量。

    Returns
    -------
    x_smooth, y_smooth, z_smooth : np.ndarray
        平滑后的叶片坐标数组。
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    num_sections, num_pts = x.shape

    # 插值到新的截面点
    t_old_section = np.linspace(0.0, 1.0, num_sections)
    t_new_section = np.linspace(0.0, 1.0, new_sections)
    t_old_pts = np.linspace(0.0, 1.0, num_pts)
    t_new_pts = np.linspace(0.0, 1.0, new_points_per_section)

    x_interp_pts = np.zeros((num_sections, new_points_per_section))
    y_interp_pts = np.zeros_like(x_interp_pts)
    z_interp_pts = np.zeros_like(x_interp_pts)

    for i in range(num_sections):
        x_interp_pts[i, :] = np.interp(t_new_pts, t_old_pts, x[i, :])
        y_interp_pts[i, :] = np.interp(t_new_pts, t_old_pts, y[i, :])
        z_interp_pts[i, :] = np.interp(t_new_pts, t_old_pts, z[i, :])

    # 插值到新的截面数
    x_smooth = np.zeros((new_sections, new_points_per_section))
    y_smooth = np.zeros_like(x_smooth)
    z_smooth = np.zeros_like(x_smooth)

    for j in range(new_points_per_section):
        x_smooth[:, j] = np.interp(t_new_section, t_old_section, x_interp_pts[:, j])
        y_smooth[:, j] = np.interp(t_new_section, t_old_section, y_interp_pts[:, j])
        z_smooth[:, j] = np.interp(t_new_section, t_old_section, z_interp_pts[:, j])

    return x_smooth, y_smooth, z_smooth


def rotate_blade(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 angle_deg: float, axis: str = 'Z'
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    绕指定轴旋转叶片或几何体。

    Parameters
    ----------
    x, y, z : np.ndarray
        坐标数组。
    angle_deg : float
        旋转角度（度）。
    axis : str
        旋转轴，'X', 'Y', 'Z'。

    Returns
    -------
    x_new, y_new, z_new : np.ndarray
        旋转后的坐标。
    """
    axis = axis.upper()
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

    if axis == 'Z':
        x_new = x * c - y * s
        y_new = x * s + y * c
        z_new = z
    elif axis == 'X':
        y_new = y * c - z * s
        z_new = y * s + z * c
        x_new = x
    elif axis == 'Y':
        x_new = x * c + z * s
        z_new = -x * s + z * c
        y_new = y
    else:
        raise ValueError("axis must be 'X', 'Y', or 'Z'")
    return x_new, y_new, z_new


def load_blade(file_path: str, num_curves: int = 7, points_per_curve: int = 202
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从数据文件加载叶片坐标并翻转右半部分，保证截面闭合。

    Parameters
    ----------
    file_path : str
        数据文件路径。
    num_curves : int
        叶片截面数量。
    points_per_curve : int
        每个截面的点数量。

    Returns
    -------
    x, y, z : np.ndarray
        叶片坐标数组，形状 (num_curves, points_per_curve)。
    """
    data = np.loadtxt(file_path, skiprows=2)
    curves = data.reshape(num_curves, points_per_curve, 3)
    x = curves[:, :, 0]
    y = curves[:, :, 1]
    z = curves[:, :, 2]

    mid_idx = points_per_curve // 2
    x[:, mid_idx:] = x[:, mid_idx:][:, ::-1]
    y[:, mid_idx:] = y[:, mid_idx:][:, ::-1]
    z[:, mid_idx:] = z[:, mid_idx:][:, ::-1]

    return x, y, z


# ---------- 主绘图函数 ----------

def plot_blade_profile_dynamic(
        file_path: str,
        rotate_copy: bool = False,
        axis: str = 'Z',
        num_blades: int = 1,
        show_hub: bool = True,
        show_shroud: bool = True,
        smooth_sections: int = 10,
        smooth_points: int = 100,
        hub_theta_points: int = 60,
        blade_alpha: float = 1.0,
        hub_alpha: float = 0.9,
        shroud_alpha: float = 0.15
) -> None:
    """
    绘制叶片、轮毂和机匣三维可视化，并支持旋转复制叶片。

    Parameters
    ----------
    file_path : str
        叶片数据文件路径。
    rotate_copy : bool
        是否旋转复制生成多片叶片。
    axis : str
        旋转轴方向 'X', 'Y', 'Z'。
    num_blades : int
        旋转复制叶片数量。
    show_hub : bool
        是否显示轮毂。
    show_shroud : bool
        是否显示机匣。
    smooth_sections : int
        平滑后的截面数量。
    smooth_points : int
        平滑后的截面点数。
    hub_theta_points : int
        轮毂圆周分段数。
    blade_alpha : float
        叶片透明度。
    hub_alpha : float
        轮毂透明度。
    shroud_alpha : float
        机匣透明度。
    """
    # ---------- 读取与平滑叶片 ----------
    x_raw, y_raw, z_raw = load_blade(file_path)
    x, y, z = smooth_blade(x_raw, y_raw, z_raw,
                           new_sections=smooth_sections, new_points_per_section=smooth_points)

    # ---------- 创建绘图 ----------
    fig = plt.figure(figsize=(12, 9), facecolor=STYLE_CONFIG["bg_color"])
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(STYLE_CONFIG["bg_color"])
    ls = LightSource(azdeg=STYLE_CONFIG["light_azdeg"], altdeg=STYLE_CONFIG["light_altdeg"])

    all_x, all_y, all_z = [], [], []
    base_rgb = np.array(to_rgb(STYLE_CONFIG["blade_material_color"]))
    blade_rgb_base = np.tile(base_rgb, (x.shape[0], x.shape[1], 1))

    # ---------- 绘制叶片 ----------
    loop_count = num_blades if rotate_copy else 1
    for i in range(loop_count):
        angle = (i * 360.0 / num_blades) if rotate_copy else 0.0
        x_rot, y_rot, z_rot = rotate_blade(x, y, z, angle, axis)
        all_x.append(x_rot)
        all_y.append(y_rot)
        all_z.append(z_rot)

        rgb_shaded = ls.shade_rgb(blade_rgb_base, elevation=z_rot,
                                  vert_exag=STYLE_CONFIG["vert_exag"],
                                  blend_mode=STYLE_CONFIG["blend_mode"])
        ax.plot_surface(x_rot, y_rot, z_rot, facecolors=rgb_shaded,
                        rstride=1, cstride=1, linewidth=0, antialiased=True,
                        shade=False, alpha=blade_alpha)

    # ---------- 内置轮毂与机匣 ----------
    hub_X_profile = np.array(
        [-340., -328.667, -317.333, -306., -294.667, -283.333, -272., -260.667, -249.333, -238.,
         -226.667, -215.333, -204., -192.667, -181.333, -170., -158.667, -147.333, -136.,
         -124.667, -113.333, -102., -90.667, -79.333, -68., -56.667, -45.333, -34.,
         -22.667, -11.333, 0.])
    hub_R_profile = np.array(
        [0., 19.315, 38.629, 57.944, 77.258, 96.573, 115.888, 135.202, 154.517, 153.667,
         153.192, 153.105, 153.296, 153.929, 155.076, 156.711, 158.457, 160.189, 161.906,
         163.611, 165.143, 166.468, 167.589, 168.514, 169.393, 169.721, 169.833, 170.449,
         171.073, 171.695, 172.317])

    shroud_X_profile = np.array(
        [-260.667, -249.333, -238., -226.667, -215.333, -204., -192.667, -181.333, -170.,
         -158.667, -147.333, -136., -124.667, -113.333, -102., -90.667, -79.333, -68.,
         -56.667, -45.333, -34., -22.667, -11.333, 0.])
    shroud_R_profile = np.array(
        [420.4, 418.727, 417.054, 415.38, 413.708, 412.034, 410.362, 408.688, 407.016,
         405.342, 403.669, 401.996, 400.323, 398.65, 396.977, 395.303, 393.63, 391.957,
         390.284, 388.611, 386.938, 385.265, 383.591, 381.919])

    # ---------- 构建圆柱网格 ----------
    theta = np.linspace(0.0, 2.0 * np.pi, hub_theta_points)

    def build_cylinder(x_g: np.ndarray, r_g: np.ndarray, t_g: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """根据圆柱坐标生成三维网格"""
        X_cyl = x_g
        Y_cyl = r_g * np.cos(t_g)
        Z_cyl = r_g * np.sin(t_g)
        return X_cyl, Y_cyl, Z_cyl

    # hub
    theta_grid_hub, x_grid_hub = np.meshgrid(theta, hub_X_profile)
    hub_r_grid = np.tile(hub_R_profile[:, np.newaxis], (1, hub_theta_points))
    h_X, h_Y, h_Z = build_cylinder(x_grid_hub, hub_r_grid, theta_grid_hub)

    # shroud
    theta_grid_shroud, x_grid_shroud = np.meshgrid(theta, shroud_X_profile)
    shroud_r_grid = np.tile(shroud_R_profile[:, np.newaxis], (1, hub_theta_points))
    s_X, s_Y, s_Z = build_cylinder(x_grid_shroud, shroud_r_grid, theta_grid_shroud)

    # ---------- 根据旋转轴调整方向 ----------
    if axis == 'Z':
        h_X, h_Y, h_Z = rotate_blade(h_X, h_Y, h_Z, 90.0, 'Y')
        s_X, s_Y, s_Z = rotate_blade(s_X, s_Y, s_Z, 90.0, 'Y')
    elif axis == 'Y':
        h_X, h_Y, h_Z = rotate_blade(h_X, h_Y, h_Z, -90.0, 'Z')
        s_X, s_Y, s_Z = rotate_blade(s_X, s_Y, s_Z, -90.0, 'Z')

    # ---------- 绘制轮毂与机匣 ----------
    if show_hub:
        hub_rgb_matrix = np.tile(np.array(to_rgb(STYLE_CONFIG["hub_color"])), (h_Z.shape[0], h_Z.shape[1], 1))
        ax.plot_surface(h_X, h_Y, h_Z, facecolors=ls.shade_rgb(hub_rgb_matrix, h_Z,
                                                               vert_exag=STYLE_CONFIG["vert_exag"],
                                                               blend_mode=STYLE_CONFIG["blend_mode"]),
                        linewidth=0, antialiased=True, shade=False, alpha=hub_alpha)
        all_x.append(h_X);
        all_y.append(h_Y);
        all_z.append(h_Z)

    if show_shroud:
        shroud_rgb_matrix = np.tile(np.array(to_rgb(STYLE_CONFIG["shroud_color"])), (s_Z.shape[0], s_Z.shape[1], 1))
        ax.plot_surface(s_X, s_Y, s_Z, facecolors=ls.shade_rgb(shroud_rgb_matrix, s_Z,
                                                               vert_exag=STYLE_CONFIG["vert_exag"],
                                                               blend_mode=STYLE_CONFIG["blend_mode"]),
                        linewidth=0, antialiased=True, shade=False, alpha=shroud_alpha)
        ax.plot_wireframe(s_X, s_Y, s_Z, color='black', alpha=0.2, rstride=8, cstride=8, linewidth=0.3)
        all_x.append(s_X);
        all_y.append(s_Y);
        all_z.append(s_Z)

    # ---------- 坐标轴等比例 ----------
    if all_x:
        set_equal_3d(ax,
                     np.concatenate([a.flatten() for a in all_x]),
                     np.concatenate([a.flatten() for a in all_y]),
                     np.concatenate([a.flatten() for a in all_z]))

    ax.set_title(f"Metal Blade Simulation ({num_blades if rotate_copy else 1} Blades, Axis: {axis})",
                 fontsize=12, color="#333333", pad=15)
    ax.set_xlabel("X [mm]");
    ax.set_ylabel("Y [mm]");
    ax.set_zlabel("Z [mm]")
    ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
    plt.show()


# ---------- 用于智能体可视化的函数（保存图片版本）----------

def visualize_blade_from_dat(
        file_path: str = None,
        rotate_copy: bool = True,
        axis: str = 'X',
        num_blades: int = 16,
        show_hub: bool = True,
        show_shroud: bool = True,
        smooth_sections: int = 10,
        smooth_points: int = 30,
        blade_alpha: float = 0.8,
        hub_alpha: float = 1.0,
        shroud_alpha: float = 0.15,
        send_sse: bool = True
) -> dict:
    """
    从 .dat 文件生成高质量叶片可视化图片（用于智能体可视化任务）
    
    与 plot_blade_profile_dynamic 的区别：
    - 使用非交互式后端（Agg），可以保存图片
    - 保存图片到 static/images 目录
    - 返回图片路径供前端展示
    
    Parameters
    ----------
    file_path : str
        叶片数据文件路径。如果为 None，使用默认的示例文件。
    其他参数同 plot_blade_profile_dynamic
    
    Returns
    -------
    dict
        包含 image_path, json_path, description, status 等字段
    """
    import uuid
    import json
    from pathlib import Path
    
    # 使用非交互式后端
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, to_rgb
    
    # 设置默认文件路径（使用项目中的示例 dat 文件）
    if file_path is None:
        current_dir = Path(__file__).parent.parent
        file_path = current_dir / "dat_file" / "0_BladeIn.dat"
    
    # 图片保存目录
    IMAGE_DIR = Path(__file__).parent.parent / "static" / "images"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # ---------- 读取与平滑叶片 ----------
        x_raw, y_raw, z_raw = load_blade(str(file_path))
        x, y, z = smooth_blade(x_raw, y_raw, z_raw,
                               new_sections=smooth_sections, new_points_per_section=smooth_points)

        # ---------- 创建绘图 ----------
        fig = plt.figure(figsize=(12, 9), facecolor=STYLE_CONFIG["bg_color"])
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(STYLE_CONFIG["bg_color"])
        ls = LightSource(azdeg=STYLE_CONFIG["light_azdeg"], altdeg=STYLE_CONFIG["light_altdeg"])

        all_x, all_y, all_z = [], [], []
        base_rgb = np.array(to_rgb(STYLE_CONFIG["blade_material_color"]))
        blade_rgb_base = np.tile(base_rgb, (x.shape[0], x.shape[1], 1))

        # ---------- 绘制叶片 ----------
        loop_count = num_blades if rotate_copy else 1
        for i in range(loop_count):
            angle = (i * 360.0 / num_blades) if rotate_copy else 0.0
            x_rot, y_rot, z_rot = rotate_blade(x, y, z, angle, axis)
            all_x.append(x_rot)
            all_y.append(y_rot)
            all_z.append(z_rot)

            rgb_shaded = ls.shade_rgb(blade_rgb_base, elevation=z_rot,
                                      vert_exag=STYLE_CONFIG["vert_exag"],
                                      blend_mode=STYLE_CONFIG["blend_mode"])
            ax.plot_surface(x_rot, y_rot, z_rot, facecolors=rgb_shaded,
                            rstride=1, cstride=1, linewidth=0, antialiased=True,
                            shade=False, alpha=blade_alpha)

        # ---------- 内置轮毂与机匣 ----------
        hub_X_profile = np.array(
            [-340., -328.667, -317.333, -306., -294.667, -283.333, -272., -260.667, -249.333, -238.,
             -226.667, -215.333, -204., -192.667, -181.333, -170., -158.667, -147.333, -136.,
             -124.667, -113.333, -102., -90.667, -79.333, -68., -56.667, -45.333, -34.,
             -22.667, -11.333, 0.])
        hub_R_profile = np.array(
            [0., 19.315, 38.629, 57.944, 77.258, 96.573, 115.888, 135.202, 154.517, 153.667,
             153.192, 153.105, 153.296, 153.929, 155.076, 156.711, 158.457, 160.189, 161.906,
             163.611, 165.143, 166.468, 167.589, 168.514, 169.393, 169.721, 169.833, 170.449,
             171.073, 171.695, 172.317])

        shroud_X_profile = np.array(
            [-260.667, -249.333, -238., -226.667, -215.333, -204., -192.667, -181.333, -170.,
             -158.667, -147.333, -136., -124.667, -113.333, -102., -90.667, -79.333, -68.,
             -56.667, -45.333, -34., -22.667, -11.333, 0.])
        shroud_R_profile = np.array(
            [420.4, 418.727, 417.054, 415.38, 413.708, 412.034, 410.362, 408.688, 407.016,
             405.342, 403.669, 401.996, 400.323, 398.65, 396.977, 395.303, 393.63, 391.957,
             390.284, 388.611, 386.938, 385.265, 383.591, 381.919])

        # ---------- 构建圆柱网格 ----------
        hub_theta_points = 60
        theta = np.linspace(0.0, 2.0 * np.pi, hub_theta_points)

        def build_cylinder(x_g, r_g, t_g):
            X_cyl = x_g
            Y_cyl = r_g * np.cos(t_g)
            Z_cyl = r_g * np.sin(t_g)
            return X_cyl, Y_cyl, Z_cyl

        # hub
        theta_grid_hub, x_grid_hub = np.meshgrid(theta, hub_X_profile)
        hub_r_grid = np.tile(hub_R_profile[:, np.newaxis], (1, hub_theta_points))
        h_X, h_Y, h_Z = build_cylinder(x_grid_hub, hub_r_grid, theta_grid_hub)

        # shroud
        theta_grid_shroud, x_grid_shroud = np.meshgrid(theta, shroud_X_profile)
        shroud_r_grid = np.tile(shroud_R_profile[:, np.newaxis], (1, hub_theta_points))
        s_X, s_Y, s_Z = build_cylinder(x_grid_shroud, shroud_r_grid, theta_grid_shroud)

        # ---------- 根据旋转轴调整方向 ----------
        if axis == 'Z':
            h_X, h_Y, h_Z = rotate_blade(h_X, h_Y, h_Z, 90.0, 'Y')
            s_X, s_Y, s_Z = rotate_blade(s_X, s_Y, s_Z, 90.0, 'Y')
        elif axis == 'Y':
            h_X, h_Y, h_Z = rotate_blade(h_X, h_Y, h_Z, -90.0, 'Z')
            s_X, s_Y, s_Z = rotate_blade(s_X, s_Y, s_Z, -90.0, 'Z')

        # ---------- 绘制轮毂与机匣 ----------
        if show_hub:
            hub_rgb_matrix = np.tile(np.array(to_rgb(STYLE_CONFIG["hub_color"])), (h_Z.shape[0], h_Z.shape[1], 1))
            ax.plot_surface(h_X, h_Y, h_Z, facecolors=ls.shade_rgb(hub_rgb_matrix, h_Z,
                                                                   vert_exag=STYLE_CONFIG["vert_exag"],
                                                                   blend_mode=STYLE_CONFIG["blend_mode"]),
                            linewidth=0, antialiased=True, shade=False, alpha=hub_alpha)
            all_x.append(h_X)
            all_y.append(h_Y)
            all_z.append(h_Z)

        if show_shroud:
            shroud_rgb_matrix = np.tile(np.array(to_rgb(STYLE_CONFIG["shroud_color"])), (s_Z.shape[0], s_Z.shape[1], 1))
            ax.plot_surface(s_X, s_Y, s_Z, facecolors=ls.shade_rgb(shroud_rgb_matrix, s_Z,
                                                                   vert_exag=STYLE_CONFIG["vert_exag"],
                                                                   blend_mode=STYLE_CONFIG["blend_mode"]),
                            linewidth=0, antialiased=True, shade=False, alpha=shroud_alpha)
            ax.plot_wireframe(s_X, s_Y, s_Z, color='black', alpha=0.2, rstride=8, cstride=8, linewidth=0.3)
            all_x.append(s_X)
            all_y.append(s_Y)
            all_z.append(s_Z)

        # ---------- 坐标轴等比例 ----------
        if all_x:
            set_equal_3d(ax,
                         np.concatenate([a.flatten() for a in all_x]),
                         np.concatenate([a.flatten() for a in all_y]),
                         np.concatenate([a.flatten() for a in all_z]))

        ax.set_title(f"Metal Blade Simulation ({num_blades if rotate_copy else 1} Blades, Axis: {axis})",
                     fontsize=12, color="#333333", pad=15)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        ax.view_init(elev=30, azim=-60)
        plt.tight_layout()

        # ---------- 保存图片 ----------
        file_id = uuid.uuid4().hex
        filename = f"blade3d_advanced_{file_id}.png"
        filepath = IMAGE_DIR / filename
        plt.savefig(filepath, format='png', dpi=150, bbox_inches='tight', facecolor=STYLE_CONFIG["bg_color"])
        plt.close(fig)
        
        print(f"[高级可视化] 叶片绘制完毕: {filename}")
        
        image_path = f"/static/images/{filename}"
        
        # 生成 3D 数据 JSON（供前端 Three.js 交互）- 包含完整的叶片阵列、轮毂和机匣
        json_filename = f"blade3d_advanced_{file_id}.json"
        json_filepath = IMAGE_DIR / json_filename
        
        # 采样 3D 数据（减少数据量但保持足够细节）
        step = max(1, x.shape[0] // 8)
        chord_step = max(1, x.shape[1] // 15)
        
        # 构建所有旋转后的叶片数据
        blades_data = []
        loop_count = num_blades if rotate_copy else 1
        for i in range(loop_count):
            angle = (i * 360.0 / num_blades) if rotate_copy else 0.0
            x_rot, y_rot, z_rot = rotate_blade(x, y, z, angle, axis)
            blades_data.append({
                "x": x_rot[::step, ::chord_step].tolist(),
                "y": y_rot[::step, ::chord_step].tolist(),
                "z": z_rot[::step, ::chord_step].tolist()
            })
        
        # 构建轮毂数据（采样）
        hub_step = max(1, h_X.shape[0] // 10)
        hub_chord_step = max(1, h_X.shape[1] // 20)
        hub_data = {
            "x": h_X[::hub_step, ::hub_chord_step].tolist(),
            "y": h_Y[::hub_step, ::hub_chord_step].tolist(),
            "z": h_Z[::hub_step, ::hub_chord_step].tolist()
        } if show_hub else None
        
        # 构建机匣数据（采样）
        shroud_step = max(1, s_X.shape[0] // 10)
        shroud_chord_step = max(1, s_X.shape[1] // 20)
        shroud_data = {
            "x": s_X[::shroud_step, ::shroud_chord_step].tolist(),
            "y": s_Y[::shroud_step, ::shroud_chord_step].tolist(),
            "z": s_Z[::shroud_step, ::shroud_chord_step].tolist()
        } if show_shroud else None
        
        blade_3d_data = {
            "format": "advanced",  # 标记为高级格式
            "blades": blades_data,  # 所有叶片的旋转后数据
            "hub": hub_data,        # 轮毂数据
            "shroud": shroud_data,  # 机匣数据
            "metadata": {
                "num_blades": num_blades if rotate_copy else 1,
                "axis": axis,
                "show_hub": show_hub,
                "show_shroud": show_shroud,
                "source_file": str(file_path)
            }
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(blade_3d_data, f)
        print(f"[高级可视化] 3D数据已保存: {json_filename} (包含{loop_count}片叶片)")
        
        json_path = f"/static/images/{json_filename}"
        
        # 通过 SSE 发送可视化图片到前端
        if send_sse:
            try:
                from .blade_visualization import _report_visualization_progress
                _report_visualization_progress(
                    'visualization_image',
                    100,
                    '高级可视化完成',
                    image_path=image_path,
                    json_path=json_path,
                    label='高级可视化结果'
                )
            except Exception as e:
                print(f"[SSE回调错误] {e}")
        
        return {
            "image_path": image_path,
            "json_path": json_path,
            "description": f"压气机叶片高级3D可视化（{num_blades}片叶片，含轮毂和机匣）",
            "status": "success"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "image_path": "",
            "json_path": "",
            "description": "高级3D叶片可视化失败",
            "status": "failed"
        }


# ---------- 测试入口 ----------
if __name__ == "__main__":
    # 交互式展示
    plot_blade_profile_dynamic(
        file_path="E:/3stage_R0_Blade_gemoturbo_shape/folder_0/0_BladeIn.dat",
        rotate_copy=True,
        axis='X',
        num_blades=16,
        show_hub=True,
        show_shroud=True,
        smooth_sections=10,
        smooth_points=30,
        blade_alpha=0.8,
        hub_alpha=1.0,
        shroud_alpha=0.15
    )
