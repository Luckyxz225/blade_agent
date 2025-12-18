"""
blade_geometry.py - 叶片几何生成模块

功能：从设计生成的CSV文件生成DAT几何文件，供可视化模块使用

文件结构：
- geometry_generate/csv_files/    存储设计生成的CSV文件
- geometry_generate/dat_files/    存储生成的DAT文件
- geometry_generate/output_files/ 中间处理文件
- geometry_generate/dependent_files/ 依赖文件（exe、模板等）
"""

import os
import shutil
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import subprocess
from multiprocessing import Pool
import datetime
from typing import List, Dict, Optional, Tuple

# 获取项目根目录
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

# 默认路径配置
DEFAULT_PATHS = {
    'csv_dir': os.path.join(_project_root, 'geometry_generate', 'csv_files'),
    'dat_dir': os.path.join(_project_root, 'geometry_generate', 'dat_files'),
    'output_dir': os.path.join(_project_root, 'geometry_generate', 'output_files'),
    'dependent_dir': os.path.join(_project_root, 'geometry_generate', 'dependent_files'),
}

# 确保目录存在
for dir_path in DEFAULT_PATHS.values():
    os.makedirs(dir_path, exist_ok=True)


# ---------------- 文件复制 ----------------
def copy_files_to_new_folder(dependent_folder, destination_folder, file_names, index):
    """
    将指定依赖文件从依赖文件夹复制到目标文件夹，并生成BGin.dat文件
    """
    os.makedirs(destination_folder, exist_ok=True)

    for file_name in file_names:
        src = os.path.join(dependent_folder, file_name)
        dst = os.path.join(destination_folder, file_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            raise FileNotFoundError(f"文件 {file_name} 不存在: {src}")

    # 创建 BGin.dat
    bg_file = os.path.join(destination_folder, "BGin.dat")
    with open(bg_file, "w") as f:
        f.write(os.path.join(destination_folder, "processed_data.txt") + "\n")
        f.write(os.path.join(destination_folder, f"{index}.prt") + "\n")


# ---------------- 数据处理与插值 ----------------
def read_and_process_dat_file(folder, i, sample_file, start_index=1):
    file_path = os.path.join(folder, "Rotor-BH13.TXT")
    with open(file_path, 'r') as f:
        lines = [line.split() for line in f if line.strip()]
    max_cols = max(len(line) for line in lines)
    lines = [line + [''] * (max_cols - len(line)) for line in lines]
    df = pd.DataFrame(lines)

    df_sample = pd.read_csv(sample_file, sep='\t', header=0)
    idx = i - start_index
    rows = [26, 27, 28]
    cols_y = [7, 12, 13, 18, 20]
    cols_x = [4, 5]

    for r, start_col in zip(rows, [3, 10, 17]):
        df.iloc[r, cols_y] = df_sample.iloc[idx, start_col:start_col + 5].values
        df.iloc[r, cols_x] = -df_sample.iloc[idx, [start_col - 2, start_col - 1]].values

    output_file = os.path.join(folder, f"R0_{i}.TXT")
    df.to_csv(output_file, index=False, sep='\t', header=False)
    return df


def format_and_save(df, output_file_path):
    col_widths = [df[col].astype(str).map(len).max() for col in df.columns]
    lines = [" ".join(str(val).ljust(col_widths[i] + 1) for i, val in enumerate(row))
             for row in df.values]
    with open(output_file_path, 'w') as f:
        f.write("\n".join(lines))


def read_and_split_file(file_path, split_lines):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    split_lines.append(len(lines))
    return [
        pd.DataFrame([line.split() for line in lines[start:end] if line.strip()])
        for start, end in zip([0] + split_lines[:-1], split_lines)
    ]


def spline_interpolation(df, x_col, y_cols, x_new_values):
    x = df.iloc[:, x_col].astype(float)
    interpolated_data = {x_col: x_new_values}
    for col in y_cols:
        y = df.iloc[:, col].astype(float)
        interpolated_data[col] = CubicSpline(x, y)(x_new_values)
    interpolated_df = pd.DataFrame(interpolated_data)
    for col in df.columns:
        if col not in y_cols and col != x_col:
            interpolated_df[col] = df.iloc[2, col]
    df_combined = pd.concat([df, interpolated_df], ignore_index=True)
    df_combined.iloc[:, x_col] = df_combined.iloc[:, x_col].astype(float)
    df_combined = df_combined.sort_values(by=x_col).reset_index(drop=True)
    df_combined.iloc[:, 0] = df_combined.index + 1
    return df_combined


def replace_none_with_space(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    max_widths = [0] * len(lines[0].split())
    formatted_data = []
    for line in lines:
        tokens = []
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
            tokens.append(token)
        formatted_data.append(tokens)

    processed_lines = [
        "\t".join(f"{val:>{max_widths[i]}}" for i, val in enumerate(row)) + '\n'
        for row in formatted_data
    ]

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)


# ---------------- 运行exe（单进程版本，避免macOS兼容性问题） ----------------
def run_exe_file_single(folder, subfolder):
    """单进程运行exe文件"""
    new_dir = os.path.join(folder, subfolder)
    original_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        exe_path = os.path.join(new_dir, "Aerofoil_Con20190510.exe")
        
        # 检查是否在Windows上
        import platform
        if platform.system() == 'Windows':
            subprocess.run([exe_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           check=True, universal_newlines=True, encoding='utf-8', errors='replace')
        else:
            # 非Windows系统尝试使用Wine
            wine_path = shutil.which('wine')
            if wine_path:
                subprocess.run([wine_path, exe_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               check=True, universal_newlines=True, encoding='utf-8', errors='replace')
            else:
                print(f"[警告] 非Windows系统且未安装Wine，跳过exe执行: {subfolder}")
                return subfolder, "skip"
        return subfolder, None
    except Exception as e:
        print(f"[错误] 运行exe失败 ({subfolder}): {e}")
        return subfolder, str(e)
    finally:
        os.chdir(original_dir)


def run_exe_sequentially(folder, folder_names):
    """顺序运行所有exe（避免多进程在某些系统上的问题）"""
    errors = []
    for fn in folder_names:
        print(f"[几何生成] 处理: {fn}")
        _, err = run_exe_file_single(folder, fn)
        if err and err != "skip":
            errors.append(fn)
    if errors:
        error_file = os.path.join(folder, 'shape_errors.txt')
        with open(error_file, 'w') as f:
            for fn in errors:
                f.write(f"{fn}\n")
        print(f"[警告] {len(errors)} 个方案处理出错，详见 {error_file}")
    return errors


# ---------------- 生成 geomTurbo ----------------
def generate_geomturbo(index, template_path, output_root):
    with open(template_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines0 = lines[:467]
    lines1 = lines[2625:]
    log = ''.join(lines0)

    blade_file = os.path.join(output_root, f"folder_{index}", f"{index}_BladeIn_SM.dat")
    with open(blade_file, 'r', encoding='utf-8') as f:
        lines2 = f.readlines()[2:]

    for k, line in enumerate(lines2):
        nums = line.strip().split()
        nums = [nums[2], nums[1], nums[0]]
        lines2[k] = ' '.join(f"{float(num):.6f}" for num in nums) + '\n'

    log += "suction\nSECTIONAL\n7   \n"
    j = 0
    for sec in range(7):
        log += f"#section {sec + 1}\nXYZ\n101\n"
        log += ''.join(lines2[j:j + 101])
        j += 202
    log += "pressure\nSECTIONAL\n7\n"
    j = 101
    for sec in range(7):
        log += f"#section {sec + 1}\nXYZ\n101\n"
        log += ''.join(lines2[j:j + 101])
        j += 202
    log += ''.join(lines1)

    output_file = os.path.join(output_root, f"folder_{index}", f"{index}.geomTurbo")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(log)
    return output_file


# ================== 核心功能函数（供可视化智能体调用）==================

def get_latest_csv_file() -> Optional[str]:
    """
    获取最新的CSV文件路径
    
    通过文件名中的时间戳识别最新的设计任务CSV文件
    
    Returns:
        最新CSV文件的完整路径，如果没有则返回None
    """
    csv_dir = DEFAULT_PATHS['csv_dir']
    if not os.path.exists(csv_dir):
        return None
    
    csv_files = [f for f in os.listdir(csv_dir) if f.startswith('design_') and f.endswith('.csv')]
    if not csv_files:
        return None
    
    # 按文件名排序（时间戳在文件名中，所以字典序就是时间序）
    csv_files.sort(reverse=True)
    return os.path.join(csv_dir, csv_files[0])


def get_csv_scheme_count(csv_path: str) -> int:
    """
    获取CSV文件中的方案数量
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        方案数量
    """
    try:
        df = pd.read_csv(csv_path, sep='\t', header=0)
        return len(df)
    except Exception as e:
        print(f"[错误] 读取CSV文件失败: {e}")
        return 0


def generate_geometry_for_schemes(
    csv_path: str,
    scheme_indices: List[int],
    task_timestamp: str = None
) -> Dict[str, any]:
    """
    为指定的方案生成几何DAT文件
    
    这是可视化智能体调用的主要接口函数
    
    Args:
        csv_path: CSV文件路径
        scheme_indices: 方案索引列表（1-indexed），如 [1, 3, 5] 表示第1、3、5个方案
        task_timestamp: 任务时间戳（可选，用于命名输出文件）
    
    Returns:
        {
            "status": "success" | "error",
            "dat_files": {
                1: "/path/to/scheme1.dat",
                3: "/path/to/scheme3.dat",
                ...
            },
            "message": "描述信息",
            "errors": ["错误列表"]
        }
    """
    print(f"\n{'='*60}")
    print(f"[几何生成] 开始生成DAT文件")
    print(f"{'='*60}")
    print(f"  • CSV文件: {os.path.basename(csv_path)}")
    print(f"  • 目标方案: {scheme_indices}")
    
    result = {
        "status": "success",
        "dat_files": {},
        "message": "",
        "errors": []
    }
    
    # 验证CSV文件存在
    if not os.path.exists(csv_path):
        result["status"] = "error"
        result["message"] = f"CSV文件不存在: {csv_path}"
        return result
    
    # 获取方案总数
    total_schemes = get_csv_scheme_count(csv_path)
    if total_schemes == 0:
        result["status"] = "error"
        result["message"] = "CSV文件为空或格式错误"
        return result
    
    # 验证方案索引有效性
    valid_indices = [i for i in scheme_indices if 1 <= i <= total_schemes]
    if not valid_indices:
        result["status"] = "error"
        result["message"] = f"无效的方案索引，可用范围: 1-{total_schemes}"
        return result
    
    # 生成时间戳
    if task_timestamp is None:
        task_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 准备路径
    dependent_folder = DEFAULT_PATHS['dependent_dir']
    output_folder = os.path.join(DEFAULT_PATHS['output_dir'], f"task_{task_timestamp}")
    dat_folder = DEFAULT_PATHS['dat_dir']
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(dat_folder, exist_ok=True)
    
    # 依赖文件列表
    file_names = ['Aerofoil_Con20190510.exe', 'Rotor-BH13.TXT']
    
    # 检查依赖文件
    for fn in file_names:
        if not os.path.exists(os.path.join(dependent_folder, fn)):
            result["status"] = "error"
            result["message"] = f"缺少依赖文件: {fn}"
            return result
    
    # Step 1: 为每个方案创建工作文件夹并复制依赖文件
    print(f"\n[Step 1] 准备工作环境...")
    for idx in valid_indices:
        folder_i = os.path.join(output_folder, f"folder_{idx}")
        try:
            copy_files_to_new_folder(dependent_folder, folder_i, file_names, idx)
        except Exception as e:
            result["errors"].append(f"方案{idx}文件复制失败: {e}")
            continue
    
    # Step 2: 数据处理
    print(f"\n[Step 2] 处理设计参数...")
    for idx in valid_indices:
        folder_i = os.path.join(output_folder, f"folder_{idx}")
        try:
            # 创建临时CSV（只包含该方案）
            df_full = pd.read_csv(csv_path, sep='\t', header=0)
            df_single = df_full.iloc[[idx-1]]  # 0-indexed
            temp_csv = os.path.join(output_folder, f"temp_scheme_{idx}.csv")
            df_single.to_csv(temp_csv, sep='\t', index=False)
            
            # 处理数据
            df = read_and_process_dat_file(folder_i, idx, temp_csv, start_index=idx)
            output_file_path = os.path.join(folder_i, "processed_data.txt")
            format_and_save(df, output_file_path)

            split_lines = [26, 29]
            dataframes = read_and_split_file(output_file_path, split_lines.copy())
            x_col = 1
            y_cols = list(range(2, 28))
            x_new_values = [0.1, 0.3, 0.7, 0.9]
            df_interp = spline_interpolation(dataframes[1], x_col, y_cols, x_new_values)
            dataframes[0].iloc[22, 0] = df_interp.shape[0]
            new_data = pd.concat([dataframes[0], df_interp, dataframes[2]], axis=0)
            format_and_save(new_data, output_file_path)
            replace_none_with_space(output_file_path, output_file_path)
            print(f"  ✅ 方案{idx} 数据处理完成")
        except Exception as e:
            result["errors"].append(f"方案{idx}数据处理失败: {e}")
            print(f"  ❌ 方案{idx} 数据处理失败: {e}")
            continue
    
    # Step 3: 运行exe生成几何
    print(f"\n[Step 3] 生成叶片几何...")
    folder_names = [f"folder_{idx}" for idx in valid_indices]
    run_exe_sequentially(output_folder, folder_names)
    
    # Step 4: 复制DAT文件到dat_files目录，并使用规范命名
    print(f"\n[Step 4] 整理输出文件...")
    for idx in valid_indices:
        folder_i = os.path.join(output_folder, f"folder_{idx}")
        # 查找生成的DAT文件（BladeIn.dat 或 类似名称）
        dat_source = None
        for fn in os.listdir(folder_i):
            if fn.endswith('_BladeIn.dat') or fn.endswith('BladeIn.dat'):
                dat_source = os.path.join(folder_i, fn)
                break
        
        if dat_source and os.path.exists(dat_source):
            # 目标文件名：{timestamp}_scheme{idx}.dat
            dat_dest_name = f"{task_timestamp}_scheme{idx}.dat"
            dat_dest = os.path.join(dat_folder, dat_dest_name)
            shutil.copy2(dat_source, dat_dest)
            result["dat_files"][idx] = dat_dest
            print(f"  ✅ 方案{idx} DAT文件: {dat_dest_name}")
        else:
            result["errors"].append(f"方案{idx}未生成DAT文件")
            print(f"  ⚠️ 方案{idx} 未找到DAT文件")
    
    # 汇总结果
    if result["dat_files"]:
        result["message"] = f"成功生成 {len(result['dat_files'])} 个方案的DAT文件"
    else:
        result["status"] = "error"
        result["message"] = "未能生成任何DAT文件"
    
    print(f"\n{'='*60}")
    print(f"[几何生成] 完成 - {result['message']}")
    print(f"{'='*60}\n")
    
    return result


def generate_geometry_from_params(
    params_21d: List[float],
    scheme_name: str = "custom"
) -> Dict[str, any]:
    """
    从21维参数直接生成几何DAT文件（用于用户手动输入参数的场景）
    
    Args:
        params_21d: 21维叶片参数列表
        scheme_name: 方案名称（用于文件命名）
    
    Returns:
        {
            "status": "success" | "error",
            "dat_file": "/path/to/file.dat",
            "message": "描述信息"
        }
    """
    print(f"\n[几何生成] 从21维参数生成DAT文件...")
    
    result = {
        "status": "success",
        "dat_file": None,
        "message": ""
    }
    
    # 验证参数
    if len(params_21d) != 21:
        result["status"] = "error"
        result["message"] = f"参数数量错误，需要21个，实际{len(params_21d)}个"
        return result
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建临时CSV文件
    csv_dir = DEFAULT_PATHS['csv_dir']
    temp_csv = os.path.join(csv_dir, f"custom_{timestamp}_{scheme_name}.csv")
    
    # 写入CSV
    header = 'Index\t' + '\t'.join([
        "Angle_in1", "Angle_out1", "Chord1", "THmax/CH1", "THmaxP1", "SWA1", "BOWA1",
        "Angle_in2", "Angle_out2", "Chord2", "THmax/CH2", "THmaxP2", "SWA2", "BOWA2",
        "Angle_in3", "Angle_out3", "Chord3", "THmax/CH3", "THmaxP3", "SWA3", "BOWA3"
    ])
    
    with open(temp_csv, 'w') as f:
        f.write(header + '\n')
        f.write('1\t' + '\t'.join([f"{p:.6f}" for p in params_21d]) + '\n')
    
    # 调用几何生成
    gen_result = generate_geometry_for_schemes(temp_csv, [1], timestamp)
    
    if gen_result["status"] == "success" and 1 in gen_result["dat_files"]:
        result["dat_file"] = gen_result["dat_files"][1]
        result["message"] = "成功从21维参数生成DAT文件"
    else:
        result["status"] = "error"
        result["message"] = gen_result.get("message", "生成失败")
    
    return result


# ---------------- 运行示例 ----------------
if __name__ == "__main__":
    # 测试获取最新CSV
    latest = get_latest_csv_file()
    print(f"最新CSV文件: {latest}")
    
    if latest:
        # 测试生成指定方案的几何
        result = generate_geometry_for_schemes(latest, [1, 2])
        print(f"生成结果: {result}")
