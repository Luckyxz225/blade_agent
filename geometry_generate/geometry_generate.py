import os
import shutil
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import subprocess
import multiprocessing
from multiprocessing import Pool, Manager
import time
from agents import function_tool

def copy_files_to_new_folder(source_folder, destination_folder, file_names, index):
    """
    将指定的文件从源文件夹复制到目标文件夹。

    :param source_folder: 源文件夹路径
    :param destination_folder: 目标文件夹路径
    :param file_names: 要复制的文件名列表
    :param index: 当前文件夹索引
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file_name in file_names[:3]:
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)

        if os.path.exists(source_file):
            shutil.copy2(source_file, destination_file)
            print(f"文件 {file_name} 已复制到 {destination_folder}")
        else:
            print(f"文件 {file_name} 不存在于 {source_folder}")
    
    with open(os.path.join(destination_folder, file_names[3]), "w") as file:
        file.write(os.path.join(destination_folder, "processed_data.txt") + "\n")
        file.write(os.path.join(destination_folder, f"{index}.prt") + "\n")


def read_and_process_dat_file(source_folder, i, sample_file, start_index=1):
    """
    读取原始TXT文件并替换指定行的参数。

    :param source_folder: 输入文件所在的文件夹路径
    :param i: 当前样本索引
    :param sample_file: 采样后的数值文件路径
    :param start_index: 数据起始偏移量（默认为1）
    :return: 替换后的DataFrame
    """
    file_name = "Rotor-BH13.TXT"
    file_path = os.path.join(source_folder, file_name)

    with open(file_path, 'r') as file:
        lines = [line.split() for line in file if line.strip()]

    max_cols = max(len(line) for line in lines)
    processed_lines = [line + [''] * (max_cols - len(line)) for line in lines]

    df = pd.DataFrame(processed_lines)
    df1 = pd.read_csv(sample_file, sep='\t', header=0)

    idx = i - start_index
    rows = [26, 27, 28]
    cols_y = [7, 12, 13, 18, 20]
    cols_x = [4, 5]

    for r, start_col in zip(rows, [3, 10, 17]):
        df.iloc[r, cols_y] = df1.iloc[idx, start_col:start_col + 5].values
        df.iloc[r, cols_x] = -df1.iloc[idx, [start_col - 2, start_col - 1]].values

    output_file_path = os.path.join(source_folder, f"R0_{i}.TXT")
    df.to_csv(output_file_path, index=False, sep='\t', header=False)
    return df


def format_and_save(df, output_file_path):
    """对齐保存DataFrame为TXT文件"""
    col_widths = [df[col].astype(str).map(len).max() for col in df.columns]
    formatted_lines = [
        " ".join(str(val).ljust(col_widths[i] + 1) for i, val in enumerate(row))
        for row in df.values
    ]
    with open(output_file_path, 'w') as file:
        file.write("\n".join(formatted_lines))


def read_and_split_file(file_path, split_lines):
    """按指定行号分割文件为多个DataFrame"""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    split_lines.append(len(lines))
    return [
        pd.DataFrame([line.split() for line in lines[start:end] if line.strip()])
        for start, end in zip([0] + split_lines[:-1], split_lines)
    ]


def spline_interpolation(df, x_col, y_cols, x_new_values):
    """对指定列进行样条插值，并合并到原DataFrame"""
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
    """替换None/nan并格式化对齐保存"""
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    max_widths = [0] * len(lines[0].split())
    formatted_data = []

    for line in lines:
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
        formatted_data.append(numbers)

    processed_lines = [
        "\t".join(f"{val:>{max_widths[i]}}" for i, val in enumerate(row)) + '\n'
        for row in formatted_data
    ]

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.writelines(processed_lines)


def run_exe_file(args):
    """
    :param args: (source_folder, file_name)元组
    :return: 调用叶片造型程序，返回文件夹名称和错误标志（如果有）
    """
    source_folder, file_name = args
    new_dir = os.path.join(source_folder, file_name)
    try:
        os.chdir(new_dir)
        exe_path = os.path.join(new_dir, "Aerofoil_Con20190510.exe")
        result = subprocess.run([exe_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               universal_newlines=True, check=True, encoding='utf-8', errors='replace')
        return file_name, None
    except subprocess.CalledProcessError as e:
        return file_name, True
    except Exception as e:
        return file_name, True


def run_in_batches(source_folder, file_names, batch_size):
    """
    分批次运行run_exe_file函数

    :param source_folder: 源文件夹路径
    :param file_names: 文件夹名称列表
    :param batch_size: 每批次运行的进程数量
    """
    errors = []

    for i in range(0, len(file_names), batch_size):
        batch = file_names[i:i + batch_size]
        with Pool(processes=batch_size) as pool:
            results = pool.map(run_exe_file, [(source_folder, file_name) for file_name in batch])
        for file_name, error in results:
            if error:
                errors.append(file_name)
            print(f"Currently processing: {file_name}")

    with open('shape_errors.txt', 'w') as f:
        for file_name in errors:
            f.write(f"{file_name}\n")


def generate_geomturbo(index, template_path, output_root):
    """
    根据几何文件模板替换叶片数据，生成 geomTurbo 文件

    :param index: 几何编号 (e.g. 6365)
    :param template_path: 模板 geomTurbo 文件路径
    :param output_root: 输出文件夹根目录
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines0 = lines[:467]
    lines1 = lines[2625:]
    log = ''.join(lines0)

    blade_file = os.path.join(output_root, f"folder_{index}", f"{index}_BladeIn_SM.dat")
    with open(blade_file, 'r', encoding='utf-8') as f:
        lines2 = f.readlines()[2:]

    for k, line in enumerate(lines2):
        numbers_str = line.strip().split()
        numbers_str = [numbers_str[2], numbers_str[1], numbers_str[0]]
        converted_line = ' '.join([f"{float(num):.6f}" for num in numbers_str])
        lines2[k] = converted_line + '\n'

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

    print(f"geomTurbo {index} finished")


def worker(i, error_log_path, lock):
    """
    单个任务执行函数
    """
    folder_path = f"E:/LLM/myAgent-main_V2/myAgent-main/geometry_generate/folder_{i}/{i}"
    save_mesh_path = f"{folder_path}/{i}"
    script_path = f"E:/LLM/myAgent-main_V2/myAgent-main/geometry_generate/folder_{i}/blade{i}.py"
    flag_file_path = f"E:/LLM/myAgent-main_V2/myAgent-main/geometry_generate/folder_{i}/completed.flag"

    # 如果已经完成，跳过
    if os.path.exists(flag_file_path):
        print(f"{i} 已完成，跳过...")
        return

    script_content = f"""
import os
script_version(2.2)
a5_open_project("E:/1.5stage/1.5_CANSHUHUA/1.5.trb")
row(1).load_geometry("E:/LLM/myAgent-main_V2/myAgent-main/geometry_generate/folder_{i}/{i}.geomTurbo")
select_all_rows()
a5_generate_b2b()
a5_start_3d_generation()
a5_save_mesh("{save_mesh_path}")
    """

    os.makedirs(folder_path, exist_ok=True)

    with open(script_path, "w") as file:
        file.write(script_content)

    try:
        result = subprocess.run(
            [
                r"D:/NUMECA_SOFTWARE/fine171/bin64/iggx86_64.exe",
                "igg", "-autogrid5", "-batch", "-script", script_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True
        )
        if result.stderr:
            print("Error occurred:", result.stderr)
            with lock:
                with open(error_log_path, "a") as error_file:
                    error_file.write(f"Error in processing folder {i}: {result.stderr}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error in processing folder {i}: {e}")
        with lock:
            with open(error_log_path, "a") as error_file:
                error_file.write(f"Error in processing folder {i}: {e}\n")

    # 写入完成标记
    with open(flag_file_path, "w") as f:
        f.write("Completed")

    print("#" * 100)
    print(f"{i}.cgns finished")
    print("#" * 100)


def run_numeca(start_index, end_index, num_processes=8):
    """
    批量运行 NUMECA 任务
    :param start_index: 起始序号（包含）
    :param end_index:   结束序号（不包含）
    :param num_processes: 并行进程数
    """
    error_log_path = "error_log.txt"

    with Manager() as manager:
        lock = manager.Lock()
        args_list = [(i, error_log_path, lock) for i in range(start_index, end_index)]
        with Pool(processes=num_processes) as pool:
            pool.starmap(worker, args_list)


def generate_compressor_geometry(start_index=1, end_index=2, processes_number=8,template_path=r"E:\3stage_R0_Blade_gemoturbo_shape\1.5.geomTurbo",sample_file_path=None,skip_steps=None):
    """
    用于创建压气机几何
    
    :param start_index: 起始索引 (默认: 1)
    :param end_index: 结束索引 (默认: 2)
    :param batch_size: 多进程批次大小 (默认: 15)
    :param template_path: geomTurbo模板文件路径，默认即可
    :param sample_file_path: 压气机设计参数保存地址
    :param skip_steps: 跳过的步骤列表 (如: [1, 3, 4])
    :param numeca_processes: NUMECA并行进程数 (默认: 8)
    """
    # ============ 可配置参数 ============
    START_INDEX = start_index
    END_INDEX = end_index
    BATCH_SIZE = processes_number
    TEMPLATE_PATH = template_path
    SKIP_STEPS = skip_steps if skip_steps is not None else []
    NUMECA_PROCESSES = processes_number
    
    # 设置采样文件路径
    if sample_file_path is None:
        base_folder = os.path.dirname(os.path.abspath(__file__))
        SAMPLE_FILE = os.path.join(base_folder, "sampling_results_final.txt")
    else:
        SAMPLE_FILE = sample_file_path
    
    FILE_NAMES = ['Aerofoil_Con20190510.exe', 'Rotor-BH13.TXT', os.path.basename(SAMPLE_FILE), 'BGin.dat']
    
    # 数据处理参数
    SPLIT_LINES = [26, 29]
    X_NEW_VALUES = [0.1, 0.3, 0.7, 0.9]
    X_COL = 1
    Y_COLS = range(2, 28)
    # ===================================
    
    print(f"几何生成配置:")
    print(f"  起始索引: {START_INDEX}")
    print(f"  结束索引: {END_INDEX}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  模板路径: {TEMPLATE_PATH}")
    print(f"  采样文件: {SAMPLE_FILE}")
    print(f"  NUMECA进程数: {NUMECA_PROCESSES}")
    print(f"  跳过步骤: {SKIP_STEPS if SKIP_STEPS else '无'}")
    print("-" * 50)
    
    start_time = time.time()
    base_folder = os.path.dirname(os.path.abspath(__file__))
    
    # 步骤1: 创建文件夹并复制文件
    if 1 not in SKIP_STEPS:
        print("=== 步骤1: 创建文件夹并复制文件 ===")
        for i in range(START_INDEX, END_INDEX):
            destination_folder = os.path.join(base_folder, f"folder_{i}")
            copy_files_to_new_folder(base_folder, destination_folder, FILE_NAMES, i)
    else:
        print("=== 跳过步骤1: 创建文件夹并复制文件 ===")
    
    # 步骤2: 数据处理和插值
    if 2 not in SKIP_STEPS:
        print("\n=== 步骤2: 数据处理和插值 ===")
        for i in range(START_INDEX, END_INDEX):
            source_folder = os.path.join(base_folder, f"folder_{i}")
            df = read_and_process_dat_file(source_folder, i, SAMPLE_FILE, START_INDEX)
            
            output_file_path = os.path.join(source_folder, "processed_data.txt")
            format_and_save(df, output_file_path)
            
            dataframes = read_and_split_file(output_file_path, SPLIT_LINES.copy())
            
            df_interpolated = spline_interpolation(dataframes[1], X_COL, Y_COLS, X_NEW_VALUES)
            
            dataframes[0].iloc[22, 0] = df_interpolated.shape[0]
            
            new_data = pd.concat([dataframes[0], df_interpolated, dataframes[2]], axis=0)
            format_and_save(new_data, output_file_path)
            
            replace_none_with_space(output_file_path, output_file_path)
            print(f"folder_{i} 数据处理完成")
    else:
        print("=== 跳过步骤2: 数据处理和插值 ===")
    
    # 步骤3: 多进程进行几何生成
    if 3 not in SKIP_STEPS:
        print("\n=== 步骤3: 多进程执行exe文件 ===")
        folder_names = [f"folder_{i}" for i in range(START_INDEX, END_INDEX)]
        run_in_batches(base_folder, folder_names, BATCH_SIZE)
    else:
        print("=== 跳过步骤3: 多进程执行exe文件 ===")
    
    # 步骤4: 生成geomTurbo文件
    if 4 not in SKIP_STEPS:
        print("\n=== 步骤4: 生成geomTurbo文件 ===")
        for i in range(START_INDEX, END_INDEX):
            generate_geomturbo(i, TEMPLATE_PATH, base_folder)
    else:
        print("=== 跳过步骤4: 生成geomTurbo文件 ===")
    
    # 步骤5: 运行NUMECA生成CGNS文件
    if 5 not in SKIP_STEPS:
        print("\n=== 步骤5: 运行NUMECA生成CGNS文件 ===")
        run_numeca(START_INDEX, END_INDEX, NUMECA_PROCESSES)
        print("NUMECA CGNS文件生成完成")
    else:
        print("=== 跳过步骤5: 运行NUMECA生成CGNS文件 ===")
    
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"\n#########################################")
    print(f"总耗时: {total_time:.2f} 分钟")
    print("所有步骤执行完成!")


if __name__ == "__main__":
    # 默认参数运行
    # 示例：使用不同参数运行
    generate_compressor_geometry(start_index=1, end_index=10, sample_file_path=r"E:\LLM\myAgent-main_V2\myAgent-main\15.0_0.88_2.0_10.csv")
