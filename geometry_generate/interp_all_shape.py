import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import os


def read_and_process_dat_file(source_folder, i, sample_file, start_index=6365):
    """
    读取原始TXT文件并替换指定行的参数。

    :param source_folder: 输入文件所在的文件夹路径
    :param i: 当前样本索引
    :param sample_file: 采样后的数值文件路径
    :param start_index: 数据起始偏移量（默认为6365）
    :return: 替换后的DataFrame
    """
    file_name = "Rotor-BH13.TXT"
    file_path = os.path.join(source_folder, file_name)

    # 读取原始文件，删除空行
    with open(file_path, 'r') as file:
        lines = [line.split() for line in file if line.strip()]

    # 填充缺失数据到相同列数
    max_cols = max(len(line) for line in lines)
    processed_lines = [line + [''] * (max_cols - len(line)) for line in lines]

    df = pd.DataFrame(processed_lines)
    df1 = pd.read_csv(sample_file, sep='\t', header=0)  # 采样后的数值

    idx = i - start_index  # 修正索引
    rows = [26, 27, 28]  # 要替换的行
    cols_y = [7, 12, 13, 18, 20]  # 替换的目标列
    cols_x = [4, 5]  # 需要取负号的列

    # 批量替换对应行
    for r, start_col in zip(rows, [3, 10, 17]):
        df.iloc[r, cols_y] = df1.iloc[idx, start_col:start_col + 5].values
        df.iloc[r, cols_x] = -df1.iloc[idx, [start_col - 2, start_col - 1]].values

    # 输出新文件
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

    # 其余未插值的列用第3行的数据填充
    for col in df.columns:
        if col not in y_cols and col != x_col:
            interpolated_df[col] = df.iloc[2, col]

    # 合并并排序
    df_combined = pd.concat([df, interpolated_df], ignore_index=True)
    df_combined.iloc[:, x_col] = df_combined.iloc[:, x_col].astype(float)
    df_combined = df_combined.sort_values(by=x_col).reset_index(drop=True)
    df_combined.iloc[:, 0] = df_combined.index + 1  # 更新第一列为行号

    return df_combined


def replace_none_with_space(input_file_path, output_file_path):
    """替换None/nan并格式化对齐保存"""
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 检查每列的最大宽度
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


if __name__ == "__main__":
    base_folder = os.path.dirname(os.path.abspath(__file__))
    sample_file = os.path.join(base_folder, "sampling_results_final.txt")

    for i in range(6365, 6375):
        source_folder = os.path.join(base_folder, f"folder_{i}")
        df = read_and_process_dat_file(source_folder, i, sample_file)

        # 保存处理后的文件
        output_file_path = os.path.join(source_folder, "processed_data.txt")
        format_and_save(df, output_file_path)

        # 分割文件
        split_lines = [26, 29]
        dataframes = read_and_split_file(output_file_path, split_lines)

        # 样条插值
        x_new_values = [0.1, 0.3, 0.7, 0.9]
        x_col = 1
        y_cols = range(2, 28)
        df_interpolated = spline_interpolation(dataframes[1], x_col, y_cols, x_new_values)

        # 更新行数
        dataframes[0].iloc[22, 0] = df_interpolated.shape[0]

        # 合并并保存最终结果
        new_data = pd.concat([dataframes[0], df_interpolated, dataframes[2]], axis=0)
        format_and_save(new_data, output_file_path)

        # 替换None并对齐
        replace_none_with_space(output_file_path, output_file_path)

        print(f"folder_{i} finished")
