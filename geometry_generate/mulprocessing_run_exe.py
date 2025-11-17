import os
import subprocess
import multiprocessing
from multiprocessing import Pool
import time
########多进程进行参数化
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
        result = subprocess.run([exe_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True, encoding='utf-8', errors='replace')
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
            # 打印当前处理的文件夹名称
            print(f"Currently processing: {file_name}")

    with open('shape_errors_6.2_5000.txt', 'w') as f:
        for file_name in errors:
            f.write(f"{file_name}\n")


if __name__ == "__main__":
    a=time.time()
    source_folder = os.path.dirname(os.path.abspath(__file__))
    file_names = []
    for i in range(6365,6375):
        file_names.append(f"folder_{i}")
    batch_size =15  # 每批次运行的进程数量

    run_in_batches(source_folder, file_names, batch_size)
    b=time.time()
    t=(b-a)/60
    print("#########################################\n")
    print(f"cost {t} minutes")