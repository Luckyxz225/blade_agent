import os
import subprocess
from multiprocessing import Pool, Manager


def worker(i, error_log_path, lock):
    """
    单个任务执行函数
    """
    folder_path = f"E:/LLM/myAgent-main_V2/myAgent-main/geometry_generate/folder_{i}/{i}"  ##几何保存地址
    save_mesh_path = f"{folder_path}/{i}"
    script_path = f"E:/LLM/myAgent-main_V2/myAgent-main/geometry_generate/folder_{i}/blade{i}.py" ###脚本保存地址
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


if __name__ == "__main__":
    # 使用示例
    run_numeca(start_index=1, end_index=1000, num_processes=10)
