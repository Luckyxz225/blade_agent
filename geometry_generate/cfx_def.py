####先使用cfx_pre录制一个pre文件，再利用以下命令批处理def文件
import subprocess
from multiprocessing import Pool
def run_cfx5export(args):
    i, p=args
    # 读取原始pre文件
    with open(r"E:\LLM\myAgent-main_V2\myAgent-main\geometry_generate\1.5_pre.txt", 'r', encoding='utf-8') as file:
        content = file.read()
        # 对文件地址进行查找和替换
    out_relative_pressure=f"Relative Pressure = {p} [Pa]" #设置出口背压
    old_string="E:/3stage_R0_Blade_gemoturbo_shape/folder_1/1/0_4000.def"
    new_string=f"E:\LLM\myAgent-main_V2\myAgent-main\geometry_generate/folder_{i}/{i}/{i}_{p}.def"#设置保存地址
    new_cgns=f"gtmImport filename=E:\LLM\myAgent-main_V2\myAgent-main\geometry_generate/folder_{i}/{i}/{i}.cgns"
    content = content.replace("gtmImport filename=E:/3stage_R0_Blade_gemoturbo_shape/folder_1/1/1.cgns",new_cgns)#替换新网格
    content = content.replace("Relative Pressure = 4000 [Pa]", out_relative_pressure)#替换出口背压
    new_content = content.replace(old_string, new_string)
    # 如果你想要覆盖原始文件，可以将 'new_file.txt' 改为 'original.txt'
    with open(f"E:\LLM\myAgent-main_V2\myAgent-main\geometry_generate/folder_{i}/{i}/{i}.pre", 'w', encoding='utf-8') as file:
        file.write(new_content)

    # 捕获标准输出和标准错误输出
    subprocess.run(
       [
        "C:/Program Files/ANSYS Inc/v212/CFX/bin/cfx5pre.exe",
        "-batch", f"E:\LLM\myAgent-main_V2\myAgent-main\geometry_generate/folder_{i}/{i}/{i}.pre"
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)

    # 如果需要，可以处理标准输出
    print("\n" + "#" * 15)
    print(f"{i} executed successfully.")



def run_in_batches(start,end,batch_size,pressure=6000):
    """
    :param start: 起始文件夹
    :param end:
    :param batch_size: 批处理数量
    :param pressure: 出口背压
    :return:
    """
    errors = []
    for i in range(start, end,batch_size):
        batch =range(i,i+batch_size)
        if i+batch_size>end:
            batch = range(i, end)
        with Pool(processes=batch_size) as pool:
            results = pool.map(run_cfx5export, [(i, pressure) for i in batch])


if __name__ == "__main__":
    run_in_batches(1,10,20,pressure=6000)
    # run_in_batches(938, 939, 1, pressure=6000)

