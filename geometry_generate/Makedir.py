import os
import shutil


def copy_files_to_new_folder(source_folder, destination_folder, file_names):
    """
    将指定的文件从源文件夹复制到目标文件夹。

    :param source_folder: 源文件夹路径
    :param destination_folder: 目标文件夹路径
    :param file_names: 要复制的文件名列表
    """
    # 如果目标文件夹不存在，创建它

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历文件名列表，将文件从源文件夹复制到目标文件夹
    for file_name in file_names[:3]:
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)

        # 检查文件是否存在
        if os.path.exists(source_file):
            shutil.copy2(source_file, destination_file)
            print(f"文件 {file_name} 已复制到 {destination_folder}")
        else:
            print(f"文件 {file_name} 不存在于 {source_folder}")
    # 写入保存目录文件
    with open(os.path.join(destination_folder, file_names[3]),"w") as file:
        file.write(os.path.join(destination_folder, "processed_data.txt")+"\n")
        file.write(os.path.join(destination_folder, f"{i}.prt") + "\n")

source_folder = os.path.dirname(os.path.abspath(__file__))  # 源文件夹路径
for i in range(6365,6375):
    destination_folder = os.path.join(source_folder, f"folder_{i}")  # 目标文件夹路径
    file_names = ['Aerofoil_Con20190510.exe', 'Rotor-BH13.TXT','sampling_results_final.txt', 'BGin.dat']  # 要复制的文件名列表
    copy_files_to_new_folder(source_folder, destination_folder, file_names)
