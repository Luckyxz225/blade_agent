import os


def generate_geomturbo(index, template_path, output_root):
    """
    根据几何文件模板替换叶片数据，生成 geomTurbo 文件

    :param index: 几何编号 (e.g. 6365)
    :param template_path: 模板 geomTurbo 文件路径 (e.g. "E:/3stage_R0_Blade_gemoturbo_shape/1.5.geomTurbo")
    :param output_root: 输出文件夹根目录 (e.g. "E:/Compressor_rotor_CDDPM/Performance/output")
    """
    # 读取模板
    with open(template_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines0 = lines[:467]
    lines1 = lines[2625:]
    log = ''.join(lines0)

    # 读取叶片数据
    blade_file = os.path.join(output_root, f"folder_{index}", f"{index}_BladeIn_SM.dat")
    with open(blade_file, 'r', encoding='utf-8') as f:
        lines2 = f.readlines()[2:]  # 去掉前两行

    # 转换数据格式
    for k, line in enumerate(lines2):
        numbers_str = line.strip().split()
        numbers_str = [numbers_str[2], numbers_str[1], numbers_str[0]]  # 调整为 x,y,z
        converted_line = ' '.join([f"{float(num):.6f}" for num in numbers_str])
        lines2[k] = converted_line + '\n'

    # 拼接 suction 面
    log += "suction\nSECTIONAL\n7   \n"
    j = 0
    for sec in range(7):
        log += f"#section {sec + 1}\nXYZ\n101\n"
        log += ''.join(lines2[j:j + 101])
        j += 202

    # 拼接 pressure 面
    log += "pressure\nSECTIONAL\n7\n"
    j = 101
    for sec in range(7):
        log += f"#section {sec + 1}\nXYZ\n101\n"
        log += ''.join(lines2[j:j + 101])
        j += 202

    log += ''.join(lines1)

    # 保存输出文件
    output_file = os.path.join(output_root, f"folder_{index}", f"{index}.geomTurbo")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(log)

    print(f"geomTurbo {index} finished")


if __name__ == "__main__":
    template_path = r"E:\3stage_R0_Blade_gemoturbo_shape\1.5.geomTurbo"
    output_root = os.path.dirname(os.path.abspath(__file__))

    for i in range(6365, 6375):
        generate_geomturbo(i, template_path, output_root)
