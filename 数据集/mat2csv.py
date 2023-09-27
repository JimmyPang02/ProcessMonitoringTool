import os
import scipy.io
import numpy as np

# 定义MAT文件夹和CSV文件夹的路径
mat_folder = "./"
csv_folder = "./csv"

# 确保输出CSV文件夹存在
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# 获取MAT文件夹中所有MAT文件的文件名列表
mat_files = [f for f in os.listdir(mat_folder) if f.endswith(".mat")]

# 遍历每个MAT文件
for mat_file in mat_files:
    mat_path = os.path.join(mat_folder, mat_file)
    data = scipy.io.loadmat(mat_path)  # 从MAT文件加载数据

    # 遍历MAT文件中的每个变量
    for var_name in data:
        if isinstance(data[var_name], np.ndarray):
            matrix = data[var_name]
            csv_file = f"{var_name}.csv"
            csv_path = os.path.join(csv_folder, csv_file)

            # 将矩阵保存为CSV文件
            np.savetxt(csv_path, matrix, delimiter=',')

print("转换完成！")
