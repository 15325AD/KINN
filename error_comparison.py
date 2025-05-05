import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Windows系统示例路径（换成实际路径）
font_path = 'C:/Windows/WinSxS/amd64_microsoft-windows-font-truetype-simhei_31bf3856ad364e35_10.0.19041.1_none_aa18c3e2137269cf/simhei.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False  # 使用ASCII减号
plt.rcParams['font.family'] = font_prop.get_name()
# 定义文件路径
folders = [
    'CPINNs_MLP_penalty',
    'BINN_MLP',
    'DEM_MLP_rbf',
    'KINN_CPINN_penalty',
    'BINN_KAN',
    'KINN_DEM_rbf'
]
base_path = './results/'  # 替换为你的文件夹路径

colors = ['b', 'g', 'r', 'b', 'g', 'r']  # 蓝，绿，红，青，洋红，黄
linestyles = ['--', '--', '--', '-', '-', '-']


# 加载数据并绘图
plt.figure(figsize=(10, 6))
for i, folder in enumerate(folders):
    error_path = f"{base_path}{folder}/error.npy"
    error_data = np.load(error_path)
    plt.semilogy(error_data, label=folder, color=colors[i], linestyle=linestyles[i], markersize=5)



plt.title('不同算法的相对误差比较',fontproperties=font_prop)
plt.xlabel('迭代步数',fontproperties=font_prop)
plt.ylabel('相对误差',fontproperties=font_prop)
plt.legend()
plt.savefig('./results/Crack_error.pdf', dpi = 300)
plt.show()



# 定义文件路径
folders = [
    'DEM_MLP_rbf',
    'KINN_DEM_rbf'
]
base_path = './results/'  # 替换为你的文件夹路径

colors = ['b', 'g']  # 蓝，绿，红，青，洋红，黄
linestyles = ['--', '-']


# 加载数据并绘图
plt.figure(figsize=(10, 6))
for i, folder in enumerate(folders):
    error_path = f"{base_path}{folder}/error.npy"
    error_data = np.load(error_path)
    plt.semilogy(error_data, label=folder+'_uni', color=colors[i], linestyle=linestyles[0], markersize=5)

    error_path = f"{base_path}{folder}/error_tri.npy"
    error_data = np.load(error_path)
    plt.semilogy(error_data, label=folder+'_tri', color=colors[i], linestyle=linestyles[1], markersize=5)

plt.title('DEM 和 KINN-DEM 中不同数值积分的比较',fontproperties=font_prop)
plt.xlabel('迭代步数',fontproperties=font_prop)
plt.ylabel('相对误差',fontproperties=font_prop)
plt.legend()
plt.savefig('./results/Crack_error_tri.pdf', dpi = 300)
plt.show()