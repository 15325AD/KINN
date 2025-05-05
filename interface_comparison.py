import numpy as np
import matplotlib.pyplot as plt
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
    inter_path = f"{base_path}{folder}/interface.npy"
    inter_data = np.load(inter_path)
    plt.plot(inter_data[0], inter_data[1], label=folder, color=colors[i], linestyle=linestyles[i], markersize=5)


inter_pos = inter_data[0]
dudyi_e = 0.5/np.sqrt(inter_pos)# the exact strain on the interface
plt.plot(inter_pos, dudyi_e, label='Exact', color='r', marker='o')

plt.xlabel('x')
plt.ylabel('应变张量$\epsilon_{32}$',fontproperties=font_prop)
plt.title('不同算法对界面上的奇异应变的比较',fontproperties=font_prop)
plt.legend()
plt.savefig('./results/Crack_strain_inter.pdf', dpi = 300)
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
    inter_path = f"{base_path}{folder}/interface.npy"
    inter_data = np.load(inter_path)
    plt.plot(inter_data[0], inter_data[1], label=folder+'_uni', color=colors[i], linestyle=linestyles[0], markersize=5)

    inter_path = f"{base_path}{folder}/interface_tri.npy"
    inter_data = np.load(inter_path)
    plt.plot(inter_data[0], inter_data[1], label=folder+'_tri', color=colors[i], linestyle=linestyles[1], markersize=5)
inter_pos = inter_data[0]
dudyi_e = 0.5/np.sqrt(inter_pos)# the exact strain on the interface
plt.plot(inter_pos, dudyi_e, label='Exact', color='r', marker='o')

plt.title('DEM 和 KINN-DEM 中不同数值积分对界面上的奇异应变的比较',fontproperties=font_prop)
plt.xlabel('x',fontproperties=font_prop)
plt.ylabel('应变张量$\epsilon_{32}$',fontproperties=font_prop)
plt.legend()
plt.savefig('./results/Crack_strain_inter_tri.pdf', dpi = 300)
plt.show()