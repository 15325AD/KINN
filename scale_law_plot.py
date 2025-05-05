import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# Windows系统示例路径（换成实际路径）
font_path = 'C:/Windows/WinSxS/amd64_microsoft-windows-font-truetype-simhei_31bf3856ad364e35_10.0.19041.1_none_aa18c3e2137269cf/simhei.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = True  # 解决负号显示问题
plt.rcParams['font.family'] = font_prop.get_name()


# 2. 加载数据
def load_and_plot(data_path, x_label, plot_name):
    loaded_data = np.load(data_path, allow_pickle=True).item()
    print(f"\n加载数据: {plot_name}")
    print(f"x轴数据 ({x_label}): {loaded_data[list(loaded_data.keys())[1]]}")
    print(f"网络结构: {loaded_data['layers_list']}")
    print(f"误差数据长度: {len(loaded_data['errors'])}")

    # 3. 计算基本参数
    num_layers = len(loaded_data['layers_list'])
    x_values = loaded_data[list(loaded_data.keys())[1]]  # 自动获取grid_size_list或spline_order_list
    points_per_layer = len(x_values)

    # 4. 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_layers))
    markers = ['o', 's', '^', 'D', 'v', '>']  # 不同的标记样式

    # 5. 绘制收敛性曲线
    for layer_idx in range(num_layers):
        start = layer_idx * points_per_layer
        end = start + points_per_layer
        ax1.semilogy(x_values,
                     loaded_data['errors'][start:end],
                     marker=markers[layer_idx % len(markers)],
                     linestyle='-',
                     color=colors[layer_idx],
                     label=f'Layers: {loaded_data["layers_list"][layer_idx]}')

    ax1.set_xlabel(x_label, fontproperties=font_prop)
    ax1.set_ylabel('相对误差', fontproperties=font_prop)
    ax1.set_title('误差收敛曲线', fontproperties=font_prop)
    ax1.legend(prop=font_prop)
    ax1.grid(True, which="both", linestyle='--', alpha=0.5)
    ax1.set_xticks(x_values)

    # 6. 绘制参数缩放定律
    for layer_idx in range(num_layers):
        start = layer_idx * points_per_layer
        end = start + points_per_layer

        params = loaded_data['parameters'][start:end]
        errors = loaded_data['errors'][start:end]

        ax2.loglog(params, errors,
                   marker=markers[layer_idx % len(markers)],
                   linestyle='--',
                   color=colors[layer_idx],
                   label=f'Layers: {loaded_data["layers_list"][layer_idx]}')

    ax2.set_xlabel('参数量 (N)', fontproperties=font_prop)
    ax2.set_ylabel('相对误差', fontproperties=font_prop)
    ax2.set_title('参数缩放定律', fontproperties=font_prop)
    ax2.legend(prop=font_prop)
    ax2.grid(True, which="both", linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'./results/{plot_name}_combined.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# 7. 执行绘图
load_and_plot('./results/Crack_grid_size.npy', '网格大小', 'Grid_size_analysis')
load_and_plot('./results/Crack_spline_order.npy', '样条阶数', 'Spline_order_analysis')