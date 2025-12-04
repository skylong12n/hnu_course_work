import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("Warning: SimHei font not found. Trying Microsoft YaHei...")
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("Warning: Microsoft YaHei font not found. Please manually configure a suitable Chinese font.")

# 创建绘图
fig, ax = plt.subplots(figsize=(14, 9)) # 增大画布尺寸
ax.set_aspect('equal')
ax.set_facecolor('white')

# 定义层的位置
input_layer_x = 0
hidden1_layer_x = 2.5 # 增大层间距
hidden2_layer_x = 5.0 # 增大层间距
output_layer_x = 7.5 # 增大层间距

# 定义神经元垂直间隔
neuron_spacing = 0.8 # 增大神经元垂直间距
gap_for_ellipsis = 1.2 # 省略号处的额外间距

# 输入层 (13维特征)
input_neurons_y = [2 + i * neuron_spacing for i in range(3)] # 前3个代表
input_neurons_y.append(input_neurons_y[-1] + gap_for_ellipsis) # 虚线分隔
input_neurons_y.append(input_neurons_y[-1] + neuron_spacing) # 最后1个代表 (相当于第13个)

# 存储神经元中心坐标，用于连接
input_neuron_coords = []
for i, y in enumerate(input_neurons_y):
    if i == 3: # 虚线部分
        ax.text(input_layer_x, y - neuron_spacing/2 + 0.1, "...", ha='center', va='center', fontsize=10, color='black') # 字体大小改善
        continue
    circle = patches.Circle((input_layer_x, y), 0.3, edgecolor='black', facecolor='white', lw=1.5)
    ax.add_patch(circle)
    # 调整F标签的字体大小
    ax.text(input_layer_x - 0.6, y, f"F{i+1}" if i < 3 else f"F{13-(len(input_neurons_y)-1-(i))}", ha='center', va='center', fontsize=9, color='black') # 减小字体
    input_neuron_coords.append((input_layer_x, y))
ax.text(input_layer_x, input_neurons_y[-1] + neuron_spacing * 1.5, "输入层\n(13维特征)", ha='center', va='center', fontsize=10, color='black') # 减小字体

# 隐藏层1 (100个神经元, ReLU)
hidden1_neurons_y = [1.5 + i * neuron_spacing for i in range(3)]
hidden1_neurons_y.append(hidden1_neurons_y[-1] + gap_for_ellipsis)
hidden1_neurons_y.append(hidden1_neurons_y[-1] + neuron_spacing)

hidden1_neuron_coords = []
for i, y in enumerate(hidden1_neurons_y):
    if i == 3:
        ax.text(hidden1_layer_x, y - neuron_spacing/2 + 0.1, "...", ha='center', va='center', fontsize=10, color='black') # 字体大小改善
        continue
    circle = patches.Circle((hidden1_layer_x, y), 0.3, edgecolor='black', facecolor='white', lw=1.5)
    ax.add_patch(circle)
    hidden1_neuron_coords.append((hidden1_layer_x, y))
ax.text(hidden1_layer_x, hidden1_neurons_y[-1] + neuron_spacing * 1.5, "隐藏层1\n(100个神经元, ReLU)", ha='center', va='center', fontsize=10, color='black') # 减小字体

# 隐藏层2 (100个神经元, ReLU)
hidden2_neurons_y = [1.5 + i * neuron_spacing for i in range(3)]
hidden2_neurons_y.append(hidden2_neurons_y[-1] + gap_for_ellipsis)
hidden2_neurons_y.append(hidden2_neurons_y[-1] + neuron_spacing)

hidden2_neuron_coords = []
for i, y in enumerate(hidden2_neurons_y):
    if i == 3:
        ax.text(hidden2_layer_x, y - neuron_spacing/2 + 0.1, "...", ha='center', va='center', fontsize=10, color='black') # 减小字体
        continue
    circle = patches.Circle((hidden2_layer_x, y), 0.3, edgecolor='black', facecolor='white', lw=1.5)
    ax.add_patch(circle)
    hidden2_neuron_coords.append((hidden2_layer_x, y))
ax.text(hidden2_layer_x, hidden2_neurons_y[-1] + neuron_spacing * 1.5, "隐藏层2\n(100个神经元, ReLU)", ha='center', va='center', fontsize=10, color='black') # 减小字体

# 输出层 (1个神经元)
output_neuron_y = (max(input_neurons_y) + min(input_neurons_y)) / 2  # 居中
circle = patches.Circle((output_layer_x, output_neuron_y), 0.3, edgecolor='black', facecolor='white', lw=1.5)
ax.add_patch(circle)
# 调整预测值的 x 坐标使其更靠左
ax.text(output_layer_x - 0.9, output_neuron_y, "预测值", ha='center', va='center', fontsize=9, color='black') # 减小字体，并向左移动
output_neuron_coords = [(output_layer_x, output_neuron_y)]
ax.text(output_layer_x, output_neuron_y + neuron_spacing * 1.5, "输出层\n(1个神经元)", ha='center', va='center', fontsize=10, color='black') # 减小字体

# 连接线 (连接所有可见的代表性神经元)
line_width = 0.5 # 减小线宽

# 输入层 -> 隐藏层1
for x1, y1 in input_neuron_coords:
    for x2, y2 in hidden1_neuron_coords:
        ax.plot([x1 + 0.3, x2 - 0.3], [y1, y2], 'k-', lw=line_width, zorder=0)

# 隐藏层1 -> 隐藏层2
for x1, y1 in hidden1_neuron_coords:
    for x2, y2 in hidden2_neuron_coords:
        ax.plot([x1 + 0.3, x2 - 0.3], [y1, y2], 'k-', lw=line_width, zorder=0)

# 隐藏层2 -> 输出层
for x1, y1 in hidden2_neuron_coords:
    x2, y2 = output_neuron_coords[0]
    ax.plot([x1 + 0.3, x2 - 0.3], [y1, y2], 'k-', lw=line_width, zorder=0)

# 文本描述
ax.text((input_layer_x + output_layer_x) / 2, -1.5,
        "优化器: RMSprop\n损失函数: 均方误差 (MSE)",
        ha='center', va='center', fontsize=11, color='black', # 减小字体
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
ax.text((input_layer_x + output_layer_x) / 2, -2.5,
        "全连接深度神经网络 (DNN) 示意图",
        ha='center', va='center', fontsize=13, fontweight='bold', color='black') # 减小字体

ax.set_xlim(-1, output_layer_x + 1)
ax.set_ylim(-3, max(input_neurons_y) + 2)
ax.axis('off') # 隐藏坐标轴

plt.show()