import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('FDM_Dataset_predictions.csv')

# 计算每个模型的评估指标
models = ['Predicted_Tension_Strength_SGD', 'Predicted_Tension_Strength_SVR', 'Predicted_Tension_Strength_NN']
actual = df['actual value']

metrics = {}
for model in models:
    pred = df[model]
    r2 = r2_score(actual, pred)
    mae = mean_absolute_error(actual, pred)
    metrics[model] = {'R2': r2, 'MAE': mae}

# 1. 预测值 vs 实际值散点图 - SGD
fig, ax = plt.subplots(figsize=(6, 5))
pred_vals = df['Predicted_Tension_Strength_SGD']
ax.scatter(actual, pred_vals, alpha=0.7, s=50, label='SGD', marker='o', color='black')

# 添加完美预测线
min_val = min(actual.min(), pred_vals.min())
max_val = max(actual.max(), pred_vals.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='预测')

# 添加拟合线
z = np.polyfit(actual, pred_vals, 1)
p = np.poly1d(z)
ax.plot(actual, p(actual), color='black', linestyle='-', alpha=0.8, linewidth=1.5, label=f'拟合线 (R2={metrics["Predicted_Tension_Strength_SGD"]["R2"]:.3f})')

ax.set_xlabel('实际值 (MPa)', fontsize=12)
ax.set_ylabel('SGD 预测值 (MPa)', fontsize=12)
ax.set_title('SGD 模型: R2 = {r2:.3f}, MAE = {mae:.3f}'.format(r2=metrics["Predicted_Tension_Strength_SGD"]["R2"], mae=metrics["Predicted_Tension_Strength_SGD"]["MAE"]), fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('SGD_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 预测值 vs 实际值散点图 - SVR
fig, ax = plt.subplots(figsize=(6, 5))
pred_vals = df['Predicted_Tension_Strength_SVR']
ax.scatter(actual, pred_vals, alpha=0.7, s=50, label='SVR', marker='s', color='black')

# 添加完美预测线
min_val = min(actual.min(), pred_vals.min())
max_val = max(actual.max(), pred_vals.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='预测')

# 添加拟合线
z = np.polyfit(actual, pred_vals, 1)
p = np.poly1d(z)
ax.plot(actual, p(actual), color='black', linestyle='-', alpha=0.8, linewidth=1.5, label=f'拟合线 (R2={metrics["Predicted_Tension_Strength_SVR"]["R2"]:.3f})')

ax.set_xlabel('实际值 (MPa)', fontsize=12)
ax.set_ylabel('SVR 预测值 (MPa)', fontsize=12)
ax.set_title('SVR 模型: R2 = {r2:.3f}, MAE = {mae:.3f}'.format(r2=metrics["Predicted_Tension_Strength_SVR"]["R2"], mae=metrics["Predicted_Tension_Strength_SVR"]["MAE"]), fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('SVR_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 预测值 vs 实际值散点图 - NN
fig, ax = plt.subplots(figsize=(6, 5))
pred_vals = df['Predicted_Tension_Strength_NN']
ax.scatter(actual, pred_vals, alpha=0.7, s=50, label='NN', marker='^', color='black')

# 添加完美预测线
min_val = min(actual.min(), pred_vals.min())
max_val = max(actual.max(), pred_vals.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='预测')

# 添加拟合线
z = np.polyfit(actual, pred_vals, 1)
p = np.poly1d(z)
ax.plot(actual, p(actual), color='black', linestyle='-', alpha=0.8, linewidth=1.5, label=f'拟合线 (R2={metrics["Predicted_Tension_Strength_NN"]["R2"]:.3f})')

ax.set_xlabel('实际值 (MPa)', fontsize=12)
ax.set_ylabel('NN 预测值 (MPa)', fontsize=12)
ax.set_title('NN 模型: R2 = {r2:.3f}, MAE = {mae:.3f}'.format(r2=metrics["Predicted_Tension_Strength_NN"]["R2"], mae=metrics["Predicted_Tension_Strength_NN"]["MAE"]), fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('NN_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 模型性能对比柱状图
fig, ax = plt.subplots(figsize=(8, 6))
x_pos = np.arange(3)
model_names = ['SGD', 'SVR', 'NN']
r2_scores = [metrics[model]['R2'] for model in models]
mae_scores = [metrics[model]['MAE'] for model in models]

# 创建双y轴
ax_twin = ax.twinx()

bars1 = ax.bar(x_pos - 0.2, r2_scores, 0.4, label='R2', color='white', edgecolor='black', alpha=0.8)
bars2 = ax_twin.bar(x_pos + 0.2, mae_scores, 0.4, label='MAE', color='white', edgecolor='black', alpha=0.8)

ax.set_xlabel('模型', fontsize=12)
ax.set_ylabel('R2 分数', fontsize=12)
ax_twin.set_ylabel('MAE (MPa)', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 1)
ax_twin.set_ylim(0, max(mae_scores) * 1.2)

# 添加数值标签
for i, (bar, r2_val, mae_val) in enumerate(zip(bars1, r2_scores, mae_scores)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{r2_val:.3f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax_twin.text(x_pos[i] + 0.2, mae_val + 0.05, f'{mae_val:.3f}',
                  ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_title('模型性能指标对比', fontsize=13)
ax.legend(loc='upper left', fontsize=10)
ax_twin.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 残差分析图 - SGD
fig, ax = plt.subplots(figsize=(6, 5))
residuals = df['Predicted_Tension_Strength_SGD'] - actual
ax.scatter(df['Predicted_Tension_Strength_SGD'], residuals, alpha=0.7, s=50, color='black', marker='o')
ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('SGD 预测值 (MPa)', fontsize=12)
ax.set_ylabel('残差 (预测值 - 实际值)', fontsize=12)
ax.set_title('SGD 残差图', fontsize=13)
ax.grid(True, alpha=0.3)

# 添加残差统计信息
rmse = np.sqrt(np.mean(residuals**2))
ax.text(0.05, 0.95, f'RMSE: {rmse:.3f}\nMAE: {np.mean(np.abs(residuals)):.3f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

plt.tight_layout()
plt.savefig('SGD_residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 残差分析图 - SVR
fig, ax = plt.subplots(figsize=(6, 5))
residuals = df['Predicted_Tension_Strength_SVR'] - actual
ax.scatter(df['Predicted_Tension_Strength_SVR'], residuals, alpha=0.7, s=50, color='black', marker='s')
ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('SVR 预测值 (MPa)', fontsize=12)
ax.set_ylabel('残差 (预测值 - 实际值)', fontsize=12)
ax.set_title('SVR 残差图', fontsize=13)
ax.grid(True, alpha=0.3)

# 添加残差统计信息
rmse = np.sqrt(np.mean(residuals**2))
ax.text(0.05, 0.95, f'RMSE: {rmse:.3f}\nMAE: {np.mean(np.abs(residuals)):.3f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

plt.tight_layout()
plt.savefig('SVR_residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 残差分析图 - NN
fig, ax = plt.subplots(figsize=(6, 5))
residuals = df['Predicted_Tension_Strength_NN'] - actual
ax.scatter(df['Predicted_Tension_Strength_NN'], residuals, alpha=0.7, s=50, color='black', marker='^')
ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('NN 预测值 (MPa)', fontsize=12)
ax.set_ylabel('残差 (预测值 - 实际值)', fontsize=12)
ax.set_title('NN 残差图', fontsize=13)
ax.grid(True, alpha=0.3)

# 添加残差统计信息
rmse = np.sqrt(np.mean(residuals**2))
ax.text(0.05, 0.95, f'RMSE: {rmse:.3f}\nMAE: {np.mean(np.abs(residuals)):.3f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

plt.tight_layout()
plt.savefig('NN_residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Bland-Altman 图 - SGD
fig, ax = plt.subplots(figsize=(6, 5))
avg_pred_actual = (df['Predicted_Tension_Strength_SGD'] + actual) / 2
diff = df['Predicted_Tension_Strength_SGD'] - actual
mean_diff = np.mean(diff)
std_diff = np.std(diff)

ax.scatter(avg_pred_actual, diff, alpha=0.7, s=50, color='black', marker='o')
ax.axhline(mean_diff, color='black', linestyle='-', linewidth=2, label=f'平均偏差: {mean_diff:.3f}')
ax.axhline(mean_diff + 1.96*std_diff, color='gray', linestyle='--', linewidth=1, label='+1.96 SD')
ax.axhline(mean_diff - 1.96*std_diff, color='gray', linestyle='--', linewidth=1, label='-1.96 SD')

ax.set_xlabel('预测值与实际值的平均值 (MPa)', fontsize=12)
ax.set_ylabel('差异 (预测值 - 实际值)', fontsize=12)
ax.set_title('SGD Bland-Altman 图', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SGD_bland_altman.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Bland-Altman 图 - SVR
fig, ax = plt.subplots(figsize=(6, 5))
avg_pred_actual = (df['Predicted_Tension_Strength_SVR'] + actual) / 2
diff = df['Predicted_Tension_Strength_SVR'] - actual
mean_diff = np.mean(diff)
std_diff = np.std(diff)

ax.scatter(avg_pred_actual, diff, alpha=0.7, s=50, color='black', marker='s')
ax.axhline(mean_diff, color='black', linestyle='-', linewidth=2, label=f'平均偏差: {mean_diff:.3f}')
ax.axhline(mean_diff + 1.96*std_diff, color='gray', linestyle='--', linewidth=1, label='+1.96 SD')
ax.axhline(mean_diff - 1.96*std_diff, color='gray', linestyle='--', linewidth=1, label='-1.96 SD')

ax.set_xlabel('预测值与实际值的平均值 (MPa)', fontsize=12)
ax.set_ylabel('差异 (预测值 - 实际值)', fontsize=12)
ax.set_title('SVR Bland-Altman 图', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SVR_bland_altman.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. Bland-Altman 图 - NN
fig, ax = plt.subplots(figsize=(6, 5))
avg_pred_actual = (df['Predicted_Tension_Strength_NN'] + actual) / 2
diff = df['Predicted_Tension_Strength_NN'] - actual
mean_diff = np.mean(diff)
std_diff = np.std(diff)

ax.scatter(avg_pred_actual, diff, alpha=0.7, s=50, color='black', marker='^')
ax.axhline(mean_diff, color='black', linestyle='-', linewidth=2, label=f'平均偏差: {mean_diff:.3f}')
ax.axhline(mean_diff + 1.96*std_diff, color='gray', linestyle='--', linewidth=1, label='+1.96 SD')
ax.axhline(mean_diff - 1.96*std_diff, color='gray', linestyle='--', linewidth=1, label='-1.96 SD')

ax.set_xlabel('预测值与实际值的平均值 (MPa)', fontsize=12)
ax.set_ylabel('差异 (预测值 - 实际值)', fontsize=12)
ax.set_title('NN Bland-Altman 图', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('NN_bland_altman.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. 打印详细统计信息
print("="*60)
print("模型性能详细统计")
print("="*60)
for model, name in zip(models, ['SGD', 'SVR', 'NN']):
    r2 = metrics[model]['R2']
    mae = metrics[model]['MAE']
    rmse = np.sqrt(np.mean((df[model] - actual)**2))
    print(f"{name:4s} - R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

print("\n" + "="*60)
print("单张图表已保存:")
print("- SGD_prediction_vs_actual.png")
print("- SVR_prediction_vs_actual.png")
print("- NN_prediction_vs_actual.png")
print("- model_performance_comparison.png")
print("- SGD_residual_analysis.png")
print("- SVR_residual_analysis.png")
print("- NN_residual_analysis.png")
print("- SGD_bland_altman.png")
print("- SVR_bland_altman.png")
print("- NN_bland_altman.png")
print("="*60)



