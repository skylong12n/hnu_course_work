import pandas as pd  # 数据处理，CSV文件输入输出

# 读取数据文件
df = pd.read_csv('3D_printer.csv')
print("数据形状:", df.shape)
print("数据列名:", df.columns.tolist())
print("前5行数据:")
print(df.head())

# 数据集统计描述
print("\n数据统计描述:")
print(df.describe())

# 数据集信息
print("\n数据信息:")
df.info()

# 数据类型
print("\n数据类型:")
print(df.dtypes)

# 检查缺失值
print("\n缺失值:")
print(df.isnull().sum())

# 检查重复值
print("\n重复值:")
print(df.duplicated().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 10  # 设置字体大小

# 绘制拉伸性能、层高、壁厚、填充密度的趋势图
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('参数趋势图', fontsize=14)

# 检查并绘制各列数据
columns_to_plot = ['Elongation', 'layer_height', 'wall_thickness', 'infill_density']
for i, col in enumerate(columns_to_plot):
    if col in df.columns:
        row, col_idx = i // 2, i % 2
        axes[row, col_idx].plot(df.index, df[col], linestyle='-', color='black')
        axes[row, col_idx].set_title(col, fontsize=12)
        axes[row, col_idx].grid(True, linestyle='--', alpha=0.6)
    else:
        print(f"警告: 数据中没有列 '{col}'")

plt.tight_layout()
plt.show()

# 绘制打印温度、床温、打印速度、风扇速度的趋势图
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('打印参数趋势图', fontsize=14)

temp_columns = ['nozzle_temperature', 'Bed_temperature', 'Print_speed', 'Fan_speed']
for i, col in enumerate(temp_columns):
    if col in df.columns:
        row, col_idx = i // 2, i % 2
        axes[row, col_idx].plot(df.index, df[col], linestyle='-', color='black')
        axes[row, col_idx].set_title(col, fontsize=12)
        axes[row, col_idx].grid(True, linestyle='--', alpha=0.6)
    else:
        print(f"警告: 数据中没有列 '{col}'")

plt.tight_layout()
plt.show()

# 绘制粗糙度和拉伸强度的趋势图
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('表面质量和强度趋势图', fontsize=14)

roughness_col = 'Roughness' if 'Roughness' in df.columns else None
tension_col = 'Tension_strenght' if 'Tension_strenght' in df.columns else 'tension_strength' if 'tension_strength' in df.columns else None

if roughness_col and roughness_col in df.columns:
    axes[0].plot(df.index, df[roughness_col], linestyle='-', color='black')
    axes[0].set_title(roughness_col, fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
else:
    print(f"警告: 数据中没有粗糙度列")

if tension_col and tension_col in df.columns:
    axes[1].plot(df.index, df[tension_col], linestyle='-', color='black')
    axes[1].set_title(tension_col, fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)
else:
    print(f"警告: 数据中没有拉伸强度列")

plt.tight_layout()
plt.show()

# 绘制材料类型饼图
material_col = 'Material' if 'Material' in df.columns else 'material' if 'material' in df.columns else None
if material_col and material_col in df.columns:
    plt.figure(figsize=(8, 8))
    counts = df[material_col].value_counts()
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90, 
            colors=plt.cm.gray_r(range(len(counts))))
    plt.title('材料类型分布', fontsize=14)
    plt.show()

# 绘制数据分布直方图
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
if numeric_cols:
    n_cols = min(4, len(numeric_cols))  # 最多4列
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols  # 计算需要的行数
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle('数据分布直方图', fontsize=14)
    
    if len(numeric_cols) == 1:
        axes = [axes]
    elif n_rows == 1 and len(numeric_cols) > 1:
        axes = axes if len(numeric_cols) > 1 else [axes]
    elif len(numeric_cols) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=20, color='lightgray', edgecolor='black')
        axes[i].set_title(col, fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.6)
    
    # 隐藏多余的子图
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# 绘制相关性热力图
numeric_df = df.select_dtypes(include=['number'])
if not numeric_df.empty:
    plt.figure(figsize=(10, 8))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='gray', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('相关性热力图', fontsize=14)
    plt.tight_layout()
    plt.show()

# 定义目标变量Y
target_col = None
for col in ['Material', 'material', 'tension_strength', 'Tension_strenght']:
    if col in df.columns:
        target_col = col
        break

if target_col:
    Y = df[target_col]
    print(f"\n目标变量: {target_col}")
    print(Y.head())
else:
    print("\n没有找到合适的目标变量列")
    # 如果没有找到合适的分类列，使用tension_strength作为回归目标
    for col in ['tension_strength', 'Tension_strenght']:
        if col in df.columns:
            target_col = col
            Y = df[col]
            print(f"使用 {target_col} 作为目标变量")
            break

# 定义特征变量X（除目标变量外的所有数值列）
if target_col:
    feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
    X = df[feature_cols]
    print(f"\n特征变量列: {feature_cols}")
    print("特征变量前5行:")
    print(X.head())
else:
    print("无法定义特征变量，因为没有找到目标变量")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# 对分类特征进行标签编码
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != target_col:  # 不对目标变量进行编码
        X = X.join(df[[col]])  # 将分类列加入特征矩阵
        X[col] = le.fit_transform(df[col])

print("\n标签编码后的特征:")
print(X.head())

from sklearn.model_selection import train_test_split
if target_col and len(feature_cols) > 0:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print(f"\n训练集形状: X_train={X_train.shape}, Y_train={Y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, Y_test={Y_test.shape}")

# 支持向量机(SVM)
from sklearn.svm import SVC

if target_col and len(feature_cols) > 0:
    svm = SVC()
    svm.fit(X_train, Y_train)

    y_pred = svm.predict(X_test)

    from sklearn.metrics import accuracy_score

    acc = accuracy_score(Y_test, y_pred)

    print(f"\nSVM准确率 : {acc}")

    # 使用整个数据集训练SVM
    svm1 = SVC()
    svm1.fit(X, Y)
    y_pred1 = svm1.predict(X)

    print(f"SVM准确率得分 : {accuracy_score(y_pred1, Y)}")

# 逻辑回归
from sklearn.linear_model import LogisticRegression

if target_col and len(feature_cols) > 0:
    # 检查目标变量是否为分类变量
    if Y.dtype == 'object' or len(Y.unique()) < 20:  # 假设唯一值少于20的是分类变量
        lr = LogisticRegression(max_iter=1000)  # 增加迭代次数以提高准确性

        lr.fit(X_train, Y_train)
        y_pred = lr.predict(X_test)
        print(f"\n逻辑回归准确率 : {accuracy_score(Y_test, y_pred)}")

        lr.fit(X, Y)
        y_pred = lr.predict(X)
        print(f"逻辑回归准确率 : {accuracy_score(Y, y_pred)}")
    else:
        print("\n目标变量为连续变量，跳过逻辑回归分类任务")

# 决策树
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

if target_col and len(feature_cols) > 0:
    # 根据目标变量类型选择分类或回归
    if Y.dtype == 'object' or len(Y.unique()) < 20:  # 分类任务
        dt = DecisionTreeClassifier(random_state=42)  # 设置随机种子以确保结果可重现
    else:  # 回归任务
        dt = DecisionTreeRegressor(random_state=42)
        
    dt.fit(X_train, Y_train)

    predictions = dt.predict(X_test)
    if Y.dtype == 'object' or len(Y.unique()) < 20:  # 分类任务
        accuracy = accuracy_score(Y_test, predictions)
        print(f"\n决策树准确率 : {accuracy}")
    else:  # 回归任务
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(Y_test, predictions)
        r2 = r2_score(Y_test, predictions)
        print(f"\n决策树均方误差 : {mse}, R²得分 : {r2}")

    dt.fit(X, Y)
    predictions = dt.predict(X)
    if Y.dtype == 'object' or len(Y.unique()) < 20:  # 分类任务
        accuracy = accuracy_score(Y, predictions)
        print(f"决策树准确率 : {accuracy}")
    else:  # 回归任务
        mse = mean_squared_error(Y, predictions)
        r2 = r2_score(Y, predictions)
        print(f"决策树均方误差 : {mse}, R²得分 : {r2}")

# 网格搜索
# SVM参数优化
from sklearn.model_selection import GridSearchCV

if target_col and len(feature_cols) > 0 and (Y.dtype == 'object' or len(Y.unique()) < 20):
    grid = {
        'C': [0.01, 0.1, 1, 10],
        'kernel': ["linear", "rbf"],  # 简化核函数选项以加快搜索
        'gamma': [0.01, 0.1, 1]
    }

    svm = SVC()
    svm_cv = GridSearchCV(svm, grid, cv=5, n_jobs=-1)
    svm_cv.fit(X_train, Y_train)

    print(f"\nSVM最佳参数: {svm_cv.best_params_}")
    print(f"SVM训练得分: {svm_cv.best_score_}")
    print(f"SVM测试得分: {svm_cv.score(X_test, Y_test)}")

# 决策树参数优化
if target_col and len(feature_cols) > 0:
    param_grid = {'max_features': ['sqrt', 'log2'],
                  'ccp_alpha': [0.0, 0.01, 0.1],
                  'max_depth': [3, 5, 7, 10, 15],  # 限制深度以避免过拟合
                  'min_samples_split': [2, 5, 10]
                 }
    
    if Y.dtype == 'object' or len(Y.unique()) < 20:  # 分类任务
        tree_clas = DecisionTreeClassifier(random_state=1024)
    else:  # 回归任务
        tree_clas = DecisionTreeRegressor(random_state=1024)
        
    grid_search = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, Y_train)

    print(f"\n决策树最佳参数 : {grid_search.best_params_}")
    print(f"决策树训练得分 : {grid_search.best_score_}")
    
    if Y.dtype == 'object' or len(Y.unique()) < 20:  # 分类任务
        print(f"决策树测试得分 : {grid_search.score(X_test, Y_test)}")
    else:  # 回归任务
        test_score = grid_search.score(X_test, Y_test)
        print(f"决策树测试得分 (R²) : {test_score}")



