# 3D打印数据集
# 数据清洗

import numpy as np
import pandas as pd

# 获取数据（从本地文件）
manufacture = pd.read_csv('manufacturing.txt')

# 如何访问数据框中的内容
# manufacture['layer_height']
# manufacture.loc[1]

# 创建填充模式的虚拟变量
manufacture['infill_pattern']
dummy = pd.get_dummies(manufacture[['infill_pattern', 'material']])
dummy.head()
input_drop = manufacture.drop(['infill_pattern', 'material'], axis=1)

# 创建X和y矩阵
X_man = pd.concat([input_drop, dummy], axis=1)
print(X_man.shape)

# 从x矩阵创建y矩阵
y_man = X_man["tension_strength"]
print(y_man.shape)
X_man = X_man.drop(["tension_strength"], axis=1)
print(X_man.shape)

# 参数化模型
# K折交叉验证

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate

pipe_man = Pipeline([('preprocess', MinMaxScaler()),
                     ('reg', SGDRegressor(random_state=42, max_iter=1000))])  # 增加max_iter解决收敛警告

pipe_man.fit(X_man, y_man)
print(pipe_man['reg'].intercept_, pipe_man['reg'].coef_)

explained = cross_validate(pipe_man, X_man, y_man, cv=5,
               scoring="explained_variance", return_train_score=False,
              return_estimator=False)

meansquare = cross_validate(pipe_man, X_man, y_man, cv=5,
               scoring="neg_mean_squared_error", return_train_score=False,
              return_estimator=False)

# 支持向量回归(SVR)

from sklearn.svm import SVR
pipe_man_non = Pipeline([('preprocess', MinMaxScaler()),
                     ('svr', SVR(kernel="rbf",
                                 C=100, gamma=0.1, epsilon=0.1))])
# print(pipe_man_non['svr'].intercept_, pipe_man_non['svr'].coef_)
explained_non = cross_validate(pipe_man_non, X_man, y_man, cv=5,
               scoring="explained_variance", return_train_score=False,
              return_estimator=False)
meansquare_non = cross_validate(pipe_man_non, X_man, y_man, cv=5,
               scoring="neg_mean_squared_error", return_train_score=False,
              return_estimator=False)

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

x = [1, 2, 3, 4, 5]
y1 = explained_non["test_score"]

y2 = explained["test_score"]

plt.plot(x, y1, linestyle='--', marker='o', color='black', label='SVR')
plt.plot(x, y2, linestyle='-', marker='s', color='black', label='回归模型')
plt.xticks(np.arange(1, 6, step=1))
plt.xlabel("交叉验证折数")
plt.ylabel("解释方差")
plt.title("参数化和非参数化模型的解释方差比较")
plt.legend()
plt.grid(True)
plt.show()

# 神经网络

import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# 将pandas数据转换为numpy数组以避免数据类型错误
X_man_np = X_man.values
y_man_np = y_man.values

neural_man = keras.models.Sequential([
    keras.layers.Dense(100, name='hidden1', activation="relu", input_shape=(X_man_np.shape[1],)),
    keras.layers.Dense(100, name='hidden2', activation="relu"),
    keras.layers.Dense(1, name='output')
])

neural_man.compile(loss=keras.losses.MeanSquaredError(name="mean_squared_error"),
              optimizer='rmsprop')

neural_man.build()
neural_man.summary()

from sklearn import ensemble
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection

X_man_train, X_man_test, y_man_train, y_man_test = train_test_split(X_man_np,
                                                                    y_man_np,
                                                                    random_state=30)

# 增加epoch到100，但使用早停机制防止过拟合
epoch = 100

# 设置早停机制
early_stopping = EarlyStopping(
    monitor='val_loss',    # 监控验证损失
    patience=10,           # 如果连续10个epoch验证损失没有改善，则停止训练
    restore_best_weights=True  # 恢复最佳权重
)

history = neural_man.fit(X_man_train, y_man_train, batch_size=256, epochs=epoch,
                         verbose=2, validation_data=(X_man_test, y_man_test),
                         callbacks=[early_stopping])

import statistics

def explained_variance(X, y, model):
    y_pred = model.predict(X, batch_size=10000)
    y_pred = y_pred.reshape(-1)
    vartop = y - y_pred
    vartop = statistics.variance(vartop)
    varbottom = statistics.variance(y)
    explained_variance = 1- (vartop/varbottom)
    return(explained_variance)

test_variance = explained_variance(X_man_test, y_man_test, neural_man)

train_variance = explained_variance(X_man_train, y_man_train, neural_man)

print("测试方差: ", test_variance)
print("训练方差: ", train_variance)

# print(history.history)
history = pd.DataFrame(history.history)

import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 使用实际的训练轮次（可能由于早停机制而少于100）
actual_epochs = len(history['loss'])
x_plot = range(0, actual_epochs, 1)

plt.plot(x_plot, history['loss'], linestyle='-', marker='s', color='black', label="训练损失")
plt.plot(x_plot, history['val_loss'], linestyle='--', marker='o', color='black', label="验证损失")
plt.grid(True)
plt.xlabel("训练轮次")
plt.ylabel("损失值")
plt.title("神经网络训练过程中的损失变化")
plt.legend()
plt.show()

print(f"实际训练轮次: {actual_epochs}")

#SVR, SGD 解释方差
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score

# 使用之前分割好的数据集
X_train = X_man_train # 重命名变量以匹配您的逻辑
X_test = X_man_test
y_train = y_man_train
y_test = y_man_test

# 初始化模型
sgd = SGDRegressor(random_state=42, max_iter=1000) # 保持与Pipeline中一致的参数
svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1) # 保持与Pipeline中一致的参数

# 为了使SGD和SVR能正确处理数据，需要先进行相同的预处理（如缩放）
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练模型
sgd.fit(X_train_scaled, y_train)
svr.fit(X_train_scaled, y_train)

# 预测
y_pred_sgd = sgd.predict(X_test_scaled)
y_pred_svr = svr.predict(X_test_scaled)

# 计算解释方差得分
explained_variance_sgd = explained_variance_score(y_test, y_pred_sgd)
explained_variance_svr = explained_variance_score(y_test, y_pred_svr)

print(f"SGD模型的解释方差得分: {explained_variance_sgd}")
print(f"SVR模型的解释方差得分: {explained_variance_svr}")

# 如果想要通过交叉验证来评估模型性能，可以使用cross_val_score函数
# 注意：对于交叉验证，需要对整个数据集进行预处理，这里我们直接使用之前的Pipeline
cv_scores_sgd_pipeline = cross_validate(pipe_man, X_man, y_man, cv=5, scoring='explained_variance', return_train_score=False, return_estimator=False)
cv_scores_svr_pipeline = cross_validate(pipe_man_non, X_man, y_man, cv=5, scoring='explained_variance', return_train_score=False, return_estimator=False)

print(f"SGD模型的交叉验证解释方差得分 (使用Pipeline): {cv_scores_sgd_pipeline['test_score'].mean()} (+/- {cv_scores_sgd_pipeline['test_score'].std() * 2})")
print(f"SVR模型的交叉验证解释方差得分 (使用Pipeline): {cv_scores_svr_pipeline['test_score'].mean()} (+/- {cv_scores_svr_pipeline['test_score'].std() * 2})")

# 对于手动预处理后的模型，也可以进行交叉验证，但需要更复杂的处理（例如使用交叉验证迭代器手动缩放数据）
# 这里我们只展示使用Pipeline的结果，因为它更标准且易于实现