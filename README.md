# hnu_course_work
课程作业：基于深度学习的3D打印工艺参数与力学性能预测模型研究（Research on 3D Printing Process Parameters and Mechanical Performance Prediction Model Based on Deep Learning）

首先将训练数据集与验证数据集存储到与代码同一文件夹下，代码按以下顺序运行：
1. am_data_preprocess 对数据集进行预处理与热力分析
2. am_model_training 训练SGD，SVR，DNN三个模型得到模型文件
3. strength_prediction 使用训练模型对验证数据进行强度预测

神经网络绘图代码：neutral_network_drawing, 原代码可参考：https://github.com/martisak/dotnets

预测数据值统计学分析：predicted_strength_data_process
