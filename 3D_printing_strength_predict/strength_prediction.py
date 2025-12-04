import numpy as np
import pandas as pd
import joblib
from tensorflow import keras


def load_models():
    """
    加载已训练的三个模型，并从SGD Pipeline中提取MinMaxScaler
    """
    try:
        sgd_model = joblib.load('sgd_model.pkl')
        svr_model = joblib.load('svr_model.pkl')
        nn_model = keras.models.load_model('neural_network_model.h5')
        feature_columns = joblib.load('feature_columns.pkl')

        # 从SGD Pipeline中提取MinMaxScaler，用于DNN预测
        minmax_scaler = sgd_model.named_steps['preprocess']

        print("模型和预处理器加载成功")
        return sgd_model, svr_model, nn_model, feature_columns, minmax_scaler
    except FileNotFoundError:
        print("错误：找不到模型文件，请先运行模型训练代码保存模型")
        return None, None, None, None, None


def preprocess_input_data(input_df, feature_columns):
    """
    预处理输入数据，创建虚拟变量并确保特征一致性。
    注意：此函数不负责缩放，缩放由单独的scaler完成。
    """
    # 重命名列以匹配训练数据的格式
    column_mapping = {
        'Layer_height_mm': 'layer_height',
        'wall_thickness': 'wall_thickness',
        'Infill_density_%': 'infill_density',
        'Infill_pattern': 'infill_pattern',
        'Bed_temperature_C': 'Bed_temperature',
        'Print_speed_mm_s': 'Print_speed',
        'Material': 'Material',
        'Fan_speed_m_s': 'Fan_speed',
        'Nozzle_diameter_mm': 'nozzle_diameter',
        'Build_volume_cm3': 'build_volume',
        'Filament_type': 'filament_type',
        'Filament_diameter_mm': 'filament_diameter',
        'Melting_temperature_C': 'melting_temperature',
        'Retraction_distance_mm': 'retraction_distance',
        'Retraction_speed_mm_s': 'retraction_speed',
        'Flow_rate_%': 'flow_rate',
        'Acceleration_mm_s2': 'acceleration',
        'Linear_advance': 'linear_advance',
        'Loading_rate_N_s': 'loading_rate',
        'Microstructure': 'microstructure',
        'Material_type': 'material_type',
        'Roughness': 'Roughness',
        'nozzle_temperature': 'nozzle_temperature'
    }

    # 创建映射后的数据框
    mapped_df = input_df.copy()
    for old_name, new_name in column_mapping.items():
        if old_name in input_df.columns:
            mapped_df = mapped_df.rename(columns={old_name: new_name})

    # 处理分类变量，创建虚拟变量
    categorical_columns = ['infill_pattern', 'Material', 'Filament_type', 'Microstructure', 'Material_type']

    for col in categorical_columns:
        if col in mapped_df.columns:
            unique_values = mapped_df[col].unique()
            for val in unique_values:
                mapped_df[f'{col}_{val}'] = (mapped_df[col] == val).astype(int)
            mapped_df = mapped_df.drop(columns=[col])

    # 确保所有特征列都存在
    for col in feature_columns:
        if col not in mapped_df.columns:
            mapped_df[col] = 0

    # 只保留训练时使用的特征列
    processed_df = mapped_df[feature_columns]

    return processed_df


def predict_tension_strength(input_file_path='FDM_Dataset.csv'):
    """
    预测3D打印件的张力强度
    """
    # 加载模型和预处理器
    sgd_model, svr_model, nn_model, feature_columns, minmax_scaler = load_models()
    if sgd_model is None:
        return

    # 读取输入数据
    try:
        input_data = pd.read_csv(input_file_path)
        print(f"成功读取输入数据 {input_file_path}，形状: {input_data.shape}")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file_path}")
        return
    except Exception as e:
        print(f"读取输入文件时出错: {e}")
        return

    # 预处理输入数据 (创建虚拟变量，对齐特征)
    processed_data = preprocess_input_data(input_data, feature_columns)
    print(f"预处理后数据形状: {processed_data.shape}")

    # --- 关键步骤：对数据进行缩放 ---
    # SGD和SVR的Pipeline会自动缩放，但我们显式地为DNN做一次
    processed_data_scaled = minmax_scaler.transform(processed_data)

    # 进行预测
    # SGD和SVR: Pipeline内部会自动缩放
    sgd_predictions = sgd_model.predict(processed_data)
    svr_predictions = svr_model.predict(processed_data)

    # DNN: 使用手动缩放后的数据
    nn_predictions = nn_model.predict(processed_data_scaled)
    nn_predictions = nn_predictions.flatten()

    # 创建结果数据框
    results_df = input_data.copy()
    results_df['Predicted_Tension_Strength_SGD'] = sgd_predictions
    results_df['Predicted_Tension_Strength_SVR'] = svr_predictions
    results_df['Predicted_Tension_Strength_NN'] = nn_predictions
    results_df['Average_Prediction'] = (sgd_predictions + svr_predictions + nn_predictions) / 3

    # 保存预测结果
    output_file = input_file_path.replace('.csv', '_predictions.csv')
    results_df.to_csv(output_file, index=False)
    print(f"预测结果已保存到 {output_file}")

    # 显示前几行预测结果
    print("\n预测结果预览:")
    prediction_cols = ['Predicted_Tension_Strength_SGD', 'Predicted_Tension_Strength_SVR',
                       'Predicted_Tension_Strength_NN', 'Average_Prediction']
    display_cols = list(input_data.columns[:5]) + prediction_cols
    print(results_df[display_cols].head())

    # 模型性能统计
    print(f"\n预测统计:")
    print(f"SGD预测范围: {sgd_predictions.min():.2f} - {sgd_predictions.max():.2f}")
    print(f"SVR预测范围: {svr_predictions.min():.2f} - {svr_predictions.max():.2f}")
    print(f"NN预测范围: {nn_predictions.min():.2f} - {nn_predictions.max():.2f}")

    # 计算模型间差异
    diff_sgd_svr = np.abs(sgd_predictions - svr_predictions)
    diff_sgd_nn = np.abs(sgd_predictions - nn_predictions)
    diff_svr_nn = np.abs(svr_predictions - nn_predictions)

    print(f"\n模型间预测差异统计:")
    print(f"SGD与SVR平均差异: {diff_sgd_svr.mean():.2f}")
    print(f"SGD与NN平均差异: {diff_sgd_nn.mean():.2f}")
    print(f"SVR与NN平均差异: {diff_svr_nn.mean():.2f}")

    return results_df


# 主程序
if __name__ == "__main__":
    print("=" * 50)
    print("3D打印张力强度预测工具 (修正版)")
    print("=" * 50)
    print("开始预测 FDM_Dataset.csv ...")
    results = predict_tension_strength('FDM_Dataset.csv')

    if results is not None:
        print("\n预测完成！")
    else:
        print("\n预测失败，请检查模型文件和输入数据")

    print("\n" + "=" * 50)