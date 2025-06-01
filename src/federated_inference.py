"""
联邦学习推理模块
支持使用联邦训练的模型进行肺结节检测
"""

import os

def run_inference(image_path, use_federated=True):
    """运行推理"""
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在 {image_path}")
        return False

    print(f"正在对图像进行推理: {image_path}")
    try:
        if use_federated:
            from federated_inference_utils import (
                predict_with_federated_model,
                visualize_federated_results,
            )

            nodules, prob_map, image, spacing, origin = predict_with_federated_model(
                image_path
            )
            visualize_federated_results(image, prob_map, nodules)
        else:
            from show_nodules import show_predicted_nodules

            show_predicted_nodules(image_path, confidence_threshold=0.3)
        return True
    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        return False


if __name__ == "__main__":
    file_path = "../uploads/server_inference/1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd"
    run_inference(file_path, use_federated=True)