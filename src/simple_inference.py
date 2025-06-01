import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from train_simple_model import Simple3DUNet
import matplotlib.pyplot as plt
from scipy import ndimage
import warnings

warnings.filterwarnings("ignore")


class SimpleLungNodulePredictor:
    def __init__(self, model_path, device=None):
        """
        简化的肺结节预测器
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """加载训练好的模型"""
        model = Simple3DUNet(in_channels=1, out_channels=2).to(self.device)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"成功加载模型: {model_path}")
        else:
            print(f"警告: 模型文件不存在 {model_path}，使用随机初始化的模型")

        model.eval()
        return model

    def normalize_image(self, image):
        """图像标准化"""
        image = np.clip(image, -1000, 400)
        image = (image + 1000) / 1400.0  # 归一化到0-1
        return image.astype(np.float32)

    def predict(self, image_path, patch_size=(64, 64, 64)):
        """
        对单个CT图像进行预测
        """
        # 加载图像
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
        origin = image.GetOrigin()

        print(f"图像形状: {image_array.shape}")
        print(f"图像间距: {spacing}")

        # 标准化图像
        normalized_image = self.normalize_image(image_array)

        # 调整图像大小到合适的patch size进行预测
        original_shape = normalized_image.shape

        # 简单的resize到patch_size
        if any(s != p for s, p in zip(original_shape, patch_size)):
            # 使用简单的中心裁剪或padding
            resized_image = self.resize_image(normalized_image, patch_size)
        else:
            resized_image = normalized_image

        # 准备输入tensor
        input_tensor = (
            torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).to(self.device)
        )

        # 进行推理
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = F.softmax(prediction, dim=1)
            nodule_prob = prediction[0, 1].cpu().numpy()  # 取结节类别的概率

        # 将预测结果调整回原始大小
        if nodule_prob.shape != original_shape:
            nodule_prob = self.resize_image(nodule_prob, original_shape)

        # 检测结节
        nodules = self.detect_nodules(nodule_prob, spacing, origin, threshold=0.3)

        return nodules, nodule_prob, image_array, spacing, origin

    def resize_image(self, image, target_shape):
        """调整图像大小"""
        # 使用scipy的zoom进行resize
        zoom_factors = [t / s for t, s in zip(target_shape, image.shape)]
        resized = ndimage.zoom(image, zoom_factors, order=1)
        return resized

    def detect_nodules(self, probability_map, spacing, origin, threshold=0.3):
        """
        从概率图中检测结节
        """
        # 二值化
        binary_mask = probability_map > threshold

        # 连通分量分析
        labeled_mask, num_features = ndimage.label(binary_mask)

        nodules = []
        for i in range(1, num_features + 1):
            # 获取连通分量
            component = labeled_mask == i

            # 计算连通分量的大小（过滤太小的区域）
            component_size = np.sum(component)
            if component_size < 5:  # 最小体素数阈值
                continue

            # 计算质心
            center_of_mass = ndimage.center_of_mass(component)
            z, y, x = center_of_mass

            # 转换为世界坐标
            world_x = x * spacing[0] + origin[0]
            world_y = y * spacing[1] + origin[1]
            world_z = z * spacing[2] + origin[2]

            # 计算该区域的平均置信度
            confidence = np.mean(probability_map[component])

            nodules.append((world_x, world_y, world_z, confidence))

        # 按置信度排序
        nodules.sort(key=lambda x: x[3], reverse=True)

        return nodules


def simple_predict_and_visualize(image_path, model_path="best_lung_nodule_model.pth"):
    """
    简化的预测并可视化结果
    """
    # 创建预测器
    predictor = SimpleLungNodulePredictor(model_path)

    # 进行预测
    print(f"正在预测: {image_path}")
    nodules, probability_map, original_image, spacing, origin = predictor.predict(
        image_path
    )

    print(f"检测到 {len(nodules)} 个可能的结节:")
    for i, (x, y, z, conf) in enumerate(nodules):
        print(f"  结节 {i+1}: 位置=({x:.1f}, {y:.1f}, {z:.1f}), 置信度={conf:.3f}")

    # 可视化结果
    simple_visualize_results(original_image, probability_map, nodules, spacing, origin)

    return nodules


def simple_visualize_results(
    original_image, probability_map, nodules, spacing, origin, max_slices=3
):
    """
    简化的可视化预测结果
    """
    # 转换结节的世界坐标为体素坐标
    nodule_voxels = []
    for x, y, z, conf in nodules:
        voxel_x = int((x - origin[0]) / spacing[0])
        voxel_y = int((y - origin[1]) / spacing[1])
        voxel_z = int((z - origin[2]) / spacing[2])

        # 确保坐标在有效范围内
        voxel_x = max(0, min(original_image.shape[2] - 1, voxel_x))
        voxel_y = max(0, min(original_image.shape[1] - 1, voxel_y))
        voxel_z = max(0, min(original_image.shape[0] - 1, voxel_z))

        nodule_voxels.append((voxel_x, voxel_y, voxel_z, conf))

    # 找到包含结节的切片
    nodule_slices = []
    for _, _, z, _ in nodule_voxels:
        nodule_slices.append(z)

    # 如果没有检测到结节，显示中间几个切片和概率最高的切片
    if not nodule_slices:
        mid_slice = original_image.shape[0] // 2
        slice_max_probs = [
            np.max(probability_map[i]) for i in range(probability_map.shape[0])
        ]
        top_prob_slice = np.argmax(slice_max_probs)
        nodule_slices = [mid_slice, top_prob_slice]

    # 限制显示的切片数量
    nodule_slices = sorted(set(nodule_slices))[:max_slices]

    fig, axes = plt.subplots(2, len(nodule_slices), figsize=(4 * len(nodule_slices), 8))
    if len(nodule_slices) == 1:
        axes = axes.reshape(2, 1)

    for i, slice_idx in enumerate(nodule_slices):
        # 显示原始图像
        axes[0, i].imshow(original_image[slice_idx], cmap="gray")
        axes[0, i].set_title(f"Original Slice {slice_idx}")
        axes[0, i].axis("off")

        # 在原始图像上标记结节
        for voxel_x, voxel_y, voxel_z, conf in nodule_voxels:
            if voxel_z == slice_idx:
                # 画一个圆圈标记结节
                circle = plt.Circle(
                    (voxel_x, voxel_y), 10, color="red", fill=False, linewidth=2
                )
                axes[0, i].add_patch(circle)
                axes[0, i].text(
                    voxel_x + 15,
                    voxel_y,
                    f"{conf:.2f}",
                    color="red",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

        # 显示概率图
        axes[1, i].imshow(original_image[slice_idx], cmap="gray", alpha=0.7)
        probability_overlay = axes[1, i].imshow(
            probability_map[slice_idx], cmap="jet", alpha=0.5, vmin=0, vmax=1
        )
        axes[1, i].set_title(f"Probability Map Slice {slice_idx}")
        axes[1, i].axis("off")

    # 添加颜色条
    plt.colorbar(
        probability_overlay, ax=axes[1, :], shrink=0.6, label="Nodule Probability"
    )

    plt.tight_layout()
    plt.show()


def test_simple_prediction():
    """测试简化预测功能"""
    # 查找第一个可用的mhd文件进行测试
    data_dir = "./LUNA16"

    # 寻找测试文件
    test_file = None
    for subset in ["subset0", "subset1", "subset2"]:
        subset_path = os.path.join(data_dir, subset)
        if os.path.exists(subset_path):
            mhd_files = [f for f in os.listdir(subset_path) if f.endswith(".mhd")]
            if mhd_files:
                test_file = os.path.join(subset_path, mhd_files[0])
                break

    if test_file:
        print(f"使用测试文件: {test_file}")
        nodules = simple_predict_and_visualize(test_file)
        return nodules
    else:
        print("未找到测试文件")
        return []


if __name__ == "__main__":
    test_simple_prediction()
