"""
联邦学习推理模块
支持使用联邦训练的模型进行肺结节检测
"""

import os
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
from scipy.ndimage import label, generate_binary_structure
import warnings
import matplotlib

matplotlib.use("Agg")  # 使用非GUI后端
from train_simple_model import Simple3DUNet

warnings.filterwarnings("ignore")

# 添加安全的全局对象，用于解决PyTorch 2.6的权限问题
torch.serialization.add_safe_globals([np.core.multiarray.scalar])


class FederatedLungNodulePredictor:
    """联邦学习肺结节预测器"""

    def __init__(self, model_path, device=None):
        """
        初始化联邦学习预测器

        Args:
            model_path: 联邦训练模型路径
            device: 计算设备
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.load_federated_model(model_path)

    def load_federated_model(self, model_path):
        """加载联邦训练的模型"""
        model = Simple3DUNet(in_channels=1, out_channels=2).to(self.device)

        if os.path.exists(model_path):
            print(f"加载联邦学习模型: {model_path}")
            # 使用安全的全局对象上下文管理器
            with torch.serialization.safe_globals([np.core.multiarray.scalar]):
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=False
                )

            # 检查是否是联邦学习保存的格式
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                round_num = checkpoint.get("round_num", "Unknown")
                print(f"模型来自联邦学习第 {round_num} 轮")
            else:
                # 兼容普通模型格式
                model.load_state_dict(checkpoint, weights_only=False)
                print("加载普通格式模型")
        else:
            print(f"警告: 模型文件不存在 {model_path}，使用随机初始化的模型")

        model.eval()
        return model

    def normalize_image(self, image):
        """图像标准化"""
        image = np.clip(image, -1000, 400)
        image = (image + 1000) / 1400.0  # 归一化到0-1
        return image.astype(np.float32)

    def predict(self, image_path, patch_size=(64, 64, 64), confidence_threshold=0.3):
        """
        对单个CT图像进行预测

        Args:
            image_path: 图像路径
            patch_size: 预测块大小
            confidence_threshold: 置信度阈值

        Returns:
            tuple: (nodules, probability_map, original_image, spacing, origin)
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

        # 预测
        probability_map = self.sliding_window_prediction(normalized_image, patch_size)

        # 检测结节
        nodules = self.detect_nodules(
            probability_map, spacing, origin, threshold=confidence_threshold
        )

        return nodules, probability_map, image_array, spacing, origin

    def sliding_window_prediction(self, image, patch_size=(64, 64, 64), stride=None):
        """
        滑动窗口预测

        Args:
            image: 输入图像
            patch_size: 窗口大小
            stride: 滑动步长

        Returns:
            概率图
        """
        if stride is None:
            stride = [s // 2 for s in patch_size]  # 默认50%重叠

        original_shape = image.shape
        probability_map = np.zeros(original_shape, dtype=np.float32)
        count_map = np.zeros(original_shape, dtype=np.float32)

        # 计算需要的patch数量
        patch_positions = []
        for z in range(0, original_shape[0], stride[0]):
            for y in range(0, original_shape[1], stride[1]):
                for x in range(0, original_shape[2], stride[2]):
                    # 确保patch不超出边界
                    z_end = min(z + patch_size[0], original_shape[0])
                    y_end = min(y + patch_size[1], original_shape[1])
                    x_end = min(x + patch_size[2], original_shape[2])

                    z_start = max(0, z_end - patch_size[0])
                    y_start = max(0, y_end - patch_size[1])
                    x_start = max(0, x_end - patch_size[2])

                    patch_positions.append(
                        (z_start, y_start, x_start, z_end, y_end, x_end)
                    )

        print(f"总共需要预测 {len(patch_positions)} 个patch")

        # 批量预测
        for i, (z1, y1, x1, z2, y2, x2) in enumerate(patch_positions):
            # 提取patch
            patch = image[z1:z2, y1:y2, x1:x2]

            # 如果patch大小不足，进行padding
            if patch.shape != patch_size:
                padded_patch = np.zeros(patch_size, dtype=np.float32)
                padded_patch[: patch.shape[0], : patch.shape[1], : patch.shape[2]] = (
                    patch
                )
                patch = padded_patch

            # 转换为tensor并预测
            input_tensor = (
                torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
            )

            with torch.no_grad():
                output = self.model(input_tensor)
                prob = torch.sigmoid(output[0, 1]).cpu().numpy()  # 取结节通道的概率

            # 将结果添加到概率图
            actual_patch_shape = (z2 - z1, y2 - y1, x2 - x1)
            prob_resized = prob[
                : actual_patch_shape[0],
                : actual_patch_shape[1],
                : actual_patch_shape[2],
            ]

            probability_map[z1:z2, y1:y2, x1:x2] += prob_resized
            count_map[z1:z2, y1:y2, x1:x2] += 1

            if (i + 1) % max(1, len(patch_positions) // 10) == 0:
                print(f"完成预测: {i + 1}/{len(patch_positions)}")

        # 平均化重叠区域
        probability_map = np.divide(
            probability_map,
            count_map,
            out=np.zeros_like(probability_map),
            where=count_map != 0,
        )

        return probability_map

    def detect_nodules(
        self, probability_map, spacing, origin, threshold=0.3, min_size=8
    ):
        """
        从概率图中检测结节

        Args:
            probability_map: 概率图
            spacing: 图像间距
            origin: 图像原点
            threshold: 概率阈值
            min_size: 最小连通域大小

        Returns:
            结节列表: [(x, y, z, confidence), ...]
        """
        # 二值化
        binary_map = (probability_map > threshold).astype(np.uint8)

        # 连通域分析
        structure = generate_binary_structure(3, 3)  # 3D连通性
        labeled_map, num_features = label(binary_map, structure=structure)

        nodules = []

        for i in range(1, num_features + 1):
            # 获取连通域
            component_mask = labeled_map == i
            component_size = np.sum(component_mask)

            # 过滤太小的连通域
            if component_size < min_size:
                continue

            # 计算质心
            coords = np.where(component_mask)
            centroid_z = np.mean(coords[0])
            centroid_y = np.mean(coords[1])
            centroid_x = np.mean(coords[2])

            # 计算该区域的平均置信度
            confidence = np.mean(probability_map[component_mask])

            # 转换为世界坐标
            world_x = centroid_x * spacing[0] + origin[0]
            world_y = centroid_y * spacing[1] + origin[1]
            world_z = centroid_z * spacing[2] + origin[2]

            nodules.append((world_x, world_y, world_z, confidence))

        # 按置信度排序
        nodules.sort(key=lambda x: x[3], reverse=True)

        return nodules

    def predict_fast(
        self, image_path, patch_size=(32, 32, 32), confidence_threshold=0.5
    ):
        """
        快速预测模式 - 使用更小的patch和更大的步长以提高速度

        Args:
            image_path: 图像路径
            patch_size: 预测块大小（较小以提高速度）
            confidence_threshold: 置信度阈值（较高以减少误检）

        Returns:
            tuple: (nodules, probability_map, original_image, spacing, origin)
        """
        # 加载图像
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
        origin = image.GetOrigin()

        print(f"快速模式 - 图像形状: {image_array.shape}")
        print(f"快速模式 - 图像间距: {spacing}")

        # 对图像进行下采样以提高速度
        downsample_factor = 2
        downsampled_shape = tuple(s // downsample_factor for s in image_array.shape)
        downsampled_image = ndimage.zoom(image_array, 1 / downsample_factor, order=1)

        # 标准化图像
        normalized_image = self.normalize_image(downsampled_image)

        # 快速预测（使用更大的步长）
        stride = [s for s in patch_size]  # 无重叠以提高速度
        probability_map_downsampled = self.sliding_window_prediction(
            normalized_image, patch_size, stride
        )

        # 将概率图上采样回原始尺寸
        probability_map = ndimage.zoom(
            probability_map_downsampled, downsample_factor, order=1
        )

        # 确保概率图与原始图像尺寸匹配
        if probability_map.shape != image_array.shape:
            probability_map = ndimage.zoom(
                probability_map,
                [s1 / s2 for s1, s2 in zip(image_array.shape, probability_map.shape)],
                order=1,
            )

        # 检测结节
        nodules = self.detect_nodules(
            probability_map, spacing, origin, threshold=confidence_threshold
        )

        print(f"快速模式完成 - 检测到 {len(nodules)} 个候选结节")

        return nodules, probability_map, image_array, spacing, origin


def predict_with_federated_model(
    image_path, model_path="./src/best_federated_lung_nodule_model.pth", fast_mode=False
):
    """
    使用联邦学习模型进行预测的便捷函数

    Args:
        image_path: CT图像路径
        model_path: 联邦学习模型路径
        fast_mode: 是否使用快速模式（减少计算时间）

    Returns:
        预测结果
    """
    predictor = FederatedLungNodulePredictor(model_path)

    if fast_mode:
        # 快速模式：降低分辨率和减少处理步骤
        nodules, prob_map, image, spacing, origin = predictor.predict_fast(image_path)
    else:
        nodules, prob_map, image, spacing, origin = predictor.predict(image_path)

    print(f"检测到 {len(nodules)} 个结节候选:")
    for i, (x, y, z, conf) in enumerate(nodules):
        print(f"  结节 {i+1}: 世界坐标=({x:.1f}, {y:.1f}, {z:.1f}), 置信度={conf:.3f}")

    return nodules, prob_map, image, spacing, origin


def visualize_federated_results(
    image_array,
    probability_map,
    nodules,
    spacing,
    origin,
    max_slices=3,
    save_path=False,
):
    """
    可视化联邦学习预测结果，正确标记肿瘤位置

    Args:
        image_array: 原始CT图像
        probability_map: 概率图
        nodules: 检测到的结节列表 [(world_x, world_y, world_z, confidence), ...]
        spacing: 图像间距
        origin: 图像原点
        max_slices: 最大显示切片数
        save_path: 是否保存图像，如果为True则返回保存路径

    Returns:
        如果save_path为True，返回保存的文件路径
    """
    if len(nodules) == 0:
        print("未检测到结节，显示概率最高的切片")
        return visualize_probability_map_only(
            image_array, probability_map, max_slices, save_path
        )

    # 转换世界坐标为体素坐标
    nodule_voxels = []
    for world_x, world_y, world_z, conf in nodules[:max_slices]:
        voxel_x = int((world_x - origin[0]) / spacing[0])
        voxel_y = int((world_y - origin[1]) / spacing[1])
        voxel_z = int((world_z - origin[2]) / spacing[2])

        # 确保坐标在有效范围内
        voxel_z = max(0, min(image_array.shape[0] - 1, voxel_z))
        voxel_y = max(0, min(image_array.shape[1] - 1, voxel_y))
        voxel_x = max(0, min(image_array.shape[2] - 1, voxel_x))

        nodule_voxels.append((voxel_x, voxel_y, voxel_z, conf))

    num_nodules = len(nodule_voxels)
    fig, axes = plt.subplots(3, num_nodules, figsize=(4 * num_nodules, 12))
    if num_nodules == 1:
        axes = axes.reshape(3, 1)

    for i, (voxel_x, voxel_y, voxel_z, conf) in enumerate(nodule_voxels):
        # 第一行：原始图像 + 圆圈标记
        axes[0, i].imshow(image_array[voxel_z], cmap="gray")
        axes[0, i].set_title(f"Original Image - Slice {voxel_z}")
        axes[0, i].axis("off")

        # 添加红色圆圈标记结节位置
        circle = plt.Circle(
            (voxel_x, voxel_y), 15, color="red", fill=False, linewidth=2
        )
        axes[0, i].add_patch(circle)

        # 添加置信度标签
        axes[0, i].text(
            voxel_x + 20,
            voxel_y,
            f"Confidence: {conf:.3f}",
            color="red",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # 第二行：概率图叠加
        axes[1, i].imshow(image_array[voxel_z], cmap="gray", alpha=0.7)
        prob_overlay = axes[1, i].imshow(
            probability_map[voxel_z], cmap="jet", alpha=0.6, vmin=0, vmax=1
        )
        axes[1, i].set_title(f"Probability Map - Slice {voxel_z}")
        axes[1, i].axis("off")

        # 在概率图上也添加圆圈标记
        circle2 = plt.Circle(
            (voxel_x, voxel_y), 15, color="white", fill=False, linewidth=2
        )
        axes[1, i].add_patch(circle2)

        # 第三行：标记框显示（类似show_nodules.py的风格）
        data_marked = image_array[voxel_z].copy()
        radius = 20
        pad = 3

        # 绘制标记框
        y_min, y_max = max(0, voxel_y - radius), min(
            data_marked.shape[0], voxel_y + radius
        )
        x_min, x_max = max(0, voxel_x - radius), min(
            data_marked.shape[1], voxel_x + radius
        )

        # 竖线
        data_marked[
            y_min:y_max, max(0, voxel_x - radius - pad) : max(0, voxel_x - radius)
        ] = np.max(data_marked)
        data_marked[
            y_min:y_max,
            min(data_marked.shape[1], voxel_x + radius) : min(
                data_marked.shape[1], voxel_x + radius + pad
            ),
        ] = np.max(data_marked)
        # 横线
        data_marked[
            max(0, voxel_y - radius - pad) : max(0, voxel_y - radius), x_min:x_max
        ] = np.max(data_marked)
        data_marked[
            min(data_marked.shape[0], voxel_y + radius) : min(
                data_marked.shape[0], voxel_y + radius + pad
            ),
            x_min:x_max,
        ] = np.max(data_marked)

        axes[2, i].imshow(data_marked, cmap="gray")
        axes[2, i].set_title(f"Marked Nodule {i+1}")
        axes[2, i].axis("off")

    plt.colorbar(prob_overlay, ax=axes[1, :], shrink=0.6, label="Nodule Probability")
    plt.suptitle("Federated Learning Model Prediction Results", fontsize=16)
    plt.tight_layout()

    if save_path:
        # 保存到临时文件
        import tempfile
        import time

        timestamp = int(time.time())
        save_file = os.path.join(
            tempfile.gettempdir(), f"federated_inference_result_{timestamp}.png"
        )
        plt.savefig(save_file, dpi=150, bbox_inches="tight")
        plt.close()
        return save_file
    else:
        plt.show()
        return None


def visualize_probability_map_only(
    image_array, probability_map, max_slices=3, save_path=False
):
    """
    当没有检测到结节时，可视化概率最高的区域

    Args:
        image_array: 原始CT图像
        probability_map: 概率图
        max_slices: 最大显示切片数
        save_path: 是否保存图像，如果为True则返回保存路径

    Returns:
        如果save_path为True，返回保存的文件路径
    """
    # 找到概率最高的切片
    slice_max_probs = [
        np.max(probability_map[i]) for i in range(probability_map.shape[0])
    ]
    top_slices = np.argsort(slice_max_probs)[-max_slices:][::-1]

    fig, axes = plt.subplots(2, len(top_slices), figsize=(4 * len(top_slices), 8))
    if len(top_slices) == 1:
        axes = axes.reshape(2, 1)

    for i, slice_idx in enumerate(top_slices):
        # 原始图像
        axes[0, i].imshow(image_array[slice_idx], cmap="gray")
        axes[0, i].set_title(f"Original Image - Slice {slice_idx}")
        axes[0, i].axis("off")

        # 概率图
        axes[1, i].imshow(image_array[slice_idx], cmap="gray", alpha=0.7)
        prob_overlay = axes[1, i].imshow(
            probability_map[slice_idx], cmap="jet", alpha=0.6, vmin=0, vmax=1
        )
        axes[1, i].set_title(f"Probability Map (Max: {slice_max_probs[slice_idx]:.3f})")
        axes[1, i].axis("off")

    plt.colorbar(prob_overlay, ax=axes[1, :], shrink=0.6, label="Nodule Probability")
    plt.suptitle("Federated Learning Model Prediction Results", fontsize=16)
    plt.tight_layout()

    if save_path:
        # 保存到临时文件
        import tempfile
        import time

        timestamp = int(time.time())
        save_file = os.path.join(
            tempfile.gettempdir(), f"federated_inference_result_{timestamp}.png"
        )
        plt.savefig(save_file, dpi=150, bbox_inches="tight")
        plt.close()  # 关闭图形以释放内存
        return save_file
    else:
        plt.show()
        return None


def demo_federated_prediction():
    """联邦学习预测演示"""
    # 查找样本文件
    data_dir = "./LUNA16"

    for subset in ["subset0", "subset1", "subset2"]:
        subset_path = os.path.join(data_dir, subset)
        if os.path.exists(subset_path):
            mhd_files = [f for f in os.listdir(subset_path) if f.endswith(".mhd")]
            if mhd_files:
                sample_file = os.path.join(subset_path, mhd_files[0])
                print(f"使用样本文件: {sample_file}")

                # 使用联邦学习模型预测
                nodules, prob_map, image, spacing, origin = (
                    predict_with_federated_model(sample_file)
                )

                # 可视化结果
                visualize_federated_results(image, prob_map, nodules, spacing, origin)
                return sample_file

    print("未找到可用的样本数据")
    return None


def show_federated_predicted_nodules(
    image_path,
    model_path="best_federated_lung_nodule_model.pth",
    confidence_threshold=0.3,
    max_show_num=5,
):
    """
    使用联邦学习模型预测并显示结节（类似show_nodules.py的接口）

    Args:
        image_path: CT图像文件路径(.mhd文件)
        model_path: 联邦学习模型路径
        confidence_threshold: 置信度阈值
        max_show_num: 最大显示数量
    """
    print(f"正在加载图像: {image_path}")

    # 创建联邦学习预测器
    predictor = FederatedLungNodulePredictor(model_path)

    # 进行预测
    nodules, probability_map, original_image, spacing, origin = predictor.predict(
        image_path
    )

    # 过滤低置信度的结节
    filtered_nodules = [
        (x, y, z, conf) for x, y, z, conf in nodules if conf >= confidence_threshold
    ]

    print(f"检测到 {len(filtered_nodules)} 个置信度 >= {confidence_threshold} 的结节:")
    for i, (x, y, z, conf) in enumerate(filtered_nodules):
        print(f"  结节 {i+1}: 世界坐标=({x:.1f}, {y:.1f}, {z:.1f}), 置信度={conf:.3f}")

    if not filtered_nodules:
        print("未检测到高置信度的结节，显示概率图...")
        # 显示概率最高的区域
        visualize_probability_map_only(original_image, probability_map, max_show_num=3)
        return

    # 可视化结果
    visualize_federated_results(
        original_image,
        probability_map,
        filtered_nodules[:max_show_num],
        spacing,
        origin,
    )


if __name__ == "__main__":
    demo_federated_prediction()
