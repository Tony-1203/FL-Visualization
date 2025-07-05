import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from simple_inference import SimpleLungNodulePredictor
import matplotlib.patches as patches


def show_nodules(ct_scan, nodules, radius=20, pad=5, max_show_num=3):
    """
    原始的结节显示函数（兼容旧接口）

    Args:
        ct_scan: CT扫描数据 (Z, Y, X)
        nodules: 结节数组，形状为 (n, 4)，包含 x, y, z, diameter
        radius: 标记框的半径
        pad: 边框宽度
        max_show_num: 最大显示数量
    """
    show_index = []
    plt.figure(figsize=(15, 5))

    for idx in range(min(nodules.shape[0], max_show_num)):
        if (
            abs(nodules[idx, 0])
            + abs(nodules[idx, 1])
            + abs(nodules[idx, 2])
            + abs(nodules[idx, 3])
            == 0
        ):
            continue

        x, y, z = int(nodules[idx, 0]), int(nodules[idx, 1]), int(nodules[idx, 2])

        # 确保坐标在有效范围内
        if z < 0 or z >= ct_scan.shape[0]:
            continue

        data = ct_scan[z].copy()  # 使用副本避免修改原数据

        # 注意 y代表纵轴，x代表横轴
        # 绘制标记框
        data[
            max(0, y - radius) : min(data.shape[0], y + radius),
            max(0, x - radius - pad) : max(0, x - radius),
        ] = 3000
        data[
            max(0, y - radius) : min(data.shape[0], y + radius),
            min(data.shape[1], x + radius) : min(data.shape[1], x + radius + pad),
        ] = 3000
        data[
            max(0, y - radius - pad) : max(0, y - radius),
            max(0, x - radius) : min(data.shape[1], x + radius),
        ] = 3000
        data[
            min(data.shape[0], y + radius) : min(data.shape[0], y + radius + pad),
            max(0, x - radius) : min(data.shape[1], x + radius),
        ] = 3000

        if z in show_index:  # 检查是否有结节在同一张切片，如果有，只显示一张
            continue

        show_index.append(z)
        plt.subplot(1, max_show_num, len(show_index))
        plt.imshow(data, cmap="gray")
        plt.title(f"Slice {z}: Nodule {idx+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_predicted_nodules(
    image_path,
    model_path="best_lung_nodule_model.pth",
    confidence_threshold=0.3,
    max_show_num=5,
    save_result=False,
):
    """
    使用训练好的模型预测并显示结节

    Args:
        image_path: CT图像文件路径(.mhd文件)
        model_path: 训练好的模型路径
        confidence_threshold: 置信度阈值
        max_show_num: 最大显示数量
        save_result: 是否保存结果图像

    Returns:
        如果save_result为True，返回保存的文件路径
    """
    print(f"正在加载图像: {image_path}")

    # 创建预测器
    predictor = SimpleLungNodulePredictor(model_path)

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
        return visualize_probability_map(
            original_image, probability_map, max_show_num=3, save_result=save_result
        )

    # 转换世界坐标为体素坐标
    nodule_voxels = []
    for x, y, z, conf in filtered_nodules[:max_show_num]:
        voxel_x = int((x - origin[0]) / spacing[0])
        voxel_y = int((y - origin[1]) / spacing[1])
        voxel_z = int((z - origin[2]) / spacing[2])

        # 确保坐标在有效范围内
        voxel_z = max(0, min(original_image.shape[0] - 1, voxel_z))
        voxel_y = max(0, min(original_image.shape[1] - 1, voxel_y))
        voxel_x = max(0, min(original_image.shape[2] - 1, voxel_x))

        nodule_voxels.append((voxel_x, voxel_y, voxel_z, conf))

    # 可视化结果
    visualize_nodules_with_model(
        original_image, probability_map, nodule_voxels, max_show_num
    )


def visualize_nodules_with_model(ct_scan, probability_map, nodules, max_show_num=5):
    """
    可视化模型预测的结节

    Args:
        ct_scan: CT扫描数据 (Z, Y, X)
        probability_map: 概率图 (Z, Y, X)
        nodules: 结节列表 [(voxel_x, voxel_y, voxel_z, confidence), ...]
        max_show_num: 最大显示数量
    """
    num_nodules = min(len(nodules), max_show_num)
    if num_nodules == 0:
        print("没有结节可显示")
        return

    fig, axes = plt.subplots(3, num_nodules, figsize=(4 * num_nodules, 12))
    if num_nodules == 1:
        axes = axes.reshape(3, 1)

    for i, (voxel_x, voxel_y, voxel_z, conf) in enumerate(nodules[:num_nodules]):
        # 原始图像
        axes[0, i].imshow(ct_scan[voxel_z], cmap="gray")
        axes[0, i].set_title(f"Original Slice {voxel_z}")
        axes[0, i].axis("off")

        # 在原始图像上标记结节位置
        circle = plt.Circle(
            (voxel_x, voxel_y), 15, color="red", fill=False, linewidth=2
        )
        axes[0, i].add_patch(circle)
        axes[0, i].text(
            voxel_x + 20,
            voxel_y,
            f"Conf: {conf:.2f}",
            color="red",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # 概率图叠加
        axes[1, i].imshow(ct_scan[voxel_z], cmap="gray", alpha=0.7)
        prob_overlay = axes[1, i].imshow(
            probability_map[voxel_z], cmap="jet", alpha=0.6, vmin=0, vmax=1
        )
        axes[1, i].set_title(f"Probability Map Slice {voxel_z}")
        axes[1, i].axis("off")

        # 标记框显示（类似原始函数）
        data_marked = ct_scan[voxel_z].copy()
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

    plt.tight_layout()

    if save_result:
        # 保存到临时文件
        import tempfile
        import time

        timestamp = int(time.time())
        save_file = os.path.join(
            tempfile.gettempdir(), f"fast_inference_result_{timestamp}.png"
        )
        plt.savefig(save_file, dpi=150, bbox_inches="tight")
        plt.close()  # 关闭图形以释放内存
        return save_file
    else:
        plt.show()
        return None


def visualize_probability_map(
    ct_scan, probability_map, max_show_num=3, save_result=False
):
    """
    可视化概率图中的高概率区域

    Args:
        ct_scan: CT扫描数据
        probability_map: 概率图
        max_show_num: 最大显示切片数
        save_result: 是否保存结果图像

    Returns:
        如果save_result为True，返回保存的文件路径
    """
    # 找到概率最高的切片
    slice_max_probs = [
        np.max(probability_map[i]) for i in range(probability_map.shape[0])
    ]
    top_slices = np.argsort(slice_max_probs)[-max_show_num:][::-1]

    fig, axes = plt.subplots(2, len(top_slices), figsize=(4 * len(top_slices), 8))
    if len(top_slices) == 1:
        axes = axes.reshape(2, 1)

    for i, slice_idx in enumerate(top_slices):
        # 原始图像
        axes[0, i].imshow(ct_scan[slice_idx], cmap="gray")
        axes[0, i].set_title(f"Original Slice {slice_idx}")
        axes[0, i].axis("off")

        # 概率图
        axes[1, i].imshow(ct_scan[slice_idx], cmap="gray", alpha=0.7)
        prob_overlay = axes[1, i].imshow(
            probability_map[slice_idx], cmap="jet", alpha=0.6, vmin=0, vmax=1
        )
        axes[1, i].set_title(f"Probability Map (Max: {slice_max_probs[slice_idx]:.3f})")
        axes[1, i].axis("off")

    plt.colorbar(prob_overlay, ax=axes[1, :], shrink=0.6, label="Nodule Probability")
    plt.tight_layout()

    if save_result:
        # 保存到临时文件
        import tempfile
        import time

        timestamp = int(time.time())
        save_file = os.path.join(
            tempfile.gettempdir(), f"fast_inference_result_{timestamp}.png"
        )
        plt.savefig(save_file, dpi=150, bbox_inches="tight")
        plt.close()  # 关闭图形以释放内存
        return save_file
    else:
        plt.show()
        return None


def demo_with_sample_data():
    """
    使用样本数据进行演示
    """
    # 查找第一个可用的mhd文件
    data_dir = "./LUNA16"

    for subset in ["subset0", "subset1", "subset2"]:
        subset_path = os.path.join(data_dir, subset)
        if os.path.exists(subset_path):
            mhd_files = [f for f in os.listdir(subset_path) if f.endswith(".mhd")]
            if mhd_files:
                sample_file = os.path.join(subset_path, mhd_files[0])
                print(f"使用样本文件: {sample_file}")

                # 使用模型预测并显示
                show_predicted_nodules(sample_file, confidence_threshold=0.2)
                return sample_file

    print("未找到可用的样本数据")
    return None


if __name__ == "__main__":
    demo_with_sample_data()
