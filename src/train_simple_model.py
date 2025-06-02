import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# 简化的3D UNet模型
class Simple3DUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(Simple3DUNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Decoder
        self.upconv3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec3 = self.conv_block(256, 128)

        self.upconv2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = self.conv_block(128, 64)

        self.upconv1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = self.conv_block(64, 32)

        self.final = nn.Conv3d(32, out_channels, 1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(pool3)

        # Decoder
        up3 = self.upconv3(bottleneck)
        up3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(up3)

        up2 = self.upconv2(dec3)
        up2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(up2)

        up1 = self.upconv1(dec2)
        up1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(up1)

        return self.final(dec1)


class SimpleLUNA16Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        csv_path,
        subset_folders=None,
        max_samples=None,
        patch_size=(64, 64, 64),
        is_custom=False,
    ):
        """
        简化的LUNA16数据集加载器
        """
        self.data_dir = data_dir
        self.patch_size = patch_size

        # 读取标注文件
        self.annotations = pd.read_csv(csv_path)
        print(f"总共有 {len(self.annotations)} 个标注")

        # 获取所有可用的mhd文件
        self.image_files = []

        if not is_custom:
            # 标准LUNA16数据集加载方式
            if subset_folders is None:
                subset_folders = [f"subset{i}" for i in range(10)]

            for subset in subset_folders:
                subset_path = os.path.join(data_dir, subset)
                if os.path.exists(subset_path):
                    mhd_files = [
                        f for f in os.listdir(subset_path) if f.endswith(".mhd")
                    ]
                    for mhd_file in mhd_files:
                        series_uid = mhd_file.replace(".mhd", "")
                        # 检查是否有对应的标注
                        if series_uid in self.annotations["seriesuid"].values:
                            self.image_files.append(
                                {
                                    "series_uid": series_uid,
                                    "image_path": os.path.join(subset_path, mhd_file),
                                    "subset": subset,
                                }
                            )
                        # else:
                        # print(series_uid, "没有对应的标注")

            print(f"找到 {len(self.image_files)} 个有标注的图像文件")
        else:
            # 自定义数据目录加载方式 - 直接扫描指定目录
            if os.path.exists(data_dir):
                mhd_files = [f for f in os.listdir(data_dir) if f.endswith(".mhd")]
                for mhd_file in mhd_files:
                    series_uid = mhd_file.replace(".mhd", "")
                    # 检查是否有对应的标注
                    if series_uid in self.annotations["seriesuid"].values:
                        self.image_files.append(
                            {
                                "series_uid": series_uid,
                                "image_path": os.path.join(data_dir, mhd_file),
                                "subset": "custom",
                            }
                        )
                print(f"找到 {len(self.image_files)} 个自定义图像文件")
            else:
                print(f"自定义数据目录 {data_dir} 不存在")

        # 限制样本数量用于快速测试
        if max_samples and max_samples < len(self.image_files):
            self.image_files = self.image_files[:max_samples]
            print(f"限制到 {max_samples} 个样本用于测试")

    def __len__(self):
        return len(self.image_files)

    def normalize_image(self, image):
        """图像标准化"""
        image = np.clip(image, -1000, 400)
        image = (image + 1000) / 1400.0  # 归一化到0-1
        return image.astype(np.float32)

    def extract_patch(self, image, label, target_size):
        """提取固定大小的patch"""
        # 如果图像比目标大小小，进行padding
        if any(s < t for s, t in zip(image.shape, target_size)):
            pad_width = [
                (max(0, (t - s) // 2), max(0, (t - s + 1) // 2))
                for s, t in zip(image.shape, target_size)
            ]
            image = np.pad(image, pad_width, mode="constant", constant_values=0)
            label = np.pad(label, pad_width, mode="constant", constant_values=0)

        # 如果图像比目标大小大，进行中心裁剪
        if any(s > t for s, t in zip(image.shape, target_size)):
            start = [(s - t) // 2 for s, t in zip(image.shape, target_size)]
            end = [st + t for st, t in zip(start, target_size)]
            image = image[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
            label = label[start[0] : end[0], start[1] : end[1], start[2] : end[2]]

        return image, label

    def __getitem__(self, idx):
        item = self.image_files[idx]
        series_uid = item["series_uid"]
        image_path = item["image_path"]

        try:
            # 加载图像
            image = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(image)

            # 获取图像的spacing和origin信息
            spacing = image.GetSpacing()  # (x, y, z)
            origin = image.GetOrigin()  # (x, y, z)

            # 获取该图像的所有标注
            nodule_annotations = self.annotations[
                self.annotations["seriesuid"] == series_uid
            ]

            # 创建标签mask
            label_array = np.zeros_like(image_array, dtype=np.float32)

            for _, annotation in nodule_annotations.iterrows():
                # 世界坐标转换为体素坐标
                world_coord = np.array(
                    [annotation["coordX"], annotation["coordY"], annotation["coordZ"]]
                )
                voxel_coord = (world_coord - np.array(origin)) / np.array(spacing)

                z, y, x = int(voxel_coord[2]), int(voxel_coord[1]), int(voxel_coord[0])
                diameter_mm = annotation["diameter_mm"]

                # 将直径转换为体素单位
                radius_voxels = max(1, int(diameter_mm / (2 * min(spacing))))

                # 在label中标记结节区域（简化为球形）
                for zi in range(
                    max(0, z - radius_voxels),
                    min(label_array.shape[0], z + radius_voxels + 1),
                ):
                    for yi in range(
                        max(0, y - radius_voxels),
                        min(label_array.shape[1], y + radius_voxels + 1),
                    ):
                        for xi in range(
                            max(0, x - radius_voxels),
                            min(label_array.shape[2], x + radius_voxels + 1),
                        ):
                            if (zi - z) ** 2 + (yi - y) ** 2 + (
                                xi - x
                            ) ** 2 <= radius_voxels**2:
                                label_array[zi, yi, xi] = 1.0

            # 标准化图像
            image_array = self.normalize_image(image_array)

            # 提取固定大小的patch
            image_array, label_array = self.extract_patch(
                image_array, label_array, self.patch_size
            )

            # 转换为tensor
            image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # 添加channel维度
            label_tensor = torch.from_numpy(label_array).long()

            return {
                "image": image_tensor,
                "label": label_tensor,
                "series_uid": series_uid,
            }

        except Exception as e:
            print(f"加载数据出错 {image_path}: {e}")
            # 返回零数据以避免中断训练
            return {
                "image": torch.zeros(1, *self.patch_size),
                "label": torch.zeros(self.patch_size, dtype=torch.long),
                "series_uid": "error",
            }


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # 将预测转换为概率
        pred = F.softmax(pred, dim=1)

        # 将target转换为one-hot编码
        target_one_hot = (
            F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        )

        # 计算Dice系数
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3, 4))
        union = torch.sum(pred, dim=(2, 3, 4)) + torch.sum(
            target_one_hot, dim=(2, 3, 4)
        )

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        # 返回1 - dice作为损失
        return 1 - torch.mean(dice)


def create_mock_dataset(patch_size=(64, 64, 64), num_samples=10):
    """创建模拟数据集用于演示"""
    class MockDataset(Dataset):
        def __init__(self, patch_size, num_samples):
            self.patch_size = patch_size
            self.num_samples = num_samples
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            # 创建随机的CT图像patch
            image = torch.randn(1, *self.patch_size) * 0.1
            # 创建随机的分割mask (大部分为背景)
            mask = torch.zeros(2, *self.patch_size)
            mask[0] = 1.0  # 背景
            # 随机添加一些前景
            if torch.rand(1) > 0.5:
                center = [s//2 for s in self.patch_size]
                size = 8
                mask[1, 
                     center[0]-size:center[0]+size,
                     center[1]-size:center[1]+size,
                     center[2]-size:center[2]+size] = 1.0
                mask[0, 
                     center[0]-size:center[0]+size,
                     center[1]-size:center[1]+size,
                     center[2]-size:center[2]+size] = 0.0
            return image, mask
    
    return MockDataset(patch_size, num_samples)


def train_simple_model(data_dir="./LUNA16", save_path="best_lung_nodule_model.pth"):
    """训练简化模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据路径
    csv_path = os.path.join(data_dir, "CSVFILES/annotations.csv")
    
    # 检查数据路径是否存在
    if not os.path.exists(data_dir):
        print(f"警告: 数据目录不存在 {data_dir}")
        print("将使用模拟数据进行训练...")
        # 创建模拟数据集用于演示
        train_dataset = create_mock_dataset()
    else:
        # 创建数据集 - 使用部分数据进行快速训练测试
        print("创建训练数据集...")
        train_dataset = SimpleLUNA16Dataset(
            data_dir=data_dir,
            csv_path=csv_path,
            subset_folders=["subset0"],  # 只使用subset0
            max_samples=5,  # 很少的样本用于快速测试
            patch_size=(64, 64, 64),
    )

    print("创建验证数据集...")
    val_dataset = SimpleLUNA16Dataset(
        data_dir=data_dir,
        csv_path=csv_path,
        subset_folders=["subset1"],  # 使用subset1作为验证集
        max_samples=2,  # 验证集更少样本
        patch_size=(64, 64, 64),
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # 小batch size
        shuffle=True,
        num_workers=0,  # 避免多进程问题
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("错误: 没有找到训练数据")
        return None

    # 创建模型
    model = Simple3DUNet(in_channels=1, out_channels=2).to(device)

    # 损失函数和优化器
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    num_epochs = 5  # 很少的epoch用于快速测试
    best_val_loss = float("inf")

    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_samples = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # 训练阶段
        for batch_idx, batch_data in enumerate(train_loader):
            try:
                if batch_data["series_uid"][0] == "error":
                    continue

                images = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_samples += 1
                print(f"  训练批次 {batch_idx+1}: loss = {loss.item():.4f}")

            except Exception as e:
                print(f"训练批次 {batch_idx} 出错: {e}")
                continue

        avg_train_loss = train_loss / max(1, train_samples)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_samples = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                try:
                    if batch_data["series_uid"][0] == "error":
                        continue

                    images = batch_data["image"].to(device)
                    labels = batch_data["label"].to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_samples += 1
                    print(f"  验证批次 {batch_idx+1}: loss = {loss.item():.4f}")

                except Exception as e:
                    print(f"验证批次 {batch_idx} 出错: {e}")
                    continue

        avg_val_loss = (
            val_loss / max(1, val_samples) if val_samples > 0 else float("inf")
        )

        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                save_path,
            )
            print(f"保存最佳模型到: {save_path}，验证损失: {best_val_loss:.4f}")

    print("训练完成!")
    return model


def main():
    """主函数 - 兼容性wrapper"""
    return train_simple_model()


if __name__ == "__main__":
    model = main()
