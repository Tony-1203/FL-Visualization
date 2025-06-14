"""
联邦学习训练系统 - LUNA16肺结节检测
使用FedAvg算法进行分布式训练

主要组件:
1. FederatedServer: 联邦服务器，负责模型聚合
2. FederatedClient: 联邦客户端，负责本地训练
3. FedAvg算法实现
4. 数据分片和分布
"""

import os
import copy
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib

matplotlib.use("Agg")  # 使用非GUI后端
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from typing import List, Dict, Tuple
import random
from collections import OrderedDict
from datetime import datetime
import sys  # 添加sys导入
import json  # 添加JSON支持用于WebSocket消息


# 日志函数 - 尝试从app.py导入，失败则创建本地版本
def add_server_log(message):
    """添加服务器日志"""
    try:
        # 尝试导入app.py的日志函数
        import builtins

        if hasattr(builtins, "add_server_log"):
            builtins.add_server_log(message)
            return
    except:
        pass

    # 如果导入失败，直接打印
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[SERVER {timestamp}] {message}")


def add_training_log(message):
    """添加训练日志"""
    try:
        # 尝试导入app.py的日志函数
        import builtins

        if hasattr(builtins, "add_training_log"):
            builtins.add_training_log(message)
            return
    except:
        pass

    # 如果导入失败，直接打印
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[TRAINING {timestamp}] {message}")


def broadcast_training_data(client_id, data_type, data):
    """
    广播训练数据到WebSocket客户端

    Args:
        client_id: 客户端ID
        data_type: 数据类型 ('training_start', 'training_progress', 'training_complete', 'round_complete')
        data: 要发送的数据
    """
    try:
        import builtins

        if hasattr(builtins, "socketio"):
            socketio = builtins.socketio
            message = {
                "type": data_type,
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                **data,
            }

            # 发送到特定客户端房间
            socketio.emit(
                f"client_{client_id}_training_update",
                message,
                room=f"client_{client_id}_training",
            )

            # 发送到服务器仪表盘（所有客户端的数据聚合）
            socketio.emit("server_training_update", message, room="server_training")

            print(f"✅ WebSocket广播成功: 客户端{client_id} - {data_type}")
    except Exception as e:
        print(f"❌ WebSocket广播失败: {e}")


def broadcast_server_training_data(data_type, data):
    """
    广播服务器端训练数据

    Args:
        data_type: 数据类型
        data: 要发送的数据
    """
    try:
        import builtins

        if hasattr(builtins, "socketio"):
            socketio = builtins.socketio
            message = {
                "type": data_type,
                "timestamp": datetime.now().isoformat(),
                **data,
            }

            socketio.emit("server_training_update", message, room="server_training")
            print(f"✅ 服务器WebSocket广播成功: {data_type}")
    except Exception as e:
        print(f"❌ 服务器WebSocket广播失败: {e}")


# 导入简化的模型和数据集
from train_simple_model import Simple3DUNet, SimpleLUNA16Dataset, DiceLoss

warnings.filterwarnings("ignore")


class EmptyDataset(Dataset):
    """空数据集，用于没有数据的客户端"""

    def __init__(self, patch_size=(64, 64, 64)):
        self.patch_size = patch_size

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        # 这个方法不会被调用，因为长度为0
        raise IndexError("Empty dataset has no items")


class FederatedServer:
    """联邦学习服务器"""

    def __init__(self, model_class, model_kwargs, device="cpu"):
        """
        初始化联邦服务器

        Args:
            model_class: 模型类
            model_kwargs: 模型初始化参数
            device: 计算设备
        """
        self.device = device
        self.global_model = model_class(**model_kwargs).to(device)
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.round_num = 0

        # 存储训练历史
        self.training_history = {"rounds": [], "average_loss": [], "client_losses": []}

    def get_global_model_params(self):
        """获取全局模型参数"""
        return copy.deepcopy(self.global_model.state_dict())

    def federated_averaging(
        self, client_params_list: List[Dict], client_weights: List[float]
    ):
        """
        执行FedAvg算法 - 联邦平均

        Args:
            client_params_list: 客户端模型参数列表
            client_weights: 客户端权重（通常基于数据量）
        """

        try:
            # 记录聚合开始
            add_server_log(
                f"模型聚合开始 - 第{self.round_num + 1}轮, {len(client_params_list)}个客户端, 权重: {client_weights}"
            )

            # 确保权重归一化
            total_weight = sum(client_weights)
            normalized_weights = [w / total_weight for w in client_weights]

            print(
                f"开始模型聚合 - {len(client_params_list)} 个客户端，权重: {normalized_weights}"
            )

            # 初始化聚合参数
            global_params = OrderedDict()

            # 获取第一个客户端的参数结构
            first_client_params = client_params_list[0]

            # 对每个参数进行加权平均
            for param_name in first_client_params.keys():
                param_tensor = first_client_params[param_name]

                # 检查是否是需要跳过的参数（如BatchNorm的num_batches_tracked）
                if param_name.endswith("num_batches_tracked"):
                    # 对于不需要聚合的参数，直接使用第一个客户端的值
                    global_params[param_name] = param_tensor.clone()
                    continue

                # 确保参数是浮点类型以支持加权平均
                if param_tensor.dtype in [torch.int32, torch.int64, torch.long]:
                    # 对于整型参数，取第一个客户端的值（通常是索引类型参数）
                    global_params[param_name] = param_tensor.clone()
                    continue

                # 对浮点参数进行加权求和
                weighted_sum = torch.zeros_like(param_tensor, dtype=torch.float32)

                for client_params, weight in zip(
                    client_params_list, normalized_weights
                ):
                    client_param = client_params[param_name]
                    if client_param.dtype != torch.float32:
                        client_param = client_param.float()
                    weighted_sum += weight * client_param

                # 恢复原始数据类型
                if param_tensor.dtype != torch.float32:
                    weighted_sum = weighted_sum.to(param_tensor.dtype)

                global_params[param_name] = weighted_sum

            # 更新全局模型
            self.global_model.load_state_dict(global_params)
            self.round_num += 1

            # 记录聚合成功
            add_server_log(f"全局模型已更新 - 第 {self.round_num} 轮")

            print(f"✅ 全局模型已更新 - 第 {self.round_num} 轮")

        except Exception as e:
            # 记录聚合失败
            add_server_log(
                f"模型聚合错误 - 第{self.round_num + 1}轮: {str(e)} ({type(e).__name__})"
            )

            print(f"❌ 模型聚合过程中出现错误: {e}")
            print(f"错误类型: {type(e).__name__}")
            raise e

    def evaluate_global_model(self, test_loader):
        """评估全局模型性能"""

        add_server_log(f"开始评估全局模型性能 - 第{self.round_num}轮")

        self.global_model.eval()
        total_loss = 0.0
        num_batches = 0

        criterion = DiceLoss()  # 使用原始训练的DiceLoss

        with torch.no_grad():
            for batch in test_loader:
                if batch["series_uid"][0] == "error":
                    continue

                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.global_model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        add_server_log(
            f"全局模型评估完成 - 第{self.round_num}轮, 损失: {avg_loss:.4f}, 批次数: {num_batches}"
        )

        print(f"全局模型验证损失: {avg_loss:.4f}")

        return avg_loss

    def save_global_model(self, save_path="federated_global_model.pth"):
        """保存全局模型"""

        try:
            torch.save(
                {
                    "model_state_dict": self.global_model.state_dict(),
                    "round_num": self.round_num,
                    "training_history": self.training_history,
                },
                save_path,
            )

            add_server_log(
                f"全局模型保存成功 - 第{self.round_num}轮, 路径: {save_path}"
            )

            print(f"全局模型已保存到: {save_path}")
        except Exception as e:
            add_server_log(
                f"模型保存失败 - 第{self.round_num}轮: {str(e)}, 路径: {save_path}"
            )
            raise e


class FederatedClient:
    """联邦学习客户端"""

    def __init__(self, client_id: int, model_class, model_kwargs, device="cpu"):
        """
        初始化联邦客户端

        Args:
            client_id: 客户端ID
            model_class: 模型类
            model_kwargs: 模型初始化参数
            device: 计算设备
        """
        self.client_id = client_id
        self.device = device
        self.model = model_class(**model_kwargs).to(device)
        self.model_class = model_class
        self.model_kwargs = model_kwargs

        # 本地训练参数
        self.learning_rate = 0.001
        self.local_epochs = 3

        # 训练历史
        self.training_history = []

    def load_global_model(self, global_params: Dict):
        """加载全局模型参数"""
        self.model.load_state_dict(global_params)

    def local_train(self, train_loader, epochs=None):
        """
        执行本地训练

        Args:
            train_loader: 训练数据加载器
            epochs: 本地训练轮数

        Returns:
            训练损失列表
        """

        if epochs is None:
            epochs = self.local_epochs

        add_training_log(
            f"客户端{self.client_id}开始本地训练: {epochs}轮, 数据量: {len(train_loader.dataset)}"
        )

        # 广播训练开始消息
        broadcast_training_data(
            self.client_id,
            "training_start",
            {
                "round": getattr(self, "current_round", 1),
                "epochs": epochs,
                "data_size": len(train_loader.dataset),
            },
        )

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = DiceLoss()  # 使用原始训练的DiceLoss

        epoch_losses = []

        print(f"客户端 {self.client_id} 开始本地训练 ({epochs} 轮)...")

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                try:
                    if batch["series_uid"][0] == "error":
                        continue

                    images = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                    # 实时广播训练进度（每10个批次）
                    if batch_idx % 10 == 0:
                        print(
                            f"  客户端 {self.client_id} - Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}"
                        )

                        # 广播实时损失
                        broadcast_training_data(
                            self.client_id,
                            "training_progress",
                            {
                                "epoch": epoch + 1,
                                "batch": batch_idx + 1,
                                "loss": loss.item(),
                                "average_loss": (
                                    total_loss / (batch_idx + 1)
                                    if batch_idx > 0
                                    else loss.item()
                                ),
                            },
                        )

                except Exception as e:
                    add_training_log(
                        f"客户端{self.client_id}批次训练错误 - Epoch{epoch + 1}, Batch{batch_idx}: {str(e)}"
                    )
                    print(f"  客户端 {self.client_id} 训练出错: {e}")
                    continue

            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            epoch_losses.append(avg_loss)

            add_training_log(
                f"客户端{self.client_id} Epoch{epoch + 1}完成 - 损失: {avg_loss:.4f}, 批次数: {num_batches}"
            )

            print(
                f"  客户端 {self.client_id} - Epoch {epoch+1} 平均损失: {avg_loss:.4f}"
            )

            # 广播每个epoch完成的消息
            broadcast_training_data(
                self.client_id,
                "training_progress",
                {
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "total_epochs": epochs,
                    "epoch_complete": True,
                },
            )

        add_training_log(
            f"客户端{self.client_id}本地训练完成 - 总轮数: {epochs}, 最终平均损失: {np.mean(epoch_losses) if epoch_losses else 0.0:.4f}"
        )

        # 广播训练完成消息
        broadcast_training_data(
            self.client_id,
            "training_complete",
            {
                "final_loss": np.mean(epoch_losses) if epoch_losses else 0.0,
                "loss_history": epoch_losses,
                "total_epochs": epochs,
            },
        )

        self.training_history.extend(epoch_losses)
        return epoch_losses

    def get_model_params(self):
        """获取本地模型参数"""
        return copy.deepcopy(self.model.state_dict())

    def get_data_size(self, data_loader):
        """获取数据集大小"""
        return len(data_loader.dataset)


class FederatedLearningCoordinator:
    """联邦学习协调器"""

    def __init__(
        self, num_clients=3, model_class=Simple3DUNet, model_kwargs=None, device="cpu"
    ):
        """
        初始化联邦学习协调器

        Args:
            num_clients: 客户端数量
            model_class: 模型类
            model_kwargs: 模型初始化参数
            device: 计算设备
        """
        if model_kwargs is None:
            model_kwargs = {"in_channels": 1, "out_channels": 2}

        self.num_clients = num_clients
        self.device = device

        # 创建服务器
        self.server = FederatedServer(model_class, model_kwargs, device)

        # 创建客户端
        self.clients = [
            FederatedClient(i, model_class, model_kwargs, device)
            for i in range(num_clients)
        ]

        print(f"联邦学习系统初始化完成 - {num_clients} 个客户端")

    def distribute_data(self, dataset, distribution_strategy="iid"):
        """
        分布数据到各个客户端

        Args:
            dataset: 完整数据集
            distribution_strategy: 分布策略 ('iid', 'non_iid')

        Returns:
            客户端数据加载器列表
        """
        total_size = len(dataset)

        if distribution_strategy == "iid":
            # IID分布 - 随机均匀分配
            indices = list(range(total_size))
            random.shuffle(indices)

            client_sizes = [total_size // self.num_clients] * self.num_clients
            # 处理余数
            for i in range(total_size % self.num_clients):
                client_sizes[i] += 1

            client_datasets = []
            start_idx = 0

            for size in client_sizes:
                end_idx = start_idx + size
                client_indices = indices[start_idx:end_idx]
                client_dataset = Subset(dataset, client_indices)
                client_datasets.append(client_dataset)
                start_idx = end_idx

        else:  # non_iid
            # Non-IID分布 - 按subset分配（每个客户端处理不同的subset）
            # 假设数据集已经按subset组织
            client_datasets = []
            subset_per_client = max(1, len(dataset.available_files) // self.num_clients)

            for i in range(self.num_clients):
                start_file = i * subset_per_client
                end_file = min(
                    (i + 1) * subset_per_client, len(dataset.available_files)
                )

                # 创建该客户端的文件列表
                client_files = dataset.available_files[start_file:end_file]

                # 过滤数据
                client_data = []
                for item in dataset.data:
                    if any(file in item["image"] for file in client_files):
                        client_data.append(item)

                # 创建客户端特定的数据集
                client_dataset = SimpleLUNA16Dataset(
                    data_dir=dataset.data_dir,
                    csv_path=dataset.csv_path,
                    patch_size=dataset.patch_size,
                    max_samples=len(client_data),
                    is_custom=True,
                )
                client_dataset.data = client_data
                client_datasets.append(client_dataset)

        # 创建数据加载器
        client_loaders = []
        for i, client_dataset in enumerate(client_datasets):
            loader = DataLoader(
                client_dataset, batch_size=1, shuffle=True, num_workers=0
            )
            client_loaders.append(loader)
            print(f"客户端 {i} 数据量: {len(client_dataset)}")

        return client_loaders

    def distribute_data_from_folders(
        self,
        client_data_dirs,
        csv_path,
        patch_size=(64, 64, 64),
        max_samples_per_client=None,
    ):
        """
        从指定的客户端文件夹分布数据到各个客户端

        Args:
            client_data_dirs: 客户端数据目录列表，例如 ["./client0", "./client1", "./client2"]
            csv_path: CSV注释文件路径
            patch_size: 数据块大小
            max_samples_per_client: 每个客户端的最大样本数量

        Returns:
            客户端数据加载器列表
        """
        if len(client_data_dirs) != self.num_clients:
            raise ValueError(
                f"客户端数据目录数量 ({len(client_data_dirs)}) 与客户端数量 ({self.num_clients}) 不匹配"
            )

        client_loaders = []

        for i, data_dir in enumerate(client_data_dirs):
            if not os.path.exists(data_dir):
                print(
                    f"警告: 客户端 {i} 的数据目录 {data_dir} 不存在，将创建空数据加载器"
                )
                # 创建空数据集
                empty_dataset = EmptyDataset(patch_size=patch_size)
                loader = DataLoader(
                    empty_dataset, batch_size=1, shuffle=False, num_workers=0
                )
                client_loaders.append(loader)
                print(f"客户端 {i} 数据量: 0 (数据目录不存在)")
                continue

            print(f"为客户端 {i} 创建数据集，使用数据目录: {data_dir}")

            try:
                # 为每个客户端创建独立的数据集
                # 使用当前目录作为单个subset
                subset_name = os.path.basename(data_dir)
                if subset_name.startswith("subset"):
                    # 如果是LUNA16的subset目录，使用标准方式
                    subset_folders = [subset_name]
                    parent_dir = os.path.dirname(data_dir)
                    is_custom = False
                    final_data_dir = parent_dir
                else:
                    # 如果是自定义目录，直接扫描该目录中的.mhd文件
                    subset_folders = None
                    is_custom = True
                    final_data_dir = data_dir

                client_dataset = SimpleLUNA16Dataset(
                    data_dir=final_data_dir,
                    csv_path=csv_path,
                    subset_folders=subset_folders,
                    patch_size=patch_size,
                    max_samples=max_samples_per_client,
                    is_custom=is_custom,
                )

                # 如果没有找到数据，创建一个空的数据集
                if len(client_dataset.image_files) == 0:
                    print(f"  警告: 在 {data_dir} 中没有找到有效的数据文件")
                    # 使用空数据集
                    empty_dataset = EmptyDataset(patch_size=patch_size)
                    loader = DataLoader(
                        empty_dataset, batch_size=1, shuffle=False, num_workers=0
                    )
                    client_loaders.append(loader)
                    print(f"客户端 {i} 数据量: 0 (来自 {data_dir})")
                else:
                    loader = DataLoader(
                        client_dataset, batch_size=1, shuffle=True, num_workers=0
                    )
                    client_loaders.append(loader)
                    print(
                        f"客户端 {i} 数据量: {len(client_dataset.image_files)} (来自 {data_dir})"
                    )

            except Exception as e:
                print(f"  错误: 为客户端 {i} 创建数据集时出错: {e}")
                # 创建空数据集作为fallback
                empty_dataset = EmptyDataset(patch_size=patch_size)
                loader = DataLoader(
                    empty_dataset, batch_size=1, shuffle=False, num_workers=0
                )
                client_loaders.append(loader)
                print(f"客户端 {i} 数据量: 0 (创建失败，使用空数据集)")

        return client_loaders

    def federated_training(
        self, train_loaders, test_loader=None, global_rounds=5, local_epochs=3
    ):
        """
        执行联邦学习训练

        Args:
            train_loaders: 客户端训练数据加载器列表
            test_loader: 测试数据加载器
            global_rounds: 全局训练轮数
            local_epochs: 本地训练轮数
        """

        print(f"开始联邦学习训练 - {global_rounds} 轮全局训练")

        add_server_log(
            f"联邦学习训练开始 - {global_rounds}轮全局训练, {local_epochs}轮本地训练, {len(train_loaders)}个客户端"
        )

        # 尝试获取Flask应用中的全局训练状态
        global_training_status = None
        try:
            # 尝试从全局变量获取训练状态
            import builtins

            if hasattr(builtins, "app_training_status"):
                global_training_status = builtins.app_training_status
                print(f"成功连接到Flask训练状态: {global_training_status}")
            else:
                # 尝试从各种可能的模块获取
                import sys

                for module_name in list(sys.modules.keys()):
                    if module_name in ["app", "__main__"]:
                        module = sys.modules[module_name]
                        if hasattr(module, "training_status"):
                            global_training_status = module.training_status
                            print(f"从模块 {module_name} 获取训练状态")
                            break
        except Exception as e:
            print(f"无法连接到Flask训练状态: {e}")
            pass

        for round_num in range(global_rounds):
            import sys  # 确保sys在作用域内可用

            print(f"\n=== 全局训练轮次 {round_num + 1}/{global_rounds} ===")

            add_server_log(f"全局训练轮次开始 - 第{round_num + 1}轮/{global_rounds}轮")

            # 更新全局训练状态（如果存在）
            if global_training_status:
                try:
                    global_training_status["current_round"] = round_num + 1
                    global_training_status["progress"] = int(
                        (round_num + 0.5) / global_rounds * 100
                    )
                    print(
                        f"更新训练状态: 轮次 {round_num + 1}/{global_rounds}, 进度 {global_training_status['progress']}%"
                    )
                    # 确保任何更改立即可见
                    import threading

                    threading.Event().wait(0.01)  # 微小的暂停，让更新在主线程可见
                except Exception as e:
                    print(f"更新训练状态失败: {e}")
                    pass

            # 1. 分发全局模型到所有客户端
            global_params = self.server.get_global_model_params()
            for client in self.clients:
                client.load_global_model(global_params)

            # 2. 各客户端执行本地训练
            client_params_list = []
            client_weights = []

            print(f"开始第 {round_num + 1} 轮客户端本地训练...")

            for i, (client, train_loader) in enumerate(
                zip(self.clients, train_loaders)
            ):
                if len(train_loader.dataset) == 0:
                    print(f"客户端 {i} 数据为空，跳过训练")
                    continue

                print(f"客户端 {i} 开始本地训练...")

                # 设置当前轮次信息
                client.current_round = round_num + 1

                # 本地训练
                client.local_epochs = local_epochs
                epoch_losses = client.local_train(train_loader, local_epochs)

                # 收集模型参数和权重
                client_params = client.get_model_params()
                client_weight = client.get_data_size(train_loader)

                client_params_list.append(client_params)
                client_weights.append(client_weight)

                print(f"客户端 {i} 本地训练完成，数据量: {client_weight}")

                # 广播客户端轮次完成消息
                broadcast_training_data(
                    i,
                    "round_complete",
                    {
                        "round": round_num + 1,
                        "average_loss": np.mean(epoch_losses) if epoch_losses else 0.0,
                        "data_size": client_weight,
                    },
                )

            # 3. 服务器执行模型聚合
            if client_params_list:
                try:
                    print(f"开始第 {round_num + 1} 轮模型聚合...")

                    self.server.federated_averaging(client_params_list, client_weights)

                    # 记录训练历史
                    client_losses = []
                    for client in self.clients:
                        if client.training_history:
                            recent_losses = client.training_history[-local_epochs:]
                            if recent_losses:
                                client_losses.extend(recent_losses)

                    avg_client_loss = np.mean(client_losses) if client_losses else 0.0

                    self.server.training_history["rounds"].append(round_num + 1)
                    self.server.training_history["average_loss"].append(avg_client_loss)

                    add_server_log(
                        f"第{round_num + 1}轮总结 - 平均客户端损失: {avg_client_loss:.4f}, 参与客户端: {len(client_params_list)}"
                    )

                    print(f"第 {round_num + 1} 轮平均客户端损失: {avg_client_loss:.4f}")

                    # 广播服务器端轮次完成消息
                    broadcast_server_training_data(
                        "round_complete",
                        {
                            "round": round_num + 1,
                            "total_rounds": global_rounds,
                            "average_loss": avg_client_loss,
                            "participating_clients": len(client_params_list),
                            "client_losses": [
                                (
                                    np.mean(client.training_history[-local_epochs:])
                                    if client.training_history
                                    else 0.0
                                )
                                for client in self.clients
                                if client.training_history
                            ],
                        },
                    )

                    # 4. 评估全局模型（可选，可能跳过以避免错误）
                    if test_loader is not None:
                        try:
                            print(f"评估第 {round_num + 1} 轮全局模型...")
                            global_loss = self.server.evaluate_global_model(test_loader)
                            self.server.training_history["average_loss"][
                                -1
                            ] = global_loss
                            print(f"全局模型评估损失: {global_loss:.4f}")
                        except Exception as e:
                            print(f"全局模型评估失败: {e}")
                            # 继续训练，不中断
                except Exception as e:
                    add_server_log(f"第{round_num + 1}轮出现错误: {str(e)}")
                    print(f"模型聚合失败: {e}")
                    break

            # 更新全局训练状态（如果存在）
            if global_training_status and round_num == global_rounds - 1:
                try:
                    global_training_status["current_round"] = global_rounds
                    global_training_status["progress"] = 100
                    global_training_status["end_time"] = (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if hasattr(datetime, "now")
                        else "训练完成"
                    )
                    print(f"训练完成，更新最终状态: {global_training_status}")
                    # 确保任何更改立即可见
                    import threading

                    threading.Event().wait(0.01)  # 微小的暂停，让更新在主线程可见
                except Exception as e:
                    print(f"更新最终训练状态失败: {e}")

        add_server_log(f"联邦学习训练完成 - 总共完成{global_rounds}轮训练")

        print("\n联邦学习训练完成！")
        return self.server.training_history

    def save_federated_model(self, save_path="federated_luna16_model.pth"):
        """保存联邦学习模型"""
        self.server.save_global_model(save_path)

    def plot_training_history(self):
        """绘制训练历史"""
        history = self.server.training_history

        if not history["rounds"]:
            print("没有训练历史可绘制")
            return

        try:
            plt.figure(figsize=(12, 4))

            # 损失曲线
            plt.subplot(1, 2, 1)
            plt.plot(
                history["rounds"], history["average_loss"], "b-o", label="平均损失"
            )
            plt.xlabel("全局训练轮次")
            plt.ylabel("损失")
            plt.title("联邦学习训练损失")
            plt.legend()
            plt.grid(True)

            # 客户端参与情况
            plt.subplot(1, 2, 2)
            client_participation = [len(self.clients)] * len(history["rounds"])
            plt.bar(
                history["rounds"], client_participation, alpha=0.7, label="参与客户端数"
            )
            plt.xlabel("全局训练轮次")
            plt.ylabel("客户端数量")
            plt.title("客户端参与情况")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()

            # 保存图片而不是显示
            plt.savefig("federated_training_history.png", dpi=150, bbox_inches="tight")
            plt.close()  # 关闭图形以释放内存
            print("训练历史图表已保存到: federated_training_history.png")
        except Exception as e:
            print(f"绘制训练历史时出错: {e}")
            print("跳过图表生成...")


def train_federated_model(
    num_clients=3, global_rounds=5, local_epochs=3, client_data_dirs=None
):
    """
    训练联邦学习模型的主函数

    Args:
        num_clients: 客户端数量
        global_rounds: 全局训练轮数
        local_epochs: 本地训练轮数
        client_data_dirs: 客户端数据目录列表，例如 ["./client0", "./client1", "./client2"]
                         如果为None，则使用原有的数据分布策略
    """
    import sys
    import io

    print("=== 联邦学习训练函数被调用 ===")
    print(
        f"参数: num_clients={num_clients}, global_rounds={global_rounds}, local_epochs={local_epochs}"
    )
    print(f"客户端数据目录: {client_data_dirs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建联邦学习协调器
    csv_path = "./src/annotations.csv"
    print("正在初始化联邦学习协调器...")

    coordinator = FederatedLearningCoordinator(
        num_clients=num_clients,
        model_class=Simple3DUNet,
        model_kwargs={"in_channels": 1, "out_channels": 2},
        device=device,
    )
    print("联邦学习协调器初始化完成")
    sys.stdout.flush()

    # 根据是否指定客户端数据目录来分配数据
    if client_data_dirs is not None:
        print("使用指定的客户端数据目录分布数据...")
        print(f"客户端数据目录: {client_data_dirs}")
        sys.stdout.flush()

        # 使用指定文件夹分布数据
        print("正在为各客户端加载数据...")
        sys.stdout.flush()

        client_loaders = coordinator.distribute_data_from_folders(
            client_data_dirs=client_data_dirs,
            csv_path=csv_path,
            patch_size=(64, 64, 64),
            max_samples_per_client=15,  # 每个客户端最大样本数
        )
        print(f"数据加载完成，共 {len(client_loaders)} 个客户端")
        sys.stdout.flush()
    else:
        print("使用默认数据分布策略...")
        print("警告: 未提供客户端数据目录，无法使用默认分布策略")
        sys.stdout.flush()
        client_loaders = []

    # 创建测试数据集 - 跳过测试阶段以避免错误
    print("跳过测试数据集创建...")
    test_loader = None
    sys.stdout.flush()

    # 执行联邦学习训练
    print("开始执行联邦学习训练...")
    sys.stdout.flush()

    training_history = coordinator.federated_training(
        train_loaders=client_loaders,
        test_loader=test_loader,
        global_rounds=global_rounds,
        local_epochs=local_epochs,
    )

    # 保存模型
    print("正在保存训练好的模型...")
    sys.stdout.flush()
    coordinator.save_federated_model("best_federated_lung_nodule_model.pth")

    # 绘制训练历史
    print("正在生成训练历史图表...")
    sys.stdout.flush()
    coordinator.plot_training_history()

    print(f"\n联邦学习训练完成！")
    print(f"客户端数量: {num_clients}")
    print(f"全局轮数: {global_rounds}")
    print(f"本地轮数: {local_epochs}")
    if client_data_dirs:
        print(f"使用的客户端数据目录: {client_data_dirs}")
    sys.stdout.flush()

    return coordinator


def run_federated_training(
    data_dir="testLUNA",
    num_clients=3,
    global_rounds=5,
    local_epochs=2,
    save_path="best_federated_lung_nodule_model.pth",
    client_data_directories=None,
):
    """
    联邦学习训练的包装函数，用于从Flask应用调用

    Args:
        data_dir: 数据目录（当client_data_directories为None时使用）
        num_clients: 客户端数量
        global_rounds: 全局训练轮数
        local_epochs: 本地训练轮数
        save_path: 模型保存路径
        client_data_directories: 客户端数据目录列表，如果提供则直接使用这些目录
    """
    # 如果直接提供了客户端数据目录，则使用它们
    if client_data_directories:
        print(f"使用提供的客户端数据目录: {client_data_directories}")
        client_data_dirs = client_data_directories
    else:
        # 检查数据目录中的客户端文件夹
        client_data_dirs = []
        if os.path.exists(data_dir):
            # 查找客户端文件夹
            potential_clients = []
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    # 检查是否包含.mhd文件
                    has_mhd = any(
                        f.endswith(".mhd")
                        for f in os.listdir(item_path)
                        if os.path.isfile(os.path.join(item_path, f))
                    )
                    if has_mhd:
                        potential_clients.append(item_path)

            if potential_clients:
                # 使用找到的客户端文件夹
                client_data_dirs = potential_clients[:num_clients]
                if len(client_data_dirs) < num_clients:
                    print(
                        f"警告: 只找到 {len(client_data_dirs)} 个客户端数据文件夹，但需要 {num_clients} 个"
                    )
                    # 补充空的客户端文件夹
                    while len(client_data_dirs) < num_clients:
                        client_data_dirs.append(None)

    # 调用原有的训练函数
    coordinator = train_federated_model(
        num_clients=len(client_data_dirs) if client_data_dirs else num_clients,
        global_rounds=global_rounds,
        local_epochs=local_epochs,
        client_data_dirs=client_data_dirs if client_data_dirs else None,
    )

    # 保存模型
    if coordinator and coordinator.global_model:
        torch.save(
            {
                "model_state_dict": coordinator.global_model.state_dict(),
                "round_num": global_rounds,
                "federated_training": True,
            },
            save_path,
        )
        print(f"联邦学习模型已保存到: {save_path}")

    return coordinator


if __name__ == "__main__":
    # 示例1: 使用默认数据分布策略（原有方式）
    # print("=== 示例1: 使用默认数据分布策略 ===")
    # coordinator = train_federated_model(num_clients=3, global_rounds=3, local_epochs=2)

    # 示例2: 使用指定的客户端数据文件夹
    print("=== 示例: 使用指定的客户端数据文件夹 ===")

    # 或者使用自定义的客户端文件夹
    client_data_directories = [
        "../uploads/client1_data",
        "../uploads/client2_data",  # 客户端1的专用数据文件夹
        "../uploads/client3_data",  # 客户端2的专用数据文件夹
    ]

    coordinator = train_federated_model(
        num_clients=3,
        global_rounds=3,
        local_epochs=2,
        client_data_dirs=client_data_directories,
    )
