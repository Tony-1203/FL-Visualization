# FL-Visualization 联邦学习可视化平台

![Python](https://img.shields.io/badge/Python-3.12.4-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.3-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📖 项目简介

FL-Visualization 是一个基于Web的联邦学习可视化平台，专注于医学影像（LUNA16肺结节检测）的分布式机器学习。该项目实现了完整的联邦学习工作流，包括客户端数据上传、模型训练、推理预测和实时监控。

### 🎯 主要特性

- **🔐 多用户系统** - 支持客户端和服务器角色，具备完整的用户认证机制
- **📁 医学影像处理** - 支持MHD/RAW格式的医学影像文件上传和处理
- **🤖 联邦学习** - 实现FedAvg算法的分布式模型训练
- **📊 实时监控** - WebSocket实时显示训练进度和状态
- **🔍 模型推理** - 支持训练后模型的推理预测功能
- **🗄️ 数据存储** - 支持Supabase云数据库和本地存储双重方案
- **🧪 完整测试** - 包含单元测试、集成测试和性能测试

## 🏗️ 系统架构

```
FL-Visualization/
├── 前端界面 (Flask Templates + SocketIO)
├── 后端服务 (Flask + WebSocket)
├── 联邦学习引擎 (PyTorch + FedAvg)
├── 数据存储 (Supabase + Local JSON)
└── 测试套件 (pytest + coverage)
```

### 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| **Web框架** | Flask | 3.0.3 |
| **实时通信** | Flask-SocketIO | 5.5.1 |
| **机器学习** | PyTorch | 2.6.0 |
| **医学影像** | SimpleITK | 2.5.0 |
| **数据分析** | NumPy, Pandas | 1.26.4, 2.2.2 |
| **数据库** | Supabase | 2.15.2 |
| **测试框架** | pytest | 7.4.3 |

## 🚀 快速开始

### 环境要求

- Python 3.12.4+
- 支持CUDA的GPU (推荐，用于模型训练)


### 1. 克隆项目

```bash
git clone git@github.com:Tony-1203/FL-Visualization.git
cd FL-Visualization/code
```

### 2. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 环境配置

创建 `.env` 文件（可选，用于Supabase集成）：

```bash
# Supabase配置 (可选)
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key

# Flask配置
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
```

### 4. 启动应用

```bash
python app.py
```

访问 `http://localhost:5000` 开始使用！

## 💡 使用指南

### 🔐 用户角色

#### 服务器管理员
- 查看所有客户端状态
- 启动联邦学习训练
- 监控训练进度
- 下载训练好的模型

#### 客户端用户
- 上传医学影像数据
- 参与联邦学习训练
- 查看个人训练历史
- 执行模型推理

### 📁 数据格式支持

- **MHD格式**: MetaImage头文件，包含影像元数据
- **RAW格式**: 对应的原始影像数据文件
- **文件对**: 每个影像需要对应的.mhd和.raw文件

#### 示例文件结构
```
uploads/
├── client1_data/
│   ├── scan001.mhd
│   ├── scan001.raw
│   ├── scan002.mhd
│   └── scan002.raw
└── client2_data/
    ├── lung_ct_001.mhd
    └── lung_ct_001.raw
```

### 🤖 联邦学习流程

1. **数据准备**: 客户端上传医学影像数据
2. **训练启动**: 服务器管理员启动联邦学习任务
3. **本地训练**: 各客户端在本地数据上训练模型
4. **模型聚合**: 服务器聚合客户端模型更新
5. **迭代优化**: 重复训练和聚合过程
6. **模型推理**: 使用训练好的模型进行预测


## 📊 功能特性

### 🖥️ Web界面

#### 登录页面
- 用户身份验证
- 角色选择（客户端/服务器）
- 自动会话管理

#### 客户端面板
- 数据上传界面
- 上传历史查看
- 训练进度监控
- 推理功能使用

#### 服务器面板
- 客户端状态总览
- 联邦学习控制
- 训练历史管理
- 系统日志查看

### 🔄 实时功能

- **WebSocket通信** - 实时状态更新
- **进度监控** - 训练过程实时显示
- **日志推送** - 系统消息实时通知
- **状态同步** - 多用户状态同步

### 🛡️ 安全特性

- **密码加密** - bcrypt哈希加密
- **会话管理** - 安全的用户会话
- **文件验证** - 上传文件格式验证
- **权限控制** - 基于角色的访问控制

## 🔧 配置说明


### 联邦学习参数

```python
# src/federated_training.py 中的训练参数
training_params = {
    "rounds": 10,              # 联邦学习轮数
    "epochs_per_round": 2,     # 每轮本地训练轮数
    "learning_rate": 0.001,    # 学习率
    "batch_size": 4,           # 批处理大小
    "num_workers": 2           # 数据加载器工作进程数
}
```

## 📱 API文档

### 文件上传API

```http
POST /client/upload
Content-Type: multipart/form-data

参数:
- files: 医学影像文件列表

响应:
{
    "message": "成功上传 N 个文件",
    "uploaded_files": ["file1.mhd", "file2.raw"],
    "total_files": 4,
    "mhd_count": 2,
    "raw_count": 2
}
```

### 训练启动API

```http
POST /server/start_training
Content-Type: application/json

{
    "rounds": 10,
    "epochs_per_round": 2,
    "learning_rate": 0.001
}
```

### 推理API

```http
POST /api/server/run_inference
Content-Type: application/json

{
    "model_session": 1,
    "image_file": "scan001.mhd"
}
```

## 🗂️ 项目结构

```
FL-Visualization/
├── code/                           # 主要代码目录
│   ├── app.py                     # Flask主应用
│   ├── requirements.txt           # Python依赖
│   ├── .env                       # 环境变量(需创建)
│   ├── local_users.json          # 本地用户数据
│   ├── training_history.json     # 训练历史记录
│   │
│   ├── src/                       # 联邦学习核心代码
│   │   ├── federated_training.py # 联邦学习训练逻辑
│   │   ├── federated_inference.py# 模型推理功能
│   │   ├── federated_inference_utils.py # 推理工具函数
│   │   ├── show_nodules.py       # 结节可视化
│   │   └── annotations.csv       # 标注数据
│   │
│   ├── templates/                 # HTML模板
│   │   ├── login.html            # 登录页面
│   │   ├── client_dashboard.html # 客户端面板
│   │   ├── server_dashboard.html # 服务器面板
│   │   ├── client_history.html   # 客户端历史
│   │   └── server_history.html   # 服务器历史
│   │
│   ├── static/                    # 静态资源
│   │   ├── css/                  # 样式文件
│   │   ├── js/                   # JavaScript文件
│   │   └── training_images/      # 训练图像
│   │
│   ├── tests/                     # 测试套件
│   │   ├── test_app.py           # 主应用测试
│   │   ├── test_federated_learning.py # 联邦学习测试
│   │   ├── test_performance_integration.py # 性能集成测试
│   │   ├── conftest.py           # pytest配置
│   │   └── README.md             # 测试文档
│   │
│   ├── uploads/                   # 文件上传目录
│   │   ├── client1_data/         # 客户端1数据
│   │   ├── client2_data/         # 客户端2数据
│   │   ├── server_inference/     # 服务器推理文件
│   │   └── client_inference/     # 客户端推理文件
│   │
│   ├── models/                    # 训练模型存储
│   │   ├── session_1_federated_model.pth
│   │   └── best_federated_lung_nodule_model.pth
│   │
│   ├── dummy_data/               # 示例数据
│   │   ├── *.mhd                # 示例MHD文件
│   │   └── *.raw                # 示例RAW文件
│   │
│   ├── pytest.ini               # pytest配置
│   ├── run_tests.py             # 完整测试脚本
│   ├── quick_test.py            # 快速测试脚本
│   ├── 测试套件总结.md           # 测试总结文档
│   └── 问题解决报告.md           # 问题解决记录
│
└── report/                       # 项目报告文档
    ├── 1. 源代码和相关文档/
    ├── 2. 需求分析报告/
    ├── 3. 系统建模报告/
    ├── 4. 架构设计文档/
    ├── 5. 工程化说明文档/
    ├── 6. 测试与质量报告/
    ├── 7. 配置与运维文档/
    ├── 8. 团队报告/
    ├── 9. 演示视频/
    └── 10. 个人心得/
```

## 🐛 常见问题

### Q: 启动时提示"ModuleNotFoundError"？
A: 确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

如果还有别的未安装的模块，请根据错误提示安装。

### Q: 无法连接Supabase数据库？
A: 检查`.env`文件配置，或删除配置使用本地存储模式。

### Q: 文件上传失败？
A: 检查文件格式（需要.mhd和.raw配对）和文件大小限制。

### Q: 训练过程中断？
A: 检查系统内存和GPU资源，可以减少batch_size参数。


## 🎯 未来计划

- [ ] 支持更多医学影像格式 (DICOM, NIfTI)
- [ ] 添加更多联邦学习算法 (FedProx, SCAFFOLD)
- [ ] 实现差分隐私保护
- [ ] 移动端适配
- [ ] Docker容器化部署
- [ ] 集群部署支持
- [ ] 更丰富的可视化图表
- [ ] 模型性能对比分析

