# FL-Visualization

[![Tests](https://github.com/YOUR_USERNAME/FL-Visualization/workflows/FL-Visualization%20Tests/badge.svg)](https://github.com/YOUR_USERNAME/FL-Visualization/actions)
[![Code Coverage](https://codecov.io/gh/YOUR_USERNAME/FL-Visualization/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/FL-Visualization)
[![Code Quality](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

一个基于Flask和Socket.IO的联邦学习可视化系统，用于肺结节检测的分布式训练和推理。

## 功能特性

- 🌐 **Web界面**: 基于Flask的现代化Web界面
- 🔄 **实时通信**: Socket.IO支持的实时状态更新
- 🤖 **联邦学习**: 支持多客户端协作训练
- 🏥 **医学图像**: 专为LUNA16肺结节数据集设计
- 📊 **可视化**: 训练过程和结果的实时可视化
- 🔐 **用户管理**: 基于角色的访问控制（客户端/服务器）

## 系统架构

- **Flask应用**: 主Web服务器 (`app.py`)
- **联邦学习模块**: 分布式训练实现 (`src/federated_training.py`)
- **推理引擎**: 模型推理服务 (`src/federated_inference_utils.py`)
- **前端界面**: 响应式Web界面 (`templates/`, `static/`)

## 快速开始

### 环境要求

- Python 3.9+
- PyTorch 2.0+
- Flask 2.3+
- 其他依赖见 `requirements.txt`

### 安装

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/FL-Visualization.git
cd FL-Visualization

# 安装依赖
pip install -r requirements.txt

# 或使用Makefile
make install
```

### 运行应用

```bash
# 启动Flask应用
python app.py

# 应用将在 http://localhost:5002 启动
```

### 使用流程

1. **登录系统**: 选择角色（客户端/服务器）
2. **客户端操作**: 上传医学图像数据
3. **服务器操作**: 配置并启动联邦学习训练
4. **监控训练**: 实时查看训练进度和日志
5. **模型推理**: 使用训练好的模型进行预测

## 开发

### 运行测试

```bash
# 运行所有测试
make test

# 运行单元测试
make test-unit

# 运行集成测试
make test-integration

# 运行测试并生成覆盖率报告
make coverage
```

### 代码质量

```bash
# 代码格式化
make format

# 代码检查
make lint
```

### 测试覆盖率

项目包含全面的测试套件：

- **单元测试**: Flask路由、联邦学习组件
- **集成测试**: 端到端工作流程测试
- **Socket.IO测试**: 实时通信功能测试

### CI/CD

项目使用GitHub Actions进行持续集成：

- **代码质量检查**: Black、flake8、isort、mypy
- **安全扫描**: Bandit安全检查
- **自动化测试**: 单元测试和集成测试
- **Docker构建**: 容器化部署测试
- **覆盖率报告**: Codecov集成

## 项目结构

```
FL-Visualization/
├── app.py                 # Flask主应用
├── requirements.txt       # Python依赖
├── pyproject.toml        # 项目配置
├── Makefile              # 开发工具命令
├── .github/
│   └── workflows/
│       └── test.yml      # GitHub Actions配置
├── src/                  # 核心模块
│   ├── federated_training.py      # 联邦学习训练
│   ├── federated_inference_utils.py # 推理工具
│   └── train_simple_model.py      # 基础模型定义
├── static/               # 静态资源
│   ├── css/             # 样式文件
│   └── js/              # JavaScript文件
├── templates/           # HTML模板
├── tests/               # 测试文件
├── uploads/             # 数据上传目录
└── dummy_data/          # 示例数据
```

## 联邦学习

### 训练流程

1. **数据分布**: 各客户端上传本地数据
2. **模型初始化**: 服务器初始化全局模型
3. **本地训练**: 客户端使用本地数据训练
4. **模型聚合**: 服务器聚合客户端模型参数
5. **迭代更新**: 重复训练和聚合过程

### 支持的算法

- **FedAvg**: 联邦平均算法
- **自定义聚合**: 支持加权平均和其他策略

## 技术栈

- **后端**: Flask, Flask-SocketIO, PyTorch
- **前端**: HTML5, CSS3, JavaScript, Socket.IO
- **数据处理**: NumPy, Pandas, SimpleITK
- **可视化**: Matplotlib
- **测试**: pytest, pytest-flask, pytest-socketio
- **代码质量**: Black, flake8, mypy, isort

## 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

### 开发规范

- 使用 Black 进行代码格式化
- 遵循 PEP 8 编码规范
- 为新功能添加测试
- 更新相关文档

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 支持

如有问题或建议，请：

1. 查看 [Issues](https://github.com/YOUR_USERNAME/FL-Visualization/issues)
2. 创建新的 Issue
3. 联系维护者

## 更新日志

### v1.0.0
- 初始版本发布
- 基础联邦学习功能
- Web界面和实时通信
- 完整的测试套件

---

*注意: 请将 `YOUR_USERNAME` 替换为您的实际GitHub用户名*
