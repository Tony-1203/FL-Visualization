# FL-Visualization - 核心代码

这是FL-Visualization联邦学习可视化平台的主要代码目录。

## 🚀 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用
python app.py

# 3. 访问应用
# 打开浏览器访问 http://localhost:5000
```

## 📁 目录结构

```
code/
├── app.py                 # Flask主应用入口
├── requirements.txt       # Python依赖列表
├── .env.example          # 环境变量示例文件
│
├── src/                  # 联邦学习核心模块
│   ├── federated_training.py      # 联邦学习训练逻辑
│   ├── federated_inference.py     # 模型推理功能
│   └── federated_inference_utils.py # 推理工具函数
│
├── templates/            # HTML模板文件
│   ├── login.html       # 登录页面
│   ├── client_dashboard.html  # 客户端控制面板
│   └── server_dashboard.html  # 服务器控制面板
│
├── static/              # 静态资源文件
│   ├── css/            # 样式表
│   └── js/             # JavaScript文件
│
├── tests/              # 测试套件
│   ├── test_app.py     # 主应用测试
│   ├── test_federated_learning.py # 联邦学习测试
│   └── conftest.py     # pytest配置
│
├── uploads/            # 文件上传目录
├── models/             # 训练模型存储
└── dummy_data/         # 示例数据文件
```


## ⚙️ 配置

### 环境变量 (.env)

连接Supabase需要创建 `.env` 文件并配置以下变量：

```bash
# Supabase配置 (可选)
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key

# Flask配置
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
```

### 应用配置

主要配置项在 `app.py` 中：

- `SECRET_KEY`: Flask应用密钥
- `UPLOAD_FOLDER`: 文件上传目录
- `MAX_CONTENT_LENGTH`: 最大上传文件大小


## 📊 功能模块

### 1. 用户认证 (`app.py`)
- 用户登录/登出
- 会话管理
- 角色权限控制

### 2. 文件管理 (`app.py`)
- 医学影像文件上传
- 文件格式验证
- 多文件批量处理

### 3. 联邦学习 (`src/federated_training.py`)
- FedAvg算法实现
- 分布式模型训练
- 模型聚合与更新

### 4. 模型推理 (`src/federated_inference.py`)
- 训练模型加载
- 医学影像预测
- 结果可视化

### 5. 实时通信 (WebSocket)
- 训练进度实时更新
- 系统状态监控
- 多用户状态同步

## 🐛 常见问题

### 启动问题

**Q: 提示模块未找到？**
```bash
# 确保已安装依赖
pip install -r requirements.txt

# 检查Python环境
python --version  # 需要3.12+
```

**Q: 端口被占用？**
```bash
# 修改app.py中的端口号
if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5001)
```

### 测试问题

**Q: 测试失败？**
```bash
# 使用正确的测试命令
python -m pytest tests/ -v

# 而不是直接使用pytest
pytest tests/ -v  # 可能导致模块导入错误
```

## 📚 更多文档

- [完整项目README](../README.md) - 项目总体介绍
- [测试文档](tests/README.md) - 详细测试指南
- [API文档](../docs/API.md) - API接口说明

## 🤝 贡献

1. 遵循现有代码风格
2. 添加必要的测试
3. 更新相关文档
4. 提交前运行测试套件

---

更多信息请查看[主项目README](../README.md)
