# 联邦学习可视化系统 - 配置与运维文档

## 文档概述

本文档详细描述了联邦学习可视化系统的配置管理、版本控制、持续集成、部署和运维计划等关键内容，为系统的开发、测试、部署和维护提供完整的指导。

## 目录

1. [系统架构概述](#1-系统架构概述)
2. [配置管理](#2-配置管理)
3. [版本控制策略](#3-版本控制策略)
4. [持续集成/持续部署(CI/CD)](#4-持续集成持续部署cicd)
5. [环境配置](#5-环境配置)
6. [部署策略](#6-部署策略)
7. [监控与日志](#7-监控与日志)
8. [运维计划](#8-运维计划)


---

## 1. 系统架构概述

### 1.1 技术栈
- **后端框架**: Flask 3.0.3
- **实时通信**: Flask-SocketIO 5.5.1
- **机器学习**: PyTorch 2.6.0, TorchVision 0.21.0
- **医学图像处理**: SimpleITK 2.5.0
- **数据处理**: NumPy 1.26.4, Pandas 2.2.2
- **可视化**: Matplotlib 3.9.1
- **数据库**: Supabase (云端数据库)
- **Web服务器**: Gunicorn 23.0.0

### 1.2 系统组件
- **Web应用服务器**: Flask应用
- **联邦学习引擎**: PyTorch联邦学习实现
- **文件存储系统**: 本地文件系统 + 云端存储
- **用户管理系统**: 基于Supabase的身份认证
- **实时通信**: WebSocket连接

---

## 2. 配置管理

### 2.1 配置文件结构

```
code/
├── config/
│   ├── development.py      # 开发环境配置
│   ├── testing.py         # 测试环境配置
│   ├── production.py      # 生产环境配置
│   └── base.py           # 基础配置
├── .env                  # 环境变量文件
├── .env.example         # 环境变量模板
└── requirements.txt     # Python依赖
```

### 2.2 环境变量配置

创建 `.env` 文件包含以下配置：

```bash
# Flask配置
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
DEBUG=True

# Supabase配置
SUPABASE_URL=your-supabase-url
SUPABASE_ANON_KEY=your-supabase-anon-key

# 文件上传配置
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216  # 16MB

# 模型配置
MODEL_PATH=models/
CACHE_TIMEOUT=3600

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### 2.3 配置管理最佳实践

1. **敏感信息隔离**: 使用环境变量存储敏感配置
2. **环境区分**: 不同环境使用不同配置文件
3. **配置验证**: 启动时验证必需配置项
4. **默认值设置**: 为非关键配置提供合理默认值

---

## 3. 版本控制策略

### 3.1 Git工作流

采用 **Git Flow** 分支策略：

```
main                    # 生产分支
├── develop            # 开发主分支
├── feature/*          # 功能分支
├── release/*          # 发布分支
└── hotfix/*           # 热修复分支
```

### 3.2 分支管理规范

#### 3.2.1 分支命名规范
- `feature/FL-123-federated-training` - 功能开发
- `bugfix/FL-456-model-loading-error` - Bug修复
- `hotfix/FL-789-security-patch` - 紧急修复
- `release/v1.2.0` - 版本发布

#### 3.2.2 提交信息规范
```
<type>(<scope>): <subject>

<body>

<footer>
```

**类型(type)**:
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

**示例**:
```
feat(federated-learning): add model aggregation algorithm

- Implement FedAvg algorithm for model aggregation
- Add support for weighted averaging
- Include unit tests for aggregation functions

Closes #123
```

### 3.3 版本号规范

采用 **语义化版本控制 (SemVer)**:
- `MAJOR.MINOR.PATCH` (如: 1.2.3)
- `MAJOR`: 不兼容的API修改
- `MINOR`: 向后兼容的功能性新增
- `PATCH`: 向后兼容的问题修正

### 3.4 标签管理

```bash
# 创建标签
git tag -a v1.0.0 -m "Release version 1.0.0"

# 推送标签
git push origin v1.0.0

# 列出标签
git tag -l
```

---

## 4. 持续集成/持续部署(CI/CD)

### 4.1 GitHub Actions工作流

创建 `.github/workflows/ci-cd.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        cd code
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        cd code
        ./run_tests.sh all
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./code/coverage.xml
  
  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t fl-visualization:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        # 推送到Docker registry的步骤
        echo "Pushing to registry..."
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        # 部署到生产环境的步骤
        echo "Deploying to production..."
```

### 4.2 测试自动化

#### 4.2.1 测试层级
1. **单元测试**: 测试单个函数/方法
2. **集成测试**: 测试组件间交互
3. **系统测试**: 端到端功能测试
4. **性能测试**: 负载和压力测试

#### 4.2.2 测试配置
```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
addopts = 
    -v 
    --cov=src
    --cov=app
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80
```

### 4.3 代码质量检查

集成以下工具：
- **Pylint**: 代码静态分析
- **Black**: 代码格式化
- **isort**: 导入排序
- **mypy**: 类型检查

---

## 5. 环境配置

### 5.1 开发环境

#### 5.1.1 系统要求
- **操作系统**: macOS 10.15+, Ubuntu 18.04+, Windows 10+
- **Python**: 3.9+
- **内存**: 8GB RAM (推荐16GB)
- **存储**: 10GB可用空间
- **GPU**: NVIDIA GPU (可选，用于加速训练)

#### 5.1.2 环境搭建

```bash
# 1. 克隆项目
git clone https://github.com/your-org/FL-Visualization.git
cd FL-Visualization/code

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 5. 初始化数据库
python init_db.py

# 6. 运行应用
python app.py
```

### 5.2 测试环境

```bash
# 运行测试套件
./run_tests.sh all

# 运行特定类型测试
./run_tests.sh unit
./run_tests.sh integration
./run_tests.sh performance
```

### 5.3 生产环境

#### 5.3.1 服务器规格
- **CPU**: 4核心+
- **内存**: 16GB RAM+
- **存储**: 100GB+ SSD
- **网络**: 100Mbps+
- **操作系统**: Ubuntu 20.04 LTS

#### 5.3.2 依赖服务
- **Web服务器**: Nginx
- **应用服务器**: Gunicorn
- **数据库**: Supabase (云端)
- **缓存**: Redis (可选)
- **消息队列**: Celery + Redis (可选)

---

## 6. 部署策略

### 6.1 Docker化部署

#### 6.1.1 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

#### 6.1.2 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - ./uploads:/app/uploads
      - ./models:/app/models
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - web
```

### 6.2 部署流程

#### 6.2.1 蓝绿部署
1. **准备**: 准备新版本环境(绿环境)
2. **部署**: 在绿环境中部署新版本
3. **测试**: 在绿环境中进行测试
4. **切换**: 将流量从蓝环境切换到绿环境
5. **验证**: 验证新版本正常运行
6. **清理**: 保留旧环境一段时间后清理

#### 6.2.2 滚动更新
1. **逐步更新**: 逐个替换服务实例
2. **健康检查**: 确保新实例健康后再更新下一个
3. **回滚机制**: 发现问题时快速回滚

### 6.3 部署脚本

```bash
#!/bin/bash
# deploy.sh

set -e

echo "开始部署联邦学习可视化系统..."

# 1. 拉取最新代码
git pull origin main

# 2. 构建Docker镜像
docker build -t fl-visualization:latest .

# 3. 停止旧容器
docker-compose down

# 4. 启动新容器
docker-compose up -d

# 5. 健康检查
echo "等待服务启动..."
sleep 30

# 检查服务状态
if curl -f http://localhost:5000/health; then
    echo "部署成功！"
else
    echo "部署失败，正在回滚..."
    docker-compose down
    docker-compose up -d
    exit 1
fi

echo "部署完成！"
```

---

## 7. 监控与日志

### 7.1 应用监控

#### 7.1.1 健康检查端点

```python
@app.route('/health')
def health_check():
    """健康检查端点"""
    checks = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': app.config.get('VERSION', 'unknown'),
        'database': check_database_connection(),
        'disk_space': check_disk_space(),
        'memory_usage': check_memory_usage()
    }
    
    # 检查是否所有组件都健康
    if all(check['status'] == 'healthy' for check in checks.values() if isinstance(check, dict)):
        return jsonify(checks), 200
    else:
        return jsonify(checks), 503
```

#### 7.1.2 性能指标

监控以下关键指标：
- **响应时间**: API响应时间
- **错误率**: 4xx, 5xx错误百分比
- **吞吐量**: 每秒请求数(RPS)
- **资源使用**: CPU、内存、磁盘使用率
- **联邦学习指标**: 训练轮次、模型精度、参与客户端数

### 7.2 日志管理

#### 7.2.1 日志配置

```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(app):
    """配置日志系统"""
    if not app.debug:
        # 生产环境日志配置
        handler = RotatingFileHandler(
            'logs/app.log', 
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        handler.setLevel(logging.INFO)
        app.logger.addHandler(handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('应用启动')
```

#### 7.2.2 日志层级

- **ERROR**: 系统错误，需要立即关注
- **WARNING**: 警告信息，可能需要关注
- **INFO**: 一般信息，业务流程记录
- **DEBUG**: 调试信息，开发环境使用

#### 7.2.3 结构化日志

```python
import json
import logging

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log_federated_event(self, event_type, client_id, round_num, **kwargs):
        """记录联邦学习事件"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'client_id': client_id,
            'round_num': round_num,
            **kwargs
        }
        self.logger.info(json.dumps(log_data))
```

### 7.3 监控工具集成

推荐使用以下监控工具：

1. **Prometheus + Grafana**: 指标收集和可视化
2. **ELK Stack**: 日志聚合和分析
3. **Sentry**: 错误跟踪和性能监控
4. **New Relic/DataDog**: 应用性能监控(APM)

---

## 8. 运维计划

### 8.1 日常运维任务

#### 8.1.1 每日任务
- [ ] 检查系统健康状态
- [ ] 监控错误日志
- [ ] 验证备份完成
- [ ] 检查磁盘空间
- [ ] 监控性能指标

#### 8.1.2 每周任务
- [ ] 系统安全更新
- [ ] 日志归档和清理
- [ ] 性能报告生成
- [ ] 容量规划评估
- [ ] 备份策略验证

#### 8.1.3 每月任务
- [ ] 系统性能优化
- [ ] 安全漏洞扫描
- [ ] 灾难恢复演练
- [ ] 依赖项更新
- [ ] 运维文档更新

### 8.2 故障响应流程

#### 8.2.1 故障等级

**P0 - 紧急**: 系统完全不可用
- 响应时间: 15分钟内
- 解决时间: 2小时内

**P1 - 高**: 核心功能受影响
- 响应时间: 1小时内
- 解决时间: 8小时内

**P2 - 中**: 部分功能异常
- 响应时间: 4小时内
- 解决时间: 24小时内

**P3 - 低**: 一般性问题
- 响应时间: 24小时内
- 解决时间: 72小时内

#### 8.2.2 故障处理步骤

1. **故障识别**: 监控告警 → 故障确认
2. **影响评估**: 评估故障范围和影响
3. **初步响应**: 应急措施，减少影响
4. **根因分析**: 定位问题根本原因
5. **解决方案**: 实施修复方案
6. **验证测试**: 确认问题已解决
7. **事后总结**: 故障复盘和改进

### 8.3 变更管理

#### 8.3.1 变更类型

**紧急变更**: 生产故障修复
- 审批流程: 简化审批
- 测试要求: 最小化测试
- 回滚准备: 必须准备

**标准变更**: 常规功能更新
- 审批流程: 正常审批流程
- 测试要求: 完整测试
- 上线窗口: 维护窗口

**重大变更**: 架构性变更
- 审批流程: 多级审批
- 测试要求: 全面测试
- 风险评估: 详细评估

#### 8.3.2 变更流程

1. **变更申请**: 提交变更请求
2. **风险评估**: 评估变更风险
3. **审批流程**: 按级别审批
4. **实施计划**: 制定详细计划
5. **测试验证**: 在测试环境验证
6. **生产实施**: 按计划执行
7. **验证确认**: 确认变更成功
8. **文档更新**: 更新相关文档

---

## 总结

本配置与运维文档为联邦学习可视化系统提供了完整的配置管理、版本控制、持续集成、部署和运维指南。通过遵循本文档的最佳实践，可以确保系统的稳定性、安全性和可维护性。

### 关键要点

1. **配置管理**: 环境分离、敏感信息保护
2. **版本控制**: Git Flow工作流、语义化版本
3. **CI/CD**: 自动化测试和部署流程
4. **监控**: 全面的应用和系统监控
5. **安全**: 多层次安全防护措施
6. **运维**: 标准化的运维流程和应急响应

### 持续改进

1. 定期评估和更新运维流程
2. 根据业务增长调整系统架构
3. 持续优化监控和告警策略
4. 加强团队技能培训和知识共享