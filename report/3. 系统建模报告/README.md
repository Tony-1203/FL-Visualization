# 系统建模报告

## 1. 概述

本文档旨在通过UML（统一建模语言）图表，从不同角度对"基于联邦学习的分布式医疗影像诊断系统"进行建模，以便更清晰、更直观地理解系统的结构、功能和行为。报告包含用例图、类图、序列图、状态图、活动图、组件图和部署图。

## 2. 用例图 (Use Case Diagram)

用例图展示了系统的主要功能和用户交互。采用水平布局，优化字体显示效果。

```mermaid
graph LR
    %% 参与者
    ClientUser["👨‍⚕️<br/>客户端用户<br/>医生/研究员"]
    ServerAdmin["🔧<br/>服务器管理员<br/>系统维护者"] 
    SystemAdmin["👑<br/>系统管理员<br/>高级权限"]

    %% 客户端功能
    subgraph "🏥 客户端功能"
        UC1["🔐<br/>用户登录验证"]
        UC2["📤<br/>上传训练数据"] 
        UC3["🎯<br/>参与联邦训练"]
        UC4["📊<br/>监控训练进度"]
        UC5["📈<br/>查看训练历史"]
    end

    %% 推理功能
    subgraph "🔍 推理功能"
        UC6["🩺<br/>肺结节推理诊断"]
        UC7["📋<br/>查看推理结果"]
        UC8["📊<br/>结果可视化"]
    end

    %% 服务器管理
    subgraph "🖥️ 服务器管理"
        UC9["🔄<br/>协调联邦训练"]
        UC10["📊<br/>聚合模型参数"]
        UC11["📡<br/>广播全局模型"]
        UC12["💾<br/>存储模型版本"]
    end

    %% 高级管理
    subgraph "⚙️ 高级管理"
        UC13["👥<br/>用户管理"]
        UC14["📊<br/>系统监控"] 
        UC15["🧠<br/>模型管理"]
        UC16["⚙️<br/>配置管理"]
    end

    %% 连接关系
    ClientUser --> UC1
    ClientUser --> UC2
    ClientUser --> UC3
    ClientUser --> UC4
    ClientUser --> UC5
    ClientUser --> UC6
    ClientUser --> UC7
    ClientUser --> UC8

    ServerAdmin --> UC9
    ServerAdmin --> UC10
    ServerAdmin --> UC11
    ServerAdmin --> UC12
    ServerAdmin --> UC14

    SystemAdmin --> UC13
    SystemAdmin --> UC14
    SystemAdmin --> UC15
    SystemAdmin --> UC16

    %% 依赖关系
    UC2 -.-> UC3
    UC3 -.-> UC4
    UC9 -.-> UC10
    UC10 -.-> UC11
```

### 用例说明：
- **客户端用户**: 代表使用系统的各个参与方（如医院、研究机构），可以上传训练数据、查看训练历史、执行推理任务。
- **服务器管理员**: 系统的核心管理者，负责启动和监控联邦学习过程、管理全局模型、执行服务器端推理。
- **系统管理员**: 具有最高权限的管理者，负责用户管理、系统监控、模型管理和配置管理。

## 3. 类图 (Class Diagram)

类图展示了系统的静态结构，包括主要的类、它们的属性、方法以及它们之间的关系。

```mermaid
classDiagram
    class FlaskApp {
        -secret_key: str
        -socketio: SocketIO
        +run()
        +route(rule, methods)
    }

    class User {
        -username: str
        -password_hash: str
        -role: str
        -email: str
        +authenticate_user(username, password)
        +hash_password(password)
        +verify_password(password, hashed)
    }

    class FederatedServer {
        -global_model: Simple3DUNet
        -round_num: int
        -training_history: dict
        +get_global_model_params()
        +federated_averaging(client_params_list, client_weights)
        +evaluate_global_model(test_loader)
        +save_global_model(save_path)
    }

    class FederatedClient {
        -client_id: int
        -model: Simple3DUNet
        -learning_rate: float
        -local_epochs: int
        +load_global_model(global_params)
        +local_train(train_loader, epochs)
        +get_model_params()
        +get_data_size(data_loader)
    }

    class FederatedLearningCoordinator {
        -num_clients: int
        -server: FederatedServer
        -clients: List[FederatedClient]
        +distribute_data(dataset, distribution_strategy)
        +distribute_data_from_folders(client_data_dirs, csv_path)
        +federated_training(train_loaders, test_loader, global_rounds)
        +plot_training_history(session_id)
    }

    class Simple3DUNet {
        -in_channels: int
        -out_channels: int
        +forward(x)
        +conv_block(in_channels, out_channels)
    }

    class SimpleLUNA16Dataset {
        -data_dir: str
        -csv_path: str
        -patch_size: tuple
        -data: List[dict]
        +__getitem__(idx)
        +__len__()
        +load_data()
    }

    class InferenceService {
        +run_inference(image_path, use_federated)
        +predict_with_federated_model(image_path)
        +visualize_federated_results(image, prob_map, nodules)
    }

    class DiceLoss {
        -smooth: float
        +__init__(smooth)
        +forward(pred, target): torch.Tensor
    }

    class FederatedLungNodulePredictor {
        -device: torch.device
        -model: Simple3DUNet
        +__init__(model_path, device)
        +load_federated_model(model_path): Simple3DUNet
        +predict(image_path): tuple
        +sliding_window_prediction(image): numpy.ndarray
        +detect_nodules(probability_map): list
        +visualize_results(image, prob_map, nodules)
    }

    class EmptyDataset {
        -patch_size: tuple
        +__init__(patch_size)
        +__len__(): int
        +__getitem__(idx): dict
    }

    class ClientTrainingChart {
        -client_id: str
        -training_data: list
        +update_chart_data(epoch, loss)
        +render_training_chart(): dict
        +export_chart_image(): str
    }

    class ServerTrainingVisualizer {
        -session_id: int
        -global_training_data: dict
        +aggregate_client_data(client_data_list)
        +plot_global_training_history(): str
        +generate_comparison_chart(): dict
        +save_training_images(session_id): str
    }

    class WebSocketHandler {
        -socketio: SocketIO
        -online_users: dict
        +handle_connect()
        +handle_disconnect()
        +join_training_room(client_id)
        +broadcast_training_update(data)
        +emit_server_status(status)
    }

    FlaskApp "1" -- "N" User : manages
    FlaskApp "1" -- "1" FederatedLearningCoordinator : controls
    FlaskApp "1" -- "1" WebSocketHandler : uses
    FederatedLearningCoordinator "1" -- "1" FederatedServer : contains
    FederatedLearningCoordinator "1" -- "N" FederatedClient : contains
    FederatedServer "1" -- "1" Simple3DUNet : uses
    FederatedClient "N" -- "1" Simple3DUNet : uses
    FederatedClient "N" -- "1" SimpleLUNA16Dataset : trains_on
    FederatedClient "N" -- "1" DiceLoss : uses
    FederatedServer "1" -- "1" DiceLoss : uses
    FlaskApp "1" -- "1" InferenceService : provides
    InferenceService "1" -- "1" FederatedLungNodulePredictor : uses
    FederatedLungNodulePredictor "1" -- "1" Simple3DUNet : uses
    FederatedClient "N" -- "1" EmptyDataset : fallback_to
    FederatedClient "1" -- "1" ClientTrainingChart : visualizes_with
    FederatedLearningCoordinator "1" -- "1" ServerTrainingVisualizer : uses
    WebSocketHandler "1" -- "1" ClientTrainingChart : broadcasts_to
    WebSocketHandler "1" -- "1" ServerTrainingVisualizer : broadcasts_to
```

- **FlaskApp**: 基于Flask的Web应用，集成了SocketIO进行实时通信，是系统的核心入口。
- **User**: 用户实体，支持基于角色的访问控制，包括客户端用户和服务器管理员。
- **FederatedServer**: 联邦学习服务器，负责全局模型管理、模型聚合和训练协调。
- **FederatedClient**: 联邦学习客户端，负责本地模型训练和参数上传。
- **FederatedLearningCoordinator**: 联邦学习协调器，管理整个联邦学习过程，包括数据分发和训练流程。
- **Simple3DUNet**: 3D UNet神经网络模型，用于肺结节检测的深度学习模型。
- **SimpleLUNA16Dataset**: LUNA16数据集的处理类，负责数据加载和预处理。
- **InferenceService**: 推理服务，提供模型推理和结果可视化功能。
- **DiceLoss**: Dice损失函数，专门用于医学图像分割任务的损失计算。
- **FederatedLungNodulePredictor**: 联邦学习肺结节预测器，负责加载联邦模型并进行推理预测。
- **EmptyDataset**: 空数据集，在联邦学习中处理没有数据的客户端情况。
- **ClientTrainingChart**: 客户端训练图表生成器，负责客户端训练过程的可视化。
- **ServerTrainingVisualizer**: 服务器训练可视化器，负责全局训练过程的图表生成和数据聚合。
- **WebSocketHandler**: WebSocket事件处理器，负责实时通信、用户连接管理和训练数据广播。

## 4. 序列图 (Sequence Diagram)

序列图展示了对象之间交互的时间顺序。下面是“联邦学习一轮训练”的核心交互过程。

```mermaid
sequenceDiagram
    participant User as 👤 用户
    participant Browser as 🌐 浏览器
    participant FlaskApp as 🖥️ Flask应用
    participant WebSocket as 📡 WebSocket处理器
    participant Coordinator as 🔄 联邦学习协调器
    participant Server as 🏢 联邦服务器
    participant Client as 🏥 联邦客户端

    note over User,Client: 📋 第一阶段：用户登录和连接建立
    User->>+Browser: 登录并访问仪表盘
    Browser->>+FlaskApp: HTTP请求 /server/dashboard
    FlaskApp->>-Browser: 返回仪表盘页面
    Browser->>+WebSocket: 建立WebSocket连接
    WebSocket->>-Browser: 连接确认

    note over User,Client: 📋 第二阶段：启动联邦训练
    User->>+Browser: 点击启动联邦训练
    Browser->>+FlaskApp: POST /server/start_training
    FlaskApp->>+Coordinator: 创建训练会话
    
    note over Coordinator,Client: 📋 第三阶段：初始化训练环境
    Coordinator->>+Server: 初始化全局模型
    Coordinator->>+Client: 创建客户端实例
    
    note over User,Client: 📋 第四阶段：联邦训练循环
    loop 全局训练轮次 (Round 1-N)
        Server->>+Client: 分发全局模型参数
        note over Client: 本地数据训练
        Client->>Client: 执行本地训练
        
        note over Client,Browser: 实时进度更新
        Client->>WebSocket: 广播训练进度
        WebSocket->>Browser: 实时更新训练状态
        Browser->>User: 显示训练图表
        
        Client->>-Server: 返回本地模型参数
        note over Server: 联邦平均聚合
        Server->>Server: 执行联邦平均算法
        
        note over Server,Browser: 聚合结果广播
        Server->>WebSocket: 广播聚合结果
        WebSocket->>Browser: 更新全局训练状态
        Browser->>User: 显示聚合进度图表
    end
    
    note over User,Client: 📋 第五阶段：训练完成和结果保存
    Server->>Coordinator: 保存最终模型
    Coordinator->>-FlaskApp: 返回训练结果
    FlaskApp->>-Browser: 训练完成响应
    
    note over User,Client: 📋 第六阶段：结果展示
    FlaskApp->>WebSocket: 广播训练完成
    WebSocket->>Browser: 训练完成通知
    Browser->>-User: 显示最终结果
```

该图展示了从用户请求参与训练开始，到服务器分发模型、客户端本地训练、服务器聚合模型，最后将状态返回给用户的完整流程。

## 5. 状态图 (State Diagram)

状态图描述了一个对象在其生命周期内的各种状态以及状态之间的转换。下面是 `TrainingSession`（训练会话）对象的状态图。

```mermaid
stateDiagram-v2
    [*] --> Idle : 创建训练会话
    Idle --> Initializing : 服务器启动训练
    Initializing --> Training : 客户端加载完成
    Training --> Aggregating : 客户端完成本轮训练
    Aggregating --> Training : 模型聚合完成，开始新轮次
    Aggregating --> Evaluating : 达到评估条件
    Evaluating --> Training : 评估完成，继续训练
    Evaluating --> Completed : 达到收敛条件
    Training --> Completed : 达到最大轮次
    Training --> Failed : 训练过程出错
    Aggregating --> Failed : 聚合过程出错
    Completed --> [*] : 保存模型，会话结束
    Failed --> [*] : 清理资源，会话结束
```

- **Idle (空闲)**: 训练会话已创建但尚未开始，等待服务器启动。
- **Initializing (初始化)**: 服务器正在初始化全局模型和客户端，准备开始训练。
- **Training (训练中)**: 客户端正在进行本地训练，服务器等待客户端返回模型参数。
- **Aggregating (聚合中)**: 服务器正在收集并聚合来自客户端的本地模型参数，更新全局模型。
- **Evaluating (评估中)**: 服务器正在评估当前全局模型的性能。
- **Completed (已完成)**: 训练过程正常完成，达到收敛条件或最大轮次。
- **Failed (失败)**: 训练过程中出现错误，需要终止会话。

## 6. 活动图 (Activity Diagram)

活动图描述了系统特定业务流程中的工作流或活动顺序。下图展示了“联邦学习单轮训练”的详细活动流程。

```mermaid
graph TD
    subgraph InitPhase["═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"]
        A[🚀 开始联邦训练] --> B{📋 检查训练条件}
        B -- 满足条件 --> C[📡 服务器分发全局模型]
        B -- 不满足条件 --> EndNeg[❌ 训练终止]
    end

    subgraph TrainingPhase["═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"]
        C --> D{🏥 客户端并行训练}
        
        subgraph ClientsParallel["客户端并行处理"]
            direction TB
            D1[🏥 客户端1<br/>接收模型] --> E1[🎯 本地数据训练] --> F1[📤 返回本地模型]
            D2[🏥 客户端2<br/>接收模型] --> E2[🎯 本地数据训练] --> F2[📤 返回本地模型]
            Dn[🏥 客户端N<br/>接收模型] --> En[🎯 本地数据训练] --> Fn[📤 返回本地模型]
        end
        
        D --> G[⏳ 服务器等待收集所有模型]
    end

    subgraph AggregationPhase["═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"]
        G --> H[🔄 联邦平均聚合更新全局模型]
        H --> I{🎯 是否达到终止条件?}
        I -- 未达到 --> C
        I -- 已达到 --> J[✅ 训练完成]
    end

    subgraph EndPhase["═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"]
        J --> K[💾 保存最终模型]
        EndNeg --> K
        K --> L[🏁 结束]
    end

    %% 样式设置
    style InitPhase fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    style TrainingPhase fill:#e8f5e8,stroke:#388e3c,stroke-width:3px
    style AggregationPhase fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    style EndPhase fill:#fce4ec,stroke:#c2185b,stroke-width:3px
    style ClientsParallel fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style A fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    style L fill:#ffcdd2,stroke:#d32f2f,stroke-width:3px
```

此图清晰地展示了从训练开始到结束，服务器和客户端之间的交互、并行处理以及决策逻辑。

## 7. 组件图 (Component Diagram)

组件图展示了系统的模块化结构和组件之间的依赖关系。

```mermaid
graph TD
    subgraph "Web应用层"
        A["Flask Web Server"]
        B["SocketIO 实时通信"]
        C["静态资源服务"]
    end

    subgraph "业务逻辑层"
        D["用户认证模块"]
        E["联邦学习协调器"]
        F["推理服务模块"]
        G["文件管理模块"]
        H["WebSocket处理器"]
    end

    subgraph "数据处理层"
        I["联邦服务器"]
        J["联邦客户端"]
        K["3D UNet模型"]
        L["LUNA16数据集处理"]
        M["损失函数模块"]
    end

    subgraph "前端组件层"
        N["客户端图表组件"]
        O["服务器图表组件"]
        P["实时通信组件"]
        Q["文件上传组件"]
    end

    subgraph "存储层"
        R["用户数据存储"]
        S["模型存储"]
        T["训练历史存储"]
        U["推理结果存储"]
    end

    A --> D
    A --> E
    A --> F
    A --> G
    B --> H
    B --> P
    
    D --> R
    E --> I
    E --> J
    F --> K
    G --> U
    H --> N
    H --> O
    
    I --> K
    J --> K
    J --> L
    J --> M
    I --> S
    E --> T

    P --> H
    N --> B
    O --> B
    Q --> G

    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#e3f2fd
    style D fill:#f3e5f5
    style E fill:#f3e5f5
    style F fill:#f3e5f5
    style G fill:#f3e5f5
    style H fill:#f3e5f5
    style I fill:#e8f5e8
    style J fill:#e8f5e8
    style K fill:#e8f5e8
    style L fill:#e8f5e8
    style M fill:#e8f5e8
    style N fill:#ffecb3
    style O fill:#ffecb3
    style P fill:#ffecb3
    style Q fill:#ffecb3
    style R fill:#fff3e0
    style S fill:#fff3e0
    style T fill:#fff3e0
    style U fill:#fff3e0
```

- **Web应用层**: 提供HTTP服务和WebSocket实时通信，处理用户请求和静态资源服务。
- **业务逻辑层**: 包含核心业务逻辑，如用户认证、联邦学习协调、推理服务、文件管理和WebSocket处理。
- **数据处理层**: 负责具体的数据处理任务，包括联邦学习算法、深度学习模型、数据集处理和损失函数计算。
- **前端组件层**: 包含前端JavaScript组件，负责实时图表展示、用户交互和文件上传等功能。
- **存储层**: 管理各类数据的持久化存储，包括用户数据、模型文件、训练历史和推理结果。

## 8. 部署图 (Deployment Diagram)

部署图描述了系统硬件和软件的物理部署结构。为了提高可读性，采用垂直分层布局。

```mermaid
graph TB
    subgraph ServerTier["═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"]
        subgraph ServerNode["🖥️ 中心服务器 (云端/本地)"]
            direction TB
            WebApp["🌐 Flask Web应用<br/>端口: 5000"]
            Socket["📡 SocketIO服务<br/>实时通信"]
            Coordinator["🔄 联邦学习协调器<br/>模型聚合"]
            
            subgraph StorageSystem["💾 存储系统"]
                direction LR
                ModelStore["📦 模型存储<br/>.pth文件"]
                UserStore["👥 用户数据存储<br/>local_users.json"]
                HistoryStore["📊 训练历史存储<br/>training_history.json"]
            end
        end
    end

    subgraph NetworkTier["═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"]
        NetworkCloud["☁️ 网络层<br/>HTTP/HTTPS + WebSocket"]
    end

    subgraph ClientTier["═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"]
        subgraph ClientGroup["🏥 客户端机构群组"]
            direction TB
            
            subgraph Hospital1["🏥 医院A"]
                direction TB
                Browser1["🌐 Web浏览器<br/>客户端界面"]
                Data1["📂 本地训练数据<br/>CT影像数据"]
            end

            subgraph Hospital2["🏥 医院B"]
                direction TB
                Browser2["🌐 Web浏览器<br/>客户端界面"]
                Data2["📂 本地训练数据<br/>CT影像数据"]
            end

            subgraph Research["🔬 研究机构"]
                direction TB
                Browser3["🌐 Web浏览器<br/>管理员界面"]
                Data3["📂 本地训练数据<br/>CT影像数据"]
            end
        end
    end

    %% 服务器内部连接
    WebApp ==> Socket
    WebApp ==> Coordinator
    Coordinator ==> ModelStore
    WebApp ==> UserStore
    Coordinator ==> HistoryStore

    %% 网络层连接
    WebApp -.-> NetworkCloud
    Socket -.-> NetworkCloud

    %% 客户端连接
    NetworkCloud -.-> Browser1
    NetworkCloud -.-> Browser2
    NetworkCloud -.-> Browser3

    %% 数据流连接
    Browser1 -.-> Data1
    Browser2 -.-> Data2
    Browser3 -.-> Data3

    %% 样式设置
    style ServerTier fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    style NetworkTier fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    style ClientTier fill:#e8f5e8,stroke:#388e3c,stroke-width:3px
    style ServerNode fill:#bbdefb,stroke:#1565c0,stroke-width:2px
    style StorageSystem fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style ClientGroup fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style Hospital1 fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    style Hospital2 fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    style Research fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
```

该图显示了基于Web的联邦学习系统的实际部署架构，中心服务器集成了Flask Web应用、SocketIO实时通信和联邦学习协调器，多个客户端机构通过Web浏览器连接到中心服务器，实现基于HTTP/HTTPS和WebSocket的分布式训练。

---


