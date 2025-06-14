/**
 * 客户端训练损失实时可视化组件
 * 显示服务器端聚合的联邦学习训练数据
 */

class ClientTrainingChart {
    constructor(containerId, clientId) {
        console.log('🏗️ 开始构造ClientTrainingChart实例...');
        console.log(`📝 参数: containerId=${containerId}, clientId=${clientId}`);
        
        this.containerId = containerId;
        this.clientId = clientId;
        this.chart = null;
        this.isTraining = false;
        this.currentRound = 0;
        this.socket = null;
        this.trainingData = {
            server: {
                rounds: [],
                aggregatedLoss: [],
                clientsCount: []
            },
            clients: {}  // {client_id: {rounds: [], losses: []}}
        };
        this.metricsData = {
            currentLoss: 0,
            avgLoss: 0,
            minLoss: Infinity,
            totalRounds: 0,
            activeClients: 0,
            lastUpdateTime: null
        };
        
        console.log('📊 初始化数据结构完成');
        console.log('🚀 开始调用init()方法...');
        this.init();
        console.log('🔌 开始调用connectWebSocket()方法...');
        this.connectWebSocket();
        console.log('✅ ClientTrainingChart构造完成');
    }

    init() {
        console.log('🎯 初始化ClientTrainingChart...');
        this.createContainer();
        this.createChart();
        this.startPeriodicUpdate();
        console.log('✅ ClientTrainingChart初始化完成');
    }

    createContainer() {
        console.log(`🏗️ 创建容器: ${this.containerId}`);
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`❌ 找不到容器: ${this.containerId}`);
            return;
        }
        console.log('✅ 容器元素找到，开始创建HTML结构...');

        container.innerHTML = `
            <div class="training-loss-container">
                <div class="training-loss-header">
                    <div class="training-loss-title">
                        <i class="fas fa-chart-line"></i>
                        联邦学习训练监控
                    </div>
                    <div class="training-status waiting" id="trainingStatus">
                        <div class="status-dot"></div>
                        <span>等待训练</span>
                    </div>
                </div>

                <div class="training-info">
                    <h4>联邦学习全局状态</h4>
                    <p>当前客户端: <strong>${this.clientId}</strong></p>
                    <p>训练轮次: <span id="currentRoundDisplay">0</span></p>
                    <p>参与客户端: <span id="activeClientsDisplay">0</span></p>
                    <p>训练状态: <span id="trainingStatusText">等待开始</span></p>
                </div>

                <div class="chart-container">
                    <canvas id="lossChart" class="chart-canvas"></canvas>
                </div>

                <div class="training-metrics">
                    <div class="metric-card">
                        <div class="metric-label">当前聚合损失</div>
                        <div class="metric-value" id="currentLossValue">--</div>
                        <div class="metric-change neutral" id="currentLossChange">
                            <i class="fas fa-server"></i>
                            <span>服务器聚合</span>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">平均聚合损失</div>
                        <div class="metric-value" id="avgLossValue">--</div>
                        <div class="metric-change neutral" id="avgLossChange">
                            <i class="fas fa-chart-line"></i>
                            <span>全局平均</span>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">最佳聚合损失</div>
                        <div class="metric-value" id="minLossValue">--</div>
                        <div class="metric-change neutral" id="minLossChange">
                            <i class="fas fa-trophy"></i>
                            <span>最佳记录</span>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">训练轮数</div>
                        <div class="metric-value" id="totalRoundsValue">0</div>
                        <div class="metric-change neutral" id="totalRoundsChange">
                            <i class="fas fa-clock"></i>
                            <span>累计轮次</span>
                        </div>
                    </div>
                </div>

                <div class="training-controls">
                    <button class="control-btn secondary" onclick="clientTrainingChart.clearData()">
                        <i class="fas fa-trash"></i>
                        清空数据
                    </button>
                    <button class="control-btn secondary" onclick="clientTrainingChart.exportData()">
                        <i class="fas fa-download"></i>
                        导出数据
                    </button>
                    <button class="control-btn primary" onclick="clientTrainingChart.refreshChart()">
                        <i class="fas fa-sync-alt"></i>
                        刷新图表
                    </button>
                </div>

                <div class="no-data-message" id="noDataMessage" style="display: none;">
                    <i class="fas fa-chart-line"></i>
                    <h3>暂无训练数据</h3>
                    <p>开始联邦学习训练后，聚合损失曲线将在此显示</p>
                </div>
            </div>
        `;
        console.log('✅ 容器HTML结构创建完成');
    }

    createChart() {
        console.log('📊 开始创建图表...');
        const ctx = document.getElementById('lossChart');
        if (!ctx) {
            console.error('❌ 找不到图表画布 #lossChart');
            return;
        }
        console.log('✅ 图表画布元素找到');

        console.log('🎨 检查Chart.js库:', typeof Chart !== 'undefined' ? '✅ 已加载' : '❌ 未加载');
        if (typeof Chart === 'undefined') {
            console.error('❌ Chart.js库未加载，无法创建图表');
            return;
        }

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '联邦学习聚合损失',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#667eea',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1000,
                    easing: 'easeInOutQuart'
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    title: {
                        display: true,
                        text: '联邦学习聚合损失趋势',
                        font: { size: 16, weight: 'bold' },
                        color: '#ffffff'
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#ffffff',
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(17, 24, 39, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#667eea',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true,
                        callbacks: {
                            title: function(context) {
                                return `轮次 ${context[0].label}`;
                            },
                            label: function(context) {
                                return `聚合损失: ${context.parsed.y.toFixed(6)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: '训练轮次',
                            color: '#ffffff',
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        },
                        ticks: {
                            color: '#a1a1aa',
                            font: {
                                size: 11
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            drawBorder: false
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: '聚合损失值',
                            color: '#ffffff',
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        },
                        ticks: {
                            color: '#a1a1aa',
                            font: {
                                size: 11
                            },
                            callback: function(value) {
                                return value.toFixed(4);
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            drawBorder: false
                        }
                    }
                }
            }
        });
        console.log('✅ 客户端训练图表创建成功');
    }

    connectWebSocket() {
        // 使用Socket.IO连接接收实时训练数据
        console.log('🔌 开始初始化WebSocket连接...');
        try {
            if (typeof io !== 'undefined') {
                console.log('✅ Socket.IO库已加载');
                this.socket = io();
                
                this.socket.on('connect', () => {
                    console.log('✅ 客户端训练可视化WebSocket已连接');
                });

                // 监听服务器训练更新 - 这是主要的数据源
                this.socket.on('server_training_update', (data) => {
                    console.log('📊 收到server_training_update事件:', data);
                    this.handleServerTrainingUpdate(data);
                });

                // 监听客户端训练更新
                this.socket.on('client_training_update', (data) => {
                    console.log('📱 收到client_training_update事件:', data);
                    this.handleClientTrainingUpdate(data);
                });

                this.socket.on('disconnect', () => {
                    console.log('❌ 客户端训练可视化WebSocket连接断开');
                });

                // 加入训练监控房间
                this.socket.emit('join_training_room', { role: 'client', client_id: this.clientId });

                this.socket.on('training_room_joined', (data) => {
                    console.log('✅ 成功加入训练监控房间:', data);
                });

                console.log('🔌 客户端训练可视化WebSocket已连接');
            } else {
                console.error('❌ Socket.IO库未加载，无法初始化WebSocket连接');
                this.updateStatus('waiting', 'Socket.IO未加载');
            }
        } catch (error) {
            console.error('❌ Socket.IO连接失败:', error);
            this.updateStatus('waiting', '连接失败');
        }
    }

    /**
     * 处理服务器训练数据更新
     */
    handleServerTrainingUpdate(data) {
        console.log('📊 收到服务器训练数据:', data);
        
        switch (data.type) {
            case 'round_complete':
                this.handleRoundComplete(data);
                break;
            case 'training_start':
                this.handleTrainingStart(data);
                break;
            case 'training_complete':
                this.handleTrainingComplete(data);
                break;
        }
    }

    /**
     * 处理客户端训练数据更新
     */
    handleClientTrainingUpdate(data) {
        if (!data.client_id) return;
        
        console.log(`📱 收到客户端${data.client_id}训练数据:`, data);
        
        // 初始化客户端数据
        if (!this.trainingData.clients[data.client_id]) {
            this.trainingData.clients[data.client_id] = {
                rounds: [],
                losses: [],
                status: 'waiting',
                currentRound: 0,
                averageLoss: 0
            };
        }
        
        const clientData = this.trainingData.clients[data.client_id];
        
        switch (data.type) {
            case 'training_start':
                clientData.status = 'training';
                break;
                
            case 'training_progress':
                if (data.current_loss !== undefined) {
                    clientData.currentLoss = data.current_loss;
                }
                break;
                
            case 'round_complete':
                if (data.round !== undefined && data.average_loss !== undefined) {
                    const existingIndex = clientData.rounds.indexOf(data.round);
                    if (existingIndex === -1) {
                        clientData.rounds.push(data.round);
                        clientData.losses.push(data.average_loss);
                    } else {
                        clientData.losses[existingIndex] = data.average_loss;
                    }
                    
                    clientData.currentRound = data.round;
                    clientData.averageLoss = data.average_loss;
                }
                break;
                
            case 'training_complete':
                clientData.status = 'completed';
                break;
        }
    }

    handleWebSocketMessage(data) {
        console.log(`📬 处理WebSocket消息:`, data);
        console.log(`🔍 消息类型: ${data.type}`);
        
        switch (data.type) {
            case 'training_start':
                console.log('🚀 处理训练开始消息');
                this.handleTrainingStart(data);
                break;
            case 'training_progress':
                console.log('📈 处理训练进度消息');
                this.handleTrainingProgress(data);
                break;
            case 'training_complete':
                console.log('✅ 处理训练完成消息');
                this.handleTrainingComplete(data);
                break;
            case 'round_complete':
                console.log('🔄 处理轮次完成消息');
                this.handleRoundComplete(data);
                break;
            default:
                console.log('❓ 未知消息类型:', data.type);
        }
    }

    handleTrainingStart(data) {
        this.isTraining = true;
        this.currentRound = data.round || 1;
        this.updateStatus('training', '训练中...');
        this.updateCurrentRound(this.currentRound);
        
        // 隐藏无数据消息
        const noDataMsg = document.getElementById('noDataMessage');
        if (noDataMsg) noDataMsg.style.display = 'none';
    }

    handleTrainingProgress(data) {
        // 客户端不再处理本地训练进度，只监听服务器聚合数据
        console.log('📈 收到训练进度数据（客户端忽略）:', data);
    }

    handleTrainingComplete(data) {
        this.isTraining = false;
        this.updateStatus('completed', '训练完成');
        console.log('✅ 联邦学习训练完成');
    }

    handleRoundComplete(data) {
        console.log('🔄 处理轮次完成数据:', data);
        console.log('📊 检查字段:', {
            round: data.round,
            average_loss: data.average_loss,
            participating_clients: data.participating_clients
        });
        
        if (data.round !== undefined && data.average_loss !== undefined) {
            console.log('✅ 数据有效，更新客户端图表数据');
            
            // 检查是否已存在该轮次的数据，避免重复
            const existingIndex = this.trainingData.server.rounds.indexOf(data.round);
            if (existingIndex === -1) {
                // 新轮次数据，直接添加
                this.trainingData.server.rounds.push(data.round);
                this.trainingData.server.aggregatedLoss.push(data.average_loss);
                this.trainingData.server.clientsCount.push(data.participating_clients || 0);
            } else {
                // 更新现有轮次数据
                console.log(`🔄 更新轮次 ${data.round} 的数据`);
                this.trainingData.server.aggregatedLoss[existingIndex] = data.average_loss;
                this.trainingData.server.clientsCount[existingIndex] = data.participating_clients || 0;
            }
            
            console.log('📈 当前客户端聚合数据:', {
                rounds: this.trainingData.server.rounds,
                losses: this.trainingData.server.aggregatedLoss,
                clientsCount: this.trainingData.server.clientsCount
            });
            
            this.currentRound = data.round;
            this.updateChart();
            this.updateMetricsDisplay();
            
            // 隐藏无数据消息
            const noDataMsg = document.getElementById('noDataMessage');
            if (noDataMsg) noDataMsg.style.display = 'none';
        } else {
            console.warn('⚠️ 轮次完成数据缺少必要字段:', data);
        }
    }

    updateChart() {
        console.log('📈 尝试更新客户端图表...');
        if (!this.chart) {
            console.error('❌ 客户端图表未初始化');
            return;
        }
        
        console.log('📊 更新图表数据:', {
            labels: this.trainingData.server.rounds,
            data: this.trainingData.server.aggregatedLoss
        });
        
        const chart = this.chart;
        chart.data.labels = this.trainingData.server.rounds;
        chart.data.datasets[0].data = this.trainingData.server.aggregatedLoss;
        chart.update('active');
        console.log('✅ 客户端图表更新完成');
    }

    updateMetricsDisplay() {
        // 基于服务器聚合数据更新指标
        const aggregatedLosses = this.trainingData.server.aggregatedLoss;
        const rounds = this.trainingData.server.rounds;
        const clientsCount = this.trainingData.server.clientsCount;
        
        if (aggregatedLosses.length === 0) return;
        
        // 计算当前、平均和最小损失
        const currentLoss = aggregatedLosses[aggregatedLosses.length - 1];
        const avgLoss = aggregatedLosses.reduce((sum, loss) => sum + loss, 0) / aggregatedLosses.length;
        const minLoss = Math.min(...aggregatedLosses);
        const totalRounds = rounds.length;
        const activeClients = clientsCount.length > 0 ? clientsCount[clientsCount.length - 1] : 0;
        
        // 更新当前聚合损失
        const currentLossEl = document.getElementById('currentLossValue');
        if (currentLossEl) {
            currentLossEl.textContent = currentLoss.toFixed(6);
        }
        
        // 更新平均聚合损失
        const avgLossEl = document.getElementById('avgLossValue');
        if (avgLossEl) {
            avgLossEl.textContent = avgLoss.toFixed(6);
        }
        
        // 更新最佳聚合损失
        const minLossEl = document.getElementById('minLossValue');
        if (minLossEl) {
            minLossEl.textContent = minLoss.toFixed(6);
        }
        
        // 更新训练轮数
        const totalRoundsEl = document.getElementById('totalRoundsValue');
        if (totalRoundsEl) {
            totalRoundsEl.textContent = totalRounds;
        }
        
        // 更新当前轮次显示
        const currentRoundEl = document.getElementById('currentRoundDisplay');
        if (currentRoundEl) {
            currentRoundEl.textContent = this.currentRound;
        }
        
        // 更新参与客户端数
        const activeClientsEl = document.getElementById('activeClientsDisplay');
        if (activeClientsEl) {
            activeClientsEl.textContent = activeClients;
        }
        
        // 更新内部指标数据
        this.metricsData.currentLoss = currentLoss;
        this.metricsData.avgLoss = avgLoss;
        this.metricsData.minLoss = minLoss;
        this.metricsData.totalRounds = totalRounds;
        this.metricsData.activeClients = activeClients;
        this.metricsData.lastUpdateTime = new Date();
        
        console.log('📊 客户端指标已更新:', this.metricsData);
    }

    updateStatus(status, text) {
        const statusEl = document.getElementById('trainingStatus');
        const statusTextEl = document.getElementById('trainingStatusText');
        
        if (statusEl) {
            statusEl.className = `training-status ${status}`;
            statusEl.querySelector('span').textContent = text;
        }
        
        if (statusTextEl) {
            statusTextEl.textContent = text;
        }
    }

    updateCurrentRound(round) {
        const roundEl = document.getElementById('currentRoundDisplay');
        if (roundEl) {
            roundEl.textContent = round;
        }
    }

    startPeriodicUpdate() {
        // 定期更新（每5秒）- 用于备用数据获取
        setInterval(() => {
            if (!this.isTraining) {
                this.fetchLatestData();
            }
        }, 5000);
    }

    async fetchLatestData() {
        try {
            console.log(`🔍 尝试获取客户端 ${this.clientId} 的最新训练数据...`);
            const response = await fetch(`/api/training/client/${this.clientId}/data`);
            if (response.ok) {
                const data = await response.json();
                console.log(`✅ 获取到最新训练数据:`, data);
                this.processLatestData(data);
            } else {
                console.error(`❌ 获取训练数据失败: ${response.status} ${response.statusText}`);
            }
        } catch (error) {
            console.error('❌ 获取训练数据异常:', error);
        }
    }

    processLatestData(data) {
        // 客户端现在只使用WebSocket实时数据，不再从API获取历史数据
        console.log('📊 收到API数据（客户端忽略，使用WebSocket数据）:', data);
        
        if (data.status) {
            this.updateStatus(data.status, data.status_text || '');
        }
        
        if (data.current_round) {
            this.updateCurrentRound(data.current_round);
        }
    }

    clearData() {
        if (confirm('确定要清空所有训练数据吗？此操作不可撤销。')) {
            this.trainingData = {
                server: {
                    rounds: [],
                    aggregatedLoss: [],
                    clientsCount: []
                },
                clients: {}
            };
            this.metricsData = {
                currentLoss: 0,
                avgLoss: 0,
                minLoss: Infinity,
                totalRounds: 0,
                activeClients: 0,
                lastUpdateTime: null
            };
            
            this.updateChart();
            this.updateMetricsDisplay();
            
            // 显示无数据消息
            const noDataMsg = document.getElementById('noDataMessage');
            if (noDataMsg) noDataMsg.style.display = 'block';
        }
    }

    exportData() {
        if (this.trainingData.server.aggregatedLoss.length === 0) {
            alert('没有可导出的数据');
            return;
        }

        const exportData = {
            client_id: this.clientId,
            export_time: new Date().toISOString(),
            training_metrics: this.metricsData,
            server_aggregated_data: this.trainingData.server,
            clients_data: this.trainingData.clients
        };

        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `client_${this.clientId}_federated_training_data_${new Date().toISOString().slice(0, 10)}.json`;
        link.click();
    }

    refreshChart() {
        if (this.chart) {
            this.chart.update('active');
        }
        this.fetchLatestData();
    }

    destroy() {
        if (this.ws) {
            this.ws.close();
        }
        if (this.chart) {
            this.chart.destroy();
        }
    }
}

// 全局实例
let clientTrainingChart = null;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 使用当前登录的用户名作为客户端ID
    const clientId = window.currentUsername || 'unknown_client';
    
    // 创建图表容器（如果不存在）
    if (!document.getElementById('clientTrainingChart')) {
        const container = document.createElement('div');
        container.id = 'clientTrainingChart';
        container.style.margin = '2rem 0';
        
        // 在合适的位置插入容器
        const mainContent = document.querySelector('.dashboard') || document.body;
        if (mainContent) {
            mainContent.appendChild(container);
        }
    }
    
    // 初始化客户端训练图表
    clientTrainingChart = new ClientTrainingChart('clientTrainingChart', clientId);
});
