/**
 * 服务器端训练损失可视化管理器
 * 处理所有客户端训练数据的聚合显示和实时更新
 */
class ServerTrainingVisualizer {
    constructor() {
        console.log('🏗️ 开始构造ServerTrainingVisualizer实例...');
        
        this.serverChart = null;
        this.socket = null;
        this.trainingData = {
            server: {
                rounds: [],
                aggregatedLoss: [],
                clientsCount: []
            },
            clients: {}  // {client_id: {rounds: [], losses: []}}
        };
        this.isTraining = false;
        this.currentRound = 0;
        
        console.log('📊 初始化数据结构完成');
        console.log('🚀 开始调用init()方法...');
        this.init();
        console.log('✅ ServerTrainingVisualizer构造完成');
    }
    
    /**
     * 初始化可视化组件
     */
    init() {
        this.initCharts();
        this.initWebSocket();
        this.updateUI();
    }
    
    /**
     * 初始化图表
     */
    initCharts() {
        console.log('🎯 开始初始化图表...');
        
        // 服务器聚合损失图表
        const serverCtx = document.getElementById('serverAggregatedChart');
        console.log('📊 查找服务器图表元素:', serverCtx ? '✅ 找到' : '❌ 未找到');
        if (serverCtx) {
            this.serverChart = new Chart(serverCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: '服务器聚合损失',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        pointBackgroundColor: '#667eea',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: '联邦学习聚合损失趋势',
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: '训练轮次'
                            },
                            grid: { display: true, alpha: 0.3 }
                        },
                        y: {
                            title: {
                                display: true,
                                text: '损失值'
                            },
                            beginAtZero: false,
                            grid: { display: true, alpha: 0.3 }
                        }
                    },
                    animation: {
                        duration: 1000,
                        easing: 'easeInOutQuart'
                    }
                }
            });
            console.log('✅ 服务器聚合图表初始化成功');
        } else {
            console.error('❌ 未找到服务器图表元素 #serverAggregatedChart');
        }
    }
    
    /**
     * 初始化WebSocket连接
     */
    initWebSocket() {
        console.log('🔌 开始初始化WebSocket连接...');
        if (typeof io !== 'undefined') {
            console.log('✅ Socket.IO库已加载');
            this.socket = io();
            
            this.socket.on('connect', () => {
                console.log('✅ WebSocket连接已建立');
            });
            
            this.socket.on('disconnect', () => {
                console.log('❌ WebSocket连接已断开');
            });
            
            // 监听服务器训练更新
            this.socket.on('server_training_update', (data) => {
                console.log('📊 收到server_training_update事件:', data);
                this.handleServerTrainingUpdate(data);
            });
            
            // 监听客户端训练更新
            this.socket.on('client_training_update', (data) => {
                console.log('📱 收到client_training_update事件:', data);
                this.handleClientTrainingUpdate(data);
            });
            
            // 加入服务器训练监控房间
            console.log('🏠 尝试加入服务器训练监控房间...');
            this.socket.emit('join_training_room', { role: 'server' });
            
            this.socket.on('training_room_joined', (data) => {
                console.log('✅ 成功加入训练监控房间:', data);
            });
            
            console.log('🔌 服务器训练可视化WebSocket已连接');
        } else {
            console.error('❌ Socket.IO库未加载，无法初始化WebSocket连接');
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
                    // 更新实时损失
                    clientData.currentLoss = data.current_loss;
                }
                break;
                
            case 'round_complete':
                if (data.round !== undefined && data.average_loss !== undefined) {
                    // 检查是否已存在该轮次的数据，避免重复
                    const existingIndex = clientData.rounds.indexOf(data.round);
                    if (existingIndex === -1) {
                        // 新轮次数据，直接添加
                        clientData.rounds.push(data.round);
                        clientData.losses.push(data.average_loss);
                    } else {
                        // 更新现有轮次数据
                        console.log(`🔄 更新客户端${data.client_id}轮次${data.round}的数据`);
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
    
    /**
     * 处理训练轮次完成
     */
    handleRoundComplete(data) {
        console.log('🔄 处理轮次完成数据:', data);
        console.log('📊 检查字段:', {
            round: data.round,
            average_loss: data.average_loss,
            participating_clients: data.participating_clients
        });
        
        if (data.round !== undefined && data.average_loss !== undefined) {
            console.log('✅ 数据有效，更新服务器图表数据');
            
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
            
            console.log('📈 当前服务器数据:', {
                rounds: this.trainingData.server.rounds,
                losses: this.trainingData.server.aggregatedLoss,
                clientsCount: this.trainingData.server.clientsCount
            });
            
            this.currentRound = data.round;
            this.updateServerChart();
            this.updateMetrics();
        } else {
            console.warn('⚠️ 轮次完成数据缺少必要字段:', data);
        }
    }
    
    /**
     * 处理训练开始
     */
    handleTrainingStart(data) {
        this.isTraining = true;
        this.updateTrainingStatus('training');
        console.log('🚀 联邦学习训练开始');
    }
    
    /**
     * 处理训练完成
     */
    handleTrainingComplete(data) {
        this.isTraining = false;
        this.updateTrainingStatus('completed');
        console.log('✅ 联邦学习训练完成');
    }
    
    /**
     * 更新服务器聚合图表
     */
    updateServerChart() {
        console.log('📈 尝试更新服务器图表...');
        if (!this.serverChart) {
            console.error('❌ 服务器图表未初始化');
            return;
        }
        
        console.log('📊 更新图表数据:', {
            labels: this.trainingData.server.rounds,
            data: this.trainingData.server.aggregatedLoss
        });
        
        const chart = this.serverChart;
        chart.data.labels = this.trainingData.server.rounds;
        chart.data.datasets[0].data = this.trainingData.server.aggregatedLoss;
        chart.update('active');
        console.log('✅ 服务器图表更新完成');
    }
    
    /**
     * 更新指标卡片
     */
    updateMetrics() {
        // 更新服务器指标
        const currentRoundEl = document.getElementById('serverCurrentRound');
        const averageLossEl = document.getElementById('serverAverageLoss');
        const activeClientsEl = document.getElementById('serverActiveClients');
        
        if (currentRoundEl) {
            currentRoundEl.textContent = this.currentRound;
        }
        
        if (averageLossEl && this.trainingData.server.aggregatedLoss.length > 0) {
            const lastLoss = this.trainingData.server.aggregatedLoss[this.trainingData.server.aggregatedLoss.length - 1];
            averageLossEl.textContent = lastLoss.toFixed(4);
        }
        
        if (activeClientsEl) {
            activeClientsEl.textContent = Object.keys(this.trainingData.clients).length;
        }
    }
    
    /**
     * 更新训练状态
     */
    updateTrainingStatus(status) {
        const statusEl = document.getElementById('serverTrainingStatus');
        if (!statusEl) return;
        
        const statusConfig = {
            waiting: {
                text: '等待训练开始',
                icon: 'fas fa-clock',
                class: 'status-waiting'
            },
            training: {
                text: '联邦学习训练中',
                icon: 'fas fa-cog fa-spin',
                class: 'status-training'
            },
            completed: {
                text: '训练已完成',
                icon: 'fas fa-check-circle',
                class: 'status-completed'
            }
        };
        
        const config = statusConfig[status] || statusConfig.waiting;
        statusEl.className = `status-badge ${config.class}`;
        statusEl.innerHTML = `<i class="${config.icon}"></i>${config.text}`;
    }
    
    /**
     * 更新UI状态
     */
    updateUI() {
        this.updateTrainingStatus('waiting');
        this.updateMetrics();
    }
    
    /**
     * 导出服务器训练数据
     */
    exportData() {
        const exportData = {
            server: this.trainingData.server,
            clients: this.trainingData.clients,
            exportTime: new Date().toISOString(),
            summary: {
                totalRounds: this.trainingData.server.rounds.length,
                totalClients: Object.keys(this.trainingData.clients).length,
                finalLoss: this.trainingData.server.aggregatedLoss.length > 0 ? 
                    this.trainingData.server.aggregatedLoss[this.trainingData.server.aggregatedLoss.length - 1] : null
            }
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `federated_training_data_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log('📥 训练数据已导出');
    }
    
    /**
     * 重置图表
     */
    resetCharts() {
        if (confirm('确定要重置所有图表数据吗？此操作不可撤销。')) {
            this.trainingData = {
                server: {
                    rounds: [],
                    aggregatedLoss: [],
                    clientsCount: []
                },
                clients: {}
            };
            
            this.currentRound = 0;
            this.isTraining = false;
            
            if (this.serverChart) {
                this.serverChart.data.labels = [];
                this.serverChart.data.datasets[0].data = [];
                this.serverChart.update();
            }
            
            this.updateUI();
            console.log('🔄 图表已重置');
        }
    }
}

// 全局函数供HTML调用
let serverTrainingVisualizer;

/**
 * 导出服务器训练数据
 */
function exportServerTrainingData() {
    if (serverTrainingVisualizer) {
        serverTrainingVisualizer.exportData();
    }
}

/**
 * 重置服务器图表
 */
function resetServerCharts() {
    if (serverTrainingVisualizer) {
        serverTrainingVisualizer.resetCharts();
    }
}

/**
 * 测试函数 - 手动调试用
 */
function testServerCharts() {
    console.log('🧪 开始手动测试服务器图表...');
    
    // 检查元素
    const serverEl = document.getElementById('serverAggregatedChart');
    console.log('📊 元素检查:');
    console.log('  - serverAggregatedChart:', serverEl ? '✅ 存在' : '❌ 不存在');
    
    // 检查库
    console.log('📦 库检查:');
    console.log('  - Chart.js:', typeof Chart !== 'undefined' ? '✅ 已加载' : '❌ 未加载');
    console.log('  - Socket.IO:', typeof io !== 'undefined' ? '✅ 已加载' : '❌ 未加载');
    
    // 检查可视化器实例
    console.log('🎯 可视化器检查:');
    console.log('  - serverTrainingVisualizer:', serverTrainingVisualizer ? '✅ 已创建' : '❌ 未创建');
    
    if (serverTrainingVisualizer) {
        console.log('  - serverChart:', serverTrainingVisualizer.serverChart ? '✅ 已初始化' : '❌ 未初始化');
        console.log('  - socket:', serverTrainingVisualizer.socket ? '✅ 已连接' : '❌ 未连接');
    }
    
    return {
        serverElement: !!serverEl,
        chartJs: typeof Chart !== 'undefined',
        socketIo: typeof io !== 'undefined',
        visualizer: !!serverTrainingVisualizer,
        serverChart: serverTrainingVisualizer?.serverChart ? true : false,
        socket: serverTrainingVisualizer?.socket ? true : false
    };
}

/**
 * 发送测试数据
 */
function sendTestServerData() {
    console.log('📤 发送测试服务器数据...');
    
    if (!serverTrainingVisualizer || !serverTrainingVisualizer.socket) {
        console.error('❌ WebSocket未连接');
        return;
    }
    
    // 模拟服务器轮次完成数据
    const testData = {
        type: 'round_complete',
        round: 1,
        average_loss: 1.5,
        participating_clients: 2,
        timestamp: new Date().toISOString()
    };
    
    console.log('📊 发送数据:', testData);
    serverTrainingVisualizer.handleServerTrainingUpdate(testData);
    
    // 模拟客户端数据
    const clientData = {
        type: 'round_complete',
        client_id: 'test_client_1',
        round: 1,
        average_loss: 1.6,
        timestamp: new Date().toISOString()
    };
    
    console.log('📱 发送客户端数据:', clientData);
    serverTrainingVisualizer.handleClientTrainingUpdate(clientData);
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('🔄 DOM内容已加载，开始初始化服务器训练可视化...');
    
    // 检查必要的依赖
    console.log('📦 检查依赖库:');
    console.log('  - Chart.js:', typeof Chart !== 'undefined' ? '✅ 已加载' : '❌ 未加载');
    console.log('  - Socket.IO:', typeof io !== 'undefined' ? '✅ 已加载' : '❌ 未加载');
    
    // 检查DOM元素
    console.log('🔍 检查DOM元素:');
    const serverChartEl = document.getElementById('serverAggregatedChart');
    console.log('  - serverAggregatedChart:', serverChartEl ? '✅ 存在' : '❌ 不存在');
    
    // 确保Chart.js已加载
    if (typeof Chart !== 'undefined') {
        console.log('🎯 开始初始化服务器训练可视化...');
        serverTrainingVisualizer = new ServerTrainingVisualizer();
        console.log('🎯 服务器训练可视化初始化完成');
    } else {
        console.error('❌ Chart.js未加载，无法初始化训练可视化');
        // 尝试延迟初始化
        setTimeout(() => {
            if (typeof Chart !== 'undefined') {
                console.log('🔄 延迟初始化服务器训练可视化...');
                serverTrainingVisualizer = new ServerTrainingVisualizer();
                console.log('🎯 延迟初始化完成');
            } else {
                console.error('❌ Chart.js仍未加载，初始化失败');
            }
        }, 1000);
    }
});
