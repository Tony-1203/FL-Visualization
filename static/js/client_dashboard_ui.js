// WebSocket 连接
let socket = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

// 初始化 WebSocket 连接
function initializeSocket() {
    socket = io();

    socket.on('connect', function() {
        console.log('WebSocket 连接成功');
        updateConnectionStatus(true);
        reconnectAttempts = 0;
        
        // 请求状态更新
        socket.emit('request_status_update');
    });

    socket.on('disconnect', function() {
        console.log('WebSocket 连接断开');
        updateConnectionStatus(false);
        
        // 尝试重连
        if (reconnectAttempts < maxReconnectAttempts) {
            setTimeout(() => {
                reconnectAttempts++;
                console.log(`尝试重连... (${reconnectAttempts}/${maxReconnectAttempts})`);
                socket.connect();
            }, 2000 * reconnectAttempts);
        }
    });

    socket.on('user_status_update', function(data) {
        updateUserStatus(data);
    });

    socket.on('heartbeat_response', function(data) {
        console.log('心跳响应:', data.timestamp);
    });

    // 定期发送心跳
    setInterval(() => {
        if (socket && socket.connected) {
            socket.emit('heartbeat');
        }
    }, 30000); // 每30秒发送一次心跳
}

// 更新连接状态
function updateConnectionStatus(isConnected) {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    
    if (isConnected) {
        statusIndicator.className = 'online-indicator';
        statusText.textContent = '已连接';
    } else {
        statusIndicator.className = 'online-indicator offline-indicator';
        statusText.textContent = '连接断开';
    }
}

// 更新用户状态
function updateUserStatus(data) {
    document.getElementById('onlineCount').textContent = data.total_online;
    document.getElementById('serverStatus').textContent = 
        data.online_servers.length > 0 ? '在线' : '离线';
    document.getElementById('lastUpdate').textContent = 
        new Date(data.timestamp).toLocaleTimeString();
}

// 创建动态粒子效果
function createParticles() {
    const particlesContainer = document.querySelector('.background-particles');
    const particleCount = 20;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 8 + 's';
        particle.style.animationDuration = (Math.random() * 8 + 8) + 's';
        particlesContainer.appendChild(particle);
    }
}

// 文件上传功能
async function uploadFiles() {
    const form = document.getElementById('uploadForm');
    const formData = new FormData(form);
    const statusDiv = document.getElementById('uploadStatus');
    const uploadBtn = document.getElementById('uploadBtn');

    // 检查文件是否选择
    const filesInput = document.getElementById('files');
    const files = filesInput.files;

    if (!files || files.length === 0) {
        showStatus('error', '请至少选择一个文件!', 'fas fa-exclamation-triangle');
        return;
    }

    // 检查文件类型
    let mhdCount = 0, rawCount = 0;
    for (let file of files) {
        if (file.name.toLowerCase().endsWith('.mhd')) mhdCount++;
        else if (file.name.toLowerCase().endsWith('.raw')) rawCount++;
    }

    if (mhdCount === 0 && rawCount === 0) {
        showStatus('error', '请选择 .mhd 或 .raw 文件!', 'fas fa-exclamation-triangle');
        return;
    }

    // 显示加载状态
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<div class="loading-spinner"></div> 上传中...';
    showStatus('loading', `正在上传 ${files.length} 个文件，请稍候...`, 'loading-spinner');

    try {
        const response = await fetch("/client/upload", {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        if (response.ok) {
            showStatus('success', 
                `<strong>上传成功!</strong><br>
                 <small>总计: ${result.total_files} 个文件 (${result.file_pairs} 对)</small>`, 
                'fas fa-check-circle'
            );
            
            // 重置表单和选择的文件显示
            form.reset();
            document.getElementById('selectedFiles').style.display = 'none';
            
            // 延迟后重新加载页面以更新状态
            setTimeout(() => window.location.reload(), 2000);
        } else {
            showStatus('error', 
                `<strong>上传失败</strong><br><small>${result.error || '未知错误'}</small>`, 
                'fas fa-exclamation-triangle'
            );
        }
    } catch (error) {
        showStatus('error', 
            `<strong>网络错误</strong><br><small>${error.message}</small>`, 
            'fas fa-wifi'
        );
    } finally {
        // 恢复按钮状态
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i class="fas fa-cloud-upload-alt"></i> 上传文件';
    }
}

// 显示选择的文件
function updateSelectedFiles() {
    const filesInput = document.getElementById('files');
    const selectedFilesDiv = document.getElementById('selectedFiles');
    const filesListDiv = document.getElementById('filesList');
    
    if (filesInput.files.length === 0) {
        selectedFilesDiv.style.display = 'none';
        return;
    }

    selectedFilesDiv.style.display = 'block';
    filesListDiv.innerHTML = '';

    Array.from(filesInput.files).forEach(file => {
        const fileTag = document.createElement('div');
        const ext = file.name.toLowerCase().endsWith('.mhd') ? 'mhd' : 
                   file.name.toLowerCase().endsWith('.raw') ? 'raw' : 'other';
        
        fileTag.className = `file-tag ${ext}`;
        fileTag.innerHTML = `
            <i class="fas fa-file${ext === 'mhd' ? '-medical' : ext === 'raw' ? '-code' : ''}"></i>
            ${file.name}
        `;
        filesListDiv.appendChild(fileTag);
    });
}

// 显示状态消息
function showStatus(type, message, icon) {
    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.style.display = 'block';
    statusDiv.className = `status-message status-${type}`;
    
    if (icon === 'loading-spinner') {
        statusDiv.innerHTML = `<div class="loading-spinner"></div><div>${message}</div>`;
    } else {
        statusDiv.innerHTML = `<i class="${icon}"></i><div>${message}</div>`;
    }
}

// 创建文件列表
function createFileList(files) {
    const dt = new DataTransfer();
    files.forEach(file => dt.items.add(file));
    return dt.files;
}

// 页面加载完成后的初始化
document.addEventListener('DOMContentLoaded', function() {
    createParticles();
    initializeSocket(); // 初始化 WebSocket 连接
    
    // 文件输入交互
    const fileInputs = document.querySelectorAll('.file-input');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            if (this.files.length > 0) {
                this.style.backgroundColor = 'rgba(40, 167, 69, 0.05)';
                updateSelectedFiles();
            } else {
                this.style.backgroundColor = 'white';
            }
        });
    });

    // 拖拽上传功能
    const uploadSection = document.querySelector('.upload-section');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadSection.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadSection.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadSection.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadSection.style.borderColor = '#667eea';
        uploadSection.style.backgroundColor = 'rgba(102, 126, 234, 0.1)';
    }

    function unhighlight(e) {
        uploadSection.style.borderColor = 'rgba(102, 126, 234, 0.3)';
        uploadSection.style.backgroundColor = 'linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05))';
    }

    uploadSection.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            // 将拖拽的文件设置到文件输入框
            const filesInput = document.getElementById('files');
            filesInput.files = files;
            
            // 触发change事件
            filesInput.dispatchEvent(new Event('change'));
        }
    }

    // 上传按钮点击事件
    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadBtn) {
        uploadBtn.addEventListener('click', uploadFiles);
    }

    // 文件输入变化事件
    const filesInput = document.getElementById('files');
    if (filesInput) {
        filesInput.addEventListener('change', updateSelectedFiles);
    }
});

// 推理功能相关变量
let selectedInferenceFile = null;
let inferenceStatusInterval = null;

// 处理推理文件上传
async function handleInferenceFileUpload(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    showUploadProgress();
    const formData = new FormData();
    
    // 添加所有文件到FormData
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
        addUploadProgressItem(files[i].name, 'uploading');
    }

    try {
        const response = await fetch('/api/client/upload_inference_file', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            showNotification(result.message, 'success');
            loadInferenceFiles(); // 重新加载文件列表
            
            // 更新上传进度
            if (result.uploaded_files) {
                result.uploaded_files.forEach(filename => {
                    updateUploadProgressItem(filename, 'success');
                });
            }
            
            if (result.errors && result.errors.length > 0) {
                result.errors.forEach(error => {
                    showNotification(error, 'warning');
                });
            }
        } else {
            showNotification(result.error || '文件上传失败', 'error');
            // 将所有文件标记为失败
            for (let i = 0; i < files.length; i++) {
                updateUploadProgressItem(files[i].name, 'error');
            }
        }
    } catch (error) {
        showNotification('上传错误: ' + error.message, 'error');
        // 将所有文件标记为失败
        for (let i = 0; i < files.length; i++) {
            updateUploadProgressItem(files[i].name, 'error');
        }
    }

    // 清空文件输入
    event.target.value = '';
    
    // 3秒后隐藏上传进度
    setTimeout(hideUploadProgress, 3000);
}

// 加载推理文件列表
async function loadInferenceFiles() {
    try {
        const response = await fetch('/api/client/list_inference_files');
        const result = await response.json();

        // 缓存文件列表数据
        window.currentFileList = result.files || [];

        const fileList = document.getElementById('inferenceFileList');
        
        if (result.files && result.files.length > 0) {
            fileList.innerHTML = result.files.map(file => `
                <div class="inference-file-item" onclick="selectInferenceFile('${file.name}', this)">
                    <div class="file-info">
                        <div class="file-name">
                            <strong>${file.name}</strong>
                            ${file.has_pair ? '<i class="fas fa-check-circle" style="color: #48bb78; margin-left: 8px;" title="文件对完整"></i>' : '<i class="fas fa-exclamation-triangle" style="color: #ed8936; margin-left: 8px;" title="缺少配对文件"></i>'}
                        </div>
                        <div class="file-details">
                            <span>大小: ${(file.total_size / 1024 / 1024).toFixed(2)} MB</span>
                            <span>上传时间: ${file.upload_time}</span>
                            ${file.has_pair ? `<span>配对文件: ${file.raw_file}</span>` : '<span style="color: #ed8936;">缺少.raw文件</span>'}
                        </div>
                        <div class="file-status">
                            <div class="status-indicator-small ${file.has_pair ? 'status-complete' : 'status-incomplete'}"></div>
                            <span>${file.has_pair ? '文件对完整' : '文件对不完整'}</span>
                        </div>
                    </div>
                    <div class="file-actions">
                        <button class="btn-danger btn-small" onclick="event.stopPropagation(); deleteInferenceFile('${file.name}')" title="删除文件对">
                            <i class="fas fa-trash"></i>
                        </button>
                        <i class="file-type-icon fas fa-file-medical"></i>
                    </div>
                </div>
            `).join('');
        } else {
            fileList.innerHTML = '<p style="color: #718096; text-align: center;">暂无上传的推理文件</p>';
        }
    } catch (error) {
        console.error('加载文件列表失败:', error);
    }
}

// 删除推理文件
async function deleteInferenceFile(filename) {
    if (!confirm(`确定要删除文件对 "${filename}" 吗？这将同时删除对应的.raw文件。`)) {
        return;
    }

    try {
        const response = await fetch('/api/client/delete_inference_file', {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filename: filename })
        });

        const result = await response.json();

        if (response.ok) {
            showNotification(result.message, 'success');
            loadInferenceFiles(); // 重新加载文件列表
            
            // 如果删除的是当前选中的文件，清除选择
            if (selectedInferenceFile === filename) {
                selectedInferenceFile = null;
                document.getElementById('runInferenceBtn').disabled = true;
            }
        } else {
            showNotification(result.error || '删除失败', 'error');
        }
    } catch (error) {
        showNotification('删除错误: ' + error.message, 'error');
    }
}

// 拖拽上传功能
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDragEnter(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
}

function handleFileDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        // 模拟文件输入
        const fileInput = document.getElementById('inferenceFileInput');
        fileInput.files = files;
        handleInferenceFileUpload({ target: fileInput });
    }
}

// 上传进度相关函数
function showUploadProgress() {
    document.getElementById('uploadProgress').style.display = 'block';
    document.getElementById('uploadProgressList').innerHTML = '';
}

function hideUploadProgress() {
    document.getElementById('uploadProgress').style.display = 'none';
}

function addUploadProgressItem(filename, status) {
    const progressList = document.getElementById('uploadProgressList');
    const progressItem = document.createElement('div');
    progressItem.className = 'progress-item';
    progressItem.id = `progress-${filename}`;
    
    const statusIcon = status === 'uploading' ? 
        '<i class="fas fa-spinner fa-spin"></i>' : 
        status === 'success' ? 
        '<i class="fas fa-check"></i>' : 
        '<i class="fas fa-times"></i>';
    
    progressItem.innerHTML = `
        <span class="progress-filename">${filename}</span>
        <span class="progress-status progress-${status}">${statusIcon} ${status === 'uploading' ? '上传中' : status === 'success' ? '成功' : '失败'}</span>
    `;
    
    progressList.appendChild(progressItem);
}

function updateUploadProgressItem(filename, status) {
    const progressItem = document.getElementById(`progress-${filename}`);
    if (progressItem) {
        const statusElement = progressItem.querySelector('.progress-status');
        statusElement.className = `progress-status ${status === 'success' ? 'progress-success' : 'progress-error'}`;
        statusElement.innerHTML = status === 'success' ? '<i class="fas fa-check"></i> 成功' : '<i class="fas fa-times"></i> 失败';
    }
}

// 清空所有文件
async function clearAllFiles() {
    if (!confirm('确定要删除所有推理文件吗？')) {
        return;
    }

    try {
        const response = await fetch('/api/client/list_inference_files');
        const result = await response.json();
        
        if (result.files && result.files.length > 0) {
            for (const file of result.files) {
                await deleteInferenceFile(file.name);
            }
        }
    } catch (error) {
        showNotification('清空文件时出错: ' + error.message, 'error');
    }
}

// 刷新文件列表
function refreshFileList() {
    showNotification('正在刷新文件列表...', 'info');
    loadInferenceFiles();
}

// 选择推理文件
function selectInferenceFile(filename, element) {
    // 移除之前的选中状态
    document.querySelectorAll('.inference-file-item').forEach(item => {
        item.classList.remove('selected');
    });

    // 添加选中状态
    element.classList.add('selected');
    selectedInferenceFile = filename;

    // 检查文件对是否完整
    const fileData = getFileData(filename);
    if (fileData && fileData.has_pair) {
        // 启用运行按钮
        document.getElementById('runInferenceBtn').disabled = false;
        showNotification(`已选择文件: ${filename}`, 'info');
    } else {
        // 禁用运行按钮
        document.getElementById('runInferenceBtn').disabled = true;
        showNotification(`文件对不完整，无法进行推理: ${filename}`, 'warning');
    }
}

// 获取文件数据
function getFileData(filename) {
    // 这里应该从最近一次加载的文件列表中获取数据
    // 为了简化，我们可以重新获取，但在实际应用中应该缓存这些数据
    return window.currentFileList ? window.currentFileList.find(f => f.name === filename) : null;
}

// 运行推理
async function runInference() {
    if (!selectedInferenceFile) {
        showNotification('请先选择一个推理文件', 'error');
        return;
    }

    const inferenceMode = document.querySelector('input[name="inferenceMode"]:checked').value;
    const fastMode = document.getElementById('fastModeCheckbox').checked;

    const requestData = {
        filename: selectedInferenceFile,
        use_federated: inferenceMode === 'federated',
        fast_mode: fastMode
    };

    try {
        const response = await fetch('/api/client/run_inference', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        const result = await response.json();

        if (response.ok) {
            showNotification('推理已开始，请等待...', 'info');
            
            // 禁用运行按钮
            document.getElementById('runInferenceBtn').disabled = true;
            
            // 显示推理状态
            document.getElementById('inferenceStatus').style.display = 'block';
            document.getElementById('inferenceResult').style.display = 'none';
            
            // 开始轮询推理状态
            inferenceStatusInterval = setInterval(fetchInferenceStatus, 2000);
            
        } else {
            showNotification(result.error || '推理启动失败', 'error');
        }
    } catch (error) {
        showNotification('推理请求错误: ' + error.message, 'error');
    }
}

// 获取推理状态
async function fetchInferenceStatus() {
    try {
        const response = await fetch('/api/client/get_inference_status');
        const status = await response.json();

        if (response.ok) {
            updateInferenceStatus(status);

            // 如果推理完成
            if (!status.is_running) {
                // 停止轮询
                if (inferenceStatusInterval) {
                    clearInterval(inferenceStatusInterval);
                    inferenceStatusInterval = null;
                }

                // 重新启用运行按钮
                document.getElementById('runInferenceBtn').disabled = false;

                // 如果有结果图像，显示它
                if (status.result_image) {
                    displayInferenceResult(status.result_image);
                }

                // 如果有错误，显示错误信息
                if (status.error) {
                    showNotification('推理失败: ' + status.error, 'error');
                    document.getElementById('inferenceStatus').style.display = 'none';
                }
            }
        }
    } catch (error) {
        console.error('获取推理状态失败:', error);
    }
}

// 更新推理状态显示
function updateInferenceStatus(status) {
    const statusText = document.getElementById('inferenceStatusText');
    const progressBar = document.getElementById('inferenceProgressBar');
    const progressText = document.getElementById('inferenceProgressText');

    statusText.textContent = status.current_step || '推理中...';
    progressBar.style.width = status.progress + '%';
    progressText.textContent = status.progress + '%';

    // 如果推理完成，隐藏状态面板
    if (!status.is_running && status.progress === 100) {
        setTimeout(() => {
            document.getElementById('inferenceStatus').style.display = 'none';
        }, 2000);
    }
}

// 显示推理结果
function displayInferenceResult(imageData) {
    const resultDiv = document.getElementById('inferenceResult');
    const resultImage = document.getElementById('resultImage');
    const resultText = document.getElementById('resultText');

    resultImage.src = imageData;
    resultImage.style.display = 'block';
    resultText.textContent = '推理完成！点击图像可查看大图。';
    resultDiv.style.display = 'block';

    // 添加点击图像放大功能
    resultImage.onclick = function() {
        const modal = document.createElement('div');
        modal.className = 'image-preview-modal';
        modal.innerHTML = `
            <div class="image-preview-content">
                <span class="close-preview">&times;</span>
                <img src="${this.src}" alt="推理结果预览">
            </div>
        `;
        document.body.appendChild(modal);

        // 点击关闭预览
        modal.addEventListener('click', function(e) {
            if (e.target === modal || e.target.className === 'close-preview') {
                modal.remove();
            }
        });

        // 动画效果
        setTimeout(() => {
            modal.style.opacity = '1';
        }, 10);
    };

    // 添加完成动画
    animateInferenceComplete();
}

// 推理完成后的动画效果
function animateInferenceComplete() {
    const resultSection = document.getElementById('inferenceResult');
    if (resultSection) {
        resultSection.style.animation = 'none';
        setTimeout(() => {
            resultSection.style.animation = 'fadeInUp 0.8s ease-out forwards';
        }, 10);
    }
    
    const resultImage = document.getElementById('resultImage');
    if (resultImage) {
        resultImage.style.animation = 'none';
        setTimeout(() => {
            resultImage.style.animation = 'zoomIn 1s ease-out forwards';
        }, 500);
    }
}

// 显示通知消息
function showNotification(message, type) {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // 添加到页面
    document.body.appendChild(notification);
    
    // 显示通知
    setTimeout(() => notification.classList.add('show'), 100);
    
    // 自动隐藏
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// 增强的推理模块交互效果
function enhanceInferenceUI() {
    // 使推理文件项目可选择
    document.addEventListener('click', function(e) {
        const fileItem = e.target.closest('.inference-file-item');
        if (fileItem) {
            // 移除其他选中项
            document.querySelectorAll('.inference-file-item').forEach(item => {
                item.classList.remove('selected');
            });
            
            // 添加选中效果
            fileItem.classList.add('selected');
            
            // 添加微妙的弹跳动画
            fileItem.style.animation = 'none';
            setTimeout(() => {
                fileItem.style.animation = 'bounce 0.5s ease';
            }, 10);
        }
    });
    
    // 上传区域拖拽动画增强
    const uploadZone = document.querySelector('.upload-zone');
    if (uploadZone) {
        uploadZone.addEventListener('dragenter', function() {
            this.style.transform = 'scale(1.02)';
            this.style.boxShadow = '0 15px 30px rgba(0, 0, 0, 0.15)';
        });
        
        uploadZone.addEventListener('dragleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
        
        uploadZone.addEventListener('drop', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
    }
    
    // 推理结果图片点击预览效果
    const resultImage = document.getElementById('resultImage');
    if (resultImage) {
        resultImage.addEventListener('click', function() {
            const modal = document.createElement('div');
            modal.className = 'image-preview-modal';
            modal.innerHTML = `
                <div class="image-preview-content">
                    <span class="close-preview">&times;</span>
                    <img src="${this.src}" alt="推理结果预览">
                </div>
            `;
            document.body.appendChild(modal);

            // 点击关闭预览
            modal.addEventListener('click', function(e) {
                if (e.target === modal || e.target.className === 'close-preview') {
                    modal.remove();
                }
            });
            
            // 动画效果
            setTimeout(() => {
                modal.style.opacity = '1';
            }, 10);
        });
    }
}
