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

// 页面加载完成后初始化增强UI
document.addEventListener('DOMContentLoaded', function() {
    enhanceInferenceUI();
});

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
