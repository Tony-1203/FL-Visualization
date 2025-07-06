"""
Supabase数据库连接和操作测试
"""

import pytest
import os
import sys
from unittest.mock import patch, Mock, MagicMock
import json

# 确保可以导入app模块
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import (
    app,
    SUPABASE_AVAILABLE,
    get_user_from_supabase,
    create_user_in_supabase,
    hash_password,
    verify_password,
)


class TestSupabaseIntegration:
    """测试Supabase集成功能"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.test_user = {
            "username": "test_user",
            "email": "test@example.com",
            "role": "client",
            "password_hash": hash_password("test_password"),
        }

    def test_supabase_environment_variables(self):
        """测试Supabase环境变量配置"""
        # 检查环境变量是否存在
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if SUPABASE_AVAILABLE:
            assert url is not None, "SUPABASE_URL环境变量未设置"
            assert key is not None, "SUPABASE_KEY环境变量未设置"
            assert url.startswith("https://"), "SUPABASE_URL格式不正确"
            print("✅ Supabase环境变量配置正确")
        else:
            print("⚠️ Supabase未配置，使用本地存储")

    @patch("app.supabase")
    def test_get_user_from_supabase_success(self, mock_supabase):
        """测试从Supabase成功获取用户"""
        # 模拟成功的数据库响应
        mock_response = Mock()
        mock_response.data = [self.test_user]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
            mock_response
        )

        with patch("app.SUPABASE_AVAILABLE", True):
            result = get_user_from_supabase("test_user")

            assert result is not None
            assert result["username"] == "test_user"
            assert result["email"] == "test@example.com"
            assert result["role"] == "client"
            print("✅ Supabase用户查询成功")

    @patch("app.supabase")
    def test_get_user_from_supabase_not_found(self, mock_supabase):
        """测试从Supabase查找不存在的用户"""
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
            mock_response
        )

        with patch("app.SUPABASE_AVAILABLE", True):
            result = get_user_from_supabase("nonexistent_user")

            assert result is None
            print("✅ 不存在用户查询处理正确")

    @patch("app.supabase")
    def test_save_user_to_supabase_success(self, mock_supabase):
        """测试向Supabase保存用户成功"""
        mock_response = Mock()
        mock_response.data = [self.test_user]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = (
            mock_response
        )

        with patch("app.SUPABASE_AVAILABLE", True):
            result = create_user_in_supabase(
                username="test_user",
                email="test@example.com",
                password="test_password_123",
                role="client",
            )

            assert result is True
            print("✅ Supabase用户保存成功")

    @patch("app.supabase")
    def test_supabase_connection_error(self, mock_supabase):
        """测试Supabase连接错误处理"""
        # 模拟数据库连接异常
        mock_supabase.table.side_effect = Exception("Connection failed")

        with patch("app.SUPABASE_AVAILABLE", True):
            result = get_user_from_supabase("test_user")

            # 在连接失败时应该返回None或进行适当的错误处理
            assert result is None or isinstance(result, dict)
            print("✅ Supabase连接错误处理正确")

    def test_password_hashing(self):
        """测试密码哈希功能"""
        password = "test_password_123"
        hashed = hash_password(password)

        assert hashed is not None
        assert isinstance(hashed, str)
        assert len(hashed) > 0
        assert hashed != password  # 确保密码被哈希

        # 测试密码验证
        assert verify_password(password, hashed) == True
        assert verify_password("wrong_password", hashed) == False
        print("✅ 密码哈希和验证功能正常")

    def test_supabase_fallback_to_local(self):
        """测试Supabase不可用时的本地存储回退"""
        with patch("app.SUPABASE_AVAILABLE", False):
            # 当Supabase不可用时，应该使用本地存储
            result = get_user_from_supabase("test_user")

            # 这里的行为取决于你的实现
            # 通常应该回退到本地JSON文件
            print("✅ Supabase回退机制测试")


class TestSupabaseDataOperations:
    """测试Supabase数据操作"""

    @patch("app.supabase")
    def test_user_registration_flow(self, mock_supabase):
        """测试用户注册流程"""
        # 模拟注册新用户
        mock_response = Mock()
        mock_response.data = []  # 用户不存在
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
            mock_response
        )

        # 模拟插入操作
        insert_response = Mock()
        insert_response.data = [{"username": "new_user", "email": "new@example.com"}]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = (
            insert_response
        )

        with patch("app.SUPABASE_AVAILABLE", True):
            # 检查用户是否存在
            existing_user = get_user_from_supabase("new_user")
            assert existing_user is None

            # 保存新用户
            result = create_user_in_supabase(
                username="new_user",
                email="new@example.com",
                password="password123",
                role="client",
            )
            assert result is True
            print("✅ 用户注册流程测试通过")

    @patch("app.supabase")
    def test_training_data_storage(self, mock_supabase):
        """测试训练数据存储到Supabase"""
        mock_response = Mock()
        mock_response.data = [
            {"id": 1, "session_id": "test_session", "status": "completed"}
        ]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = (
            mock_response
        )

        with patch("app.SUPABASE_AVAILABLE", True):
            # 这里测试训练数据的存储逻辑
            print("✅ 训练数据存储测试预留")

    @patch("app.supabase")
    def test_model_metadata_storage(self, mock_supabase):
        """测试模型元数据存储"""
        mock_response = Mock()
        mock_response.data = [{"id": 1, "model_name": "test_model", "accuracy": 0.95}]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = (
            mock_response
        )

        with patch("app.SUPABASE_AVAILABLE", True):
            # 这里测试模型元数据的存储逻辑
            print("✅ 模型元数据存储测试预留")


class TestSupabaseErrorHandling:
    """测试Supabase错误处理"""

    @patch("app.supabase")
    def test_network_timeout(self, mock_supabase):
        """测试网络超时处理"""
        mock_supabase.table.side_effect = TimeoutError("Network timeout")

        with patch("app.SUPABASE_AVAILABLE", True):
            result = get_user_from_supabase("test_user")
            assert result is None
            print("✅ 网络超时处理正确")

    @patch("app.supabase")
    def test_authentication_error(self, mock_supabase):
        """测试认证错误处理"""
        mock_supabase.table.side_effect = Exception("Authentication failed")

        with patch("app.SUPABASE_AVAILABLE", True):
            result = get_user_from_supabase("test_user")
            assert result is None
            print("✅ 认证错误处理正确")

    @patch("app.supabase")
    def test_rate_limit_handling(self, mock_supabase):
        """测试速率限制处理"""
        mock_supabase.table.side_effect = Exception("Rate limit exceeded")

        with patch("app.SUPABASE_AVAILABLE", True):
            result = get_user_from_supabase("test_user")
            assert result is None
            print("✅ 速率限制处理正确")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
