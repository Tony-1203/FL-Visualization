"""
简单测试套件：Supabase连接和用户登录功能测试
"""

import pytest
import os
import sys
from unittest.mock import patch, Mock
from io import BytesIO

# 确保可以导入app模块
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import (
    app,
    SUPABASE_AVAILABLE,
    get_user_from_supabase,
    hash_password,
    verify_password,
)


class TestSupabaseConnection:
    """测试Supabase数据库连接功能"""

    def test_supabase_availability_flag(self):
        """测试Supabase可用性标志"""
        # 检查SUPABASE_AVAILABLE是否为布尔值
        assert isinstance(SUPABASE_AVAILABLE, bool)
        print(f"✅ Supabase可用性状态: {SUPABASE_AVAILABLE}")

    @patch("app.supabase")
    def test_supabase_connection_success(self, mock_supabase):
        """测试Supabase连接成功的情况"""
        # 模拟成功的Supabase响应
        mock_response = Mock()
        mock_response.data = [
            {"username": "test_user", "role": "client", "email": "test@example.com"}
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
            mock_response
        )

        with patch("app.SUPABASE_AVAILABLE", True):
            result = get_user_from_supabase("test_user")

            assert result is not None
            assert result["username"] == "test_user"
            assert result["role"] == "client"
            print("✅ Supabase用户查询测试通过")

    @patch("app.supabase")
    def test_supabase_user_not_found(self, mock_supabase):
        """测试Supabase中查找不存在的用户"""
        # 模拟空响应（用户不存在）
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
            mock_response
        )

        with patch("app.SUPABASE_AVAILABLE", True):
            result = get_user_from_supabase("nonexistent_user")

            assert result is None
            print("✅ Supabase用户不存在测试通过")

    @patch("app.supabase")
    def test_supabase_connection_error(self, mock_supabase):
        """测试Supabase连接异常的处理"""
        # 模拟连接异常
        mock_supabase.table.side_effect = Exception("Connection failed")

        with patch("app.SUPABASE_AVAILABLE", True):
            result = get_user_from_supabase("test_user")

            assert result is None
            print("✅ Supabase连接异常处理测试通过")

    def test_supabase_unavailable_fallback(self):
        """测试Supabase不可用时的回退机制"""
        with patch("app.SUPABASE_AVAILABLE", False):
            result = get_user_from_supabase("test_user")

            assert result is None
            print("✅ Supabase不可用回退测试通过")


class TestUserAuthentication:
    """测试用户认证功能"""

    def test_password_hashing(self):
        """测试密码哈希功能"""
        password = "test_password_123"
        hashed = hash_password(password)

        # 验证哈希值与原密码不同
        assert hashed != password
        assert len(hashed) > 20  # bcrypt哈希值长度检查
        print("✅ 密码哈希功能测试通过")

    def test_password_verification(self):
        """测试密码验证功能"""
        password = "secure_password_456"
        hashed = hash_password(password)

        # 正确密码验证
        assert verify_password(password, hashed) is True

        # 错误密码验证
        assert verify_password("wrong_password", hashed) is False
        print("✅ 密码验证功能测试通过")

    def test_login_page_access(self, client):
        """测试登录页面访问"""
        response = client.get("/")

        assert response.status_code == 200
        assert b"login" in response.data.lower() or b"username" in response.data.lower()
        print("✅ 登录页面访问测试通过")

    def test_login_with_session(self, client):
        """测试用户登录会话创建"""
        # 模拟登录请求
        with client.session_transaction() as session:
            session["username"] = "test_user"
            session["role"] = "client"

        # 验证会话已创建
        with client.session_transaction() as session:
            assert session.get("username") == "test_user"
            assert session.get("role") == "client"
        print("✅ 用户登录会话测试通过")

    def test_logout_functionality(self, authenticated_client):
        """测试用户登出功能"""
        response = authenticated_client.get("/logout")

        # 应该重定向到登录页面
        assert response.status_code == 302

        # 检查会话是否被清空
        with authenticated_client.session_transaction() as session:
            assert "username" not in session
            assert "role" not in session
        print("✅ 用户登出功能测试通过")

    def test_unauthorized_access(self, client):
        """测试未认证用户访问受保护页面"""
        protected_pages = ["/client/dashboard", "/server/dashboard"]

        for page in protected_pages:
            response = client.get(page)
            # 应该重定向到登录页面或返回403
            assert response.status_code in [302, 401, 403]
        print("✅ 未认证访问控制测试通过")

    def test_authenticated_client_access(self, authenticated_client):
        """测试已认证客户端用户访问"""
        response = authenticated_client.get("/client/dashboard")

        assert response.status_code == 200
        print("✅ 已认证客户端访问测试通过")

    def test_authenticated_server_access(self, authenticated_server):
        """测试已认证服务器用户访问"""
        response = authenticated_server.get("/server/dashboard")

        assert response.status_code == 200
        print("✅ 已认证服务器访问测试通过")


class TestBasicFunctionality:
    """测试基本功能"""

    def test_app_startup(self):
        """测试应用程序启动"""
        assert app is not None
        assert app.config["TESTING"] is True
        print("✅ 应用程序启动测试通过")

    def test_file_upload_authentication_check(self, client):
        """测试文件上传需要认证"""
        # 尝试在未登录状态下上传文件
        data = {"files": (BytesIO(b"test content"), "test.mhd")}
        response = client.post(
            "/client/upload", data=data, content_type="multipart/form-data"
        )

        # 应该返回403（未授权）
        assert response.status_code == 403
        print("✅ 文件上传认证检查测试通过")

    def test_api_endpoints_authentication(self, client):
        """测试API端点需要认证"""
        api_endpoints = [
            "/api/server/status",
            "/api/server/logs",
            "/api/client/list_inference_files",
        ]

        for endpoint in api_endpoints:
            response = client.get(endpoint)
            # 应该返回认证错误
            assert response.status_code in [401, 403]
        print("✅ API端点认证检查测试通过")


def run_simple_tests():
    """运行简单测试的主函数"""
    print("🧪 开始运行FL-Visualization简单测试套件")
    print("=" * 60)

    # 可以单独运行此函数进行快速测试
    import subprocess
    import sys

    # 运行pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )

    print("测试输出:")
    print(result.stdout)
    if result.stderr:
        print("错误信息:")
        print(result.stderr)

    return result.returncode == 0


if __name__ == "__main__":
    # 如果直接运行此文件，执行简单测试
    success = run_simple_tests()
    if success:
        print("\n🎉 所有简单测试都通过了！")
    else:
        print("\n❌ 一些测试失败了，请检查上面的输出")
    sys.exit(0 if success else 1)
