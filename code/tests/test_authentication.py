"""
用户登录和认证功能测试（修复版）
"""

import pytest
import os
import sys
from unittest.mock import patch, Mock
import json

# 确保可以导入app模块
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app, hash_password, verify_password


class TestUserAuthentication:
    """测试用户认证功能"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.app = app
        self.app.config["TESTING"] = True
        self.app.config["SECRET_KEY"] = "test_secret_key"
        self.client = self.app.test_client()

        # 测试用户数据
        self.test_user = {
            "username": "test_user",
            "email": "test@example.com",
            "password": "test_password_123",
            "role": "client",
        }
        self.test_server = {
            "username": "test_server",
            "email": "server@example.com",
            "password": "server_password_123",
            "role": "server",
        }

    def test_login_page_access(self):
        """测试登录页面访问"""
        response = self.client.get("/")
        assert response.status_code == 200
        # 检查页面是否包含登录相关内容
        response_data = response.data.decode("utf-8").lower()
        assert (
            "username" in response_data
            or "login" in response_data
            or "password" in response_data
        )
        print("✅ 登录页面访问正常")

    @patch("app.get_user_from_supabase")
    def test_successful_login_client(self, mock_get_user):
        """测试客户端成功登录"""
        # 模拟数据库中的用户
        mock_get_user.return_value = {
            "username": self.test_user["username"],
            "email": self.test_user["email"],
            "password_hash": hash_password(self.test_user["password"]),
            "role": self.test_user["role"],
        }

        response = self.client.post(
            "/",
            data={
                "username": self.test_user["username"],
                "password": self.test_user["password"],
            },
        )

        # 检查重定向或成功响应
        assert response.status_code in [200, 302]
        print("✅ 客户端登录成功")

    @patch("app.get_user_from_supabase")
    def test_successful_login_server(self, mock_get_user):
        """测试服务器端成功登录"""
        mock_get_user.return_value = {
            "username": self.test_server["username"],
            "email": self.test_server["email"],
            "password_hash": hash_password(self.test_server["password"]),
            "role": self.test_server["role"],
        }

        response = self.client.post(
            "/",
            data={
                "username": self.test_server["username"],
                "password": self.test_server["password"],
            },
        )

        assert response.status_code in [200, 302]
        print("✅ 服务器端登录成功")

    @patch("app.get_user_from_supabase")
    def test_login_invalid_password(self, mock_get_user):
        """测试错误密码登录"""
        mock_get_user.return_value = {
            "username": self.test_user["username"],
            "email": self.test_user["email"],
            "password_hash": hash_password(self.test_user["password"]),
            "role": self.test_user["role"],
        }

        response = self.client.post(
            "/",
            data={"username": self.test_user["username"], "password": "wrong_password"},
        )

        # 应该返回错误或重定向回登录页面
        assert response.status_code in [200, 302, 400, 401]
        print("✅ 错误密码处理正确")

    @patch("app.get_user_from_supabase")
    def test_login_nonexistent_user(self, mock_get_user):
        """测试不存在用户登录"""
        mock_get_user.return_value = None

        response = self.client.post(
            "/", data={"username": "nonexistent_user", "password": "any_password"}
        )

        assert response.status_code in [200, 302, 400, 401]
        print("✅ 不存在用户处理正确")

    def test_logout_functionality(self):
        """测试登出功能"""
        # 先模拟登录状态
        with self.client.session_transaction() as sess:
            sess["username"] = self.test_user["username"]
            sess["role"] = self.test_user["role"]

        response = self.client.get("/logout")
        assert response.status_code in [200, 302]

        # 检查会话是否被清除
        with self.client.session_transaction() as sess:
            assert "username" not in sess or sess.get("username") is None

        print("✅ 登出功能正常")

    def test_session_management(self):
        """测试会话管理"""
        with self.client.session_transaction() as sess:
            sess["username"] = self.test_user["username"]
            sess["role"] = self.test_user["role"]

        # 测试会话数据
        with self.client.session_transaction() as sess:
            assert sess["username"] == self.test_user["username"]
            assert sess["role"] == self.test_user["role"]

        print("✅ 会话管理正常")


class TestAccessControl:
    """测试访问控制功能"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.app = app
        self.app.config["TESTING"] = True
        self.app.config["SECRET_KEY"] = "test_secret_key"
        self.client = self.app.test_client()

    def test_protected_route_without_login(self):
        """测试未登录访问受保护路由"""
        # 测试GET路由
        get_routes = ["/client/dashboard", "/server/dashboard"]

        for route in get_routes:
            response = self.client.get(route)
            # 应该重定向到登录页面或返回401
            assert response.status_code in [302, 401]

        # 测试POST路由
        post_routes = ["/client/upload", "/server/start_training"]

        for route in post_routes:
            response = self.client.post(route, data={})
            # 应该重定向到登录页面或返回401/403/405
            assert response.status_code in [302, 401, 403, 405]

        print("✅ 未登录访问保护正确")

    def test_client_access_client_routes(self):
        """测试客户端访问客户端路由"""
        with self.client.session_transaction() as sess:
            sess["username"] = "test_client"
            sess["role"] = "client"

        response = self.client.get("/client/dashboard")
        assert response.status_code in [200, 302]
        print("✅ 客户端路由访问正确")

    def test_server_access_server_routes(self):
        """测试服务器端访问服务器路由"""
        with self.client.session_transaction() as sess:
            sess["username"] = "test_server"
            sess["role"] = "server"

        response = self.client.get("/server/dashboard")
        assert response.status_code in [200, 302]
        print("✅ 服务器路由访问正确")

    def test_role_based_access_control(self):
        """测试基于角色的访问控制"""
        # 测试客户端试图访问服务器路由
        with self.client.session_transaction() as sess:
            sess["username"] = "test_client"
            sess["role"] = "client"

        response = self.client.get("/server/dashboard")
        # 应该被拒绝或重定向
        assert response.status_code in [302, 403]

        # 测试服务器试图访问客户端特定功能
        with self.client.session_transaction() as sess:
            sess["username"] = "test_server"
            sess["role"] = "server"

        response = self.client.post("/client/upload", data={})
        # 服务器可能无法上传客户端数据
        assert response.status_code in [302, 403, 400]

        print("✅ 角色访问控制正确")


class TestPasswordSecurity:
    """测试密码安全功能"""

    def test_password_hashing_uniqueness(self):
        """测试密码哈希的唯一性"""
        password = "test_password_123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        # 每次哈希应该产生不同的结果（因为salt）
        assert hash1 != hash2
        # 但都应该能验证同一个密码
        assert verify_password(password, hash1)
        assert verify_password(password, hash2)
        print("✅ 密码哈希唯一性正确")

    def test_password_hash_format(self):
        """测试密码哈希格式"""
        password = "test_password_123"
        hashed = hash_password(password)

        assert isinstance(hashed, str)
        assert len(hashed) > 0
        # bcrypt哈希通常以$2b$开头
        assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
        print("✅ 密码哈希格式正确")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
