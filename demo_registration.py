#!/usr/bin/env python3
"""
演示用户注册功能
"""

import os
import sys
import requests
import json


def test_registration():
    """测试用户注册功能"""
    base_url = "http://127.0.0.1:5000"

    print("🔍 测试用户注册功能")
    print("=" * 50)

    # 测试数据
    test_users = [
        {
            "username": "test_client",
            "email": "test@example.com",
            "password": "password123",
            "confirm_password": "password123",
        },
        {
            "username": "doctor_zhang",
            "email": "zhang@hospital.com",
            "password": "secure_pass",
            "confirm_password": "secure_pass",
        },
    ]

    for user_data in test_users:
        print(f"\n📝 尝试注册用户: {user_data['username']}")

        try:
            # 发送注册请求
            response = requests.post(
                f"{base_url}/register", data=user_data, allow_redirects=False
            )

            if response.status_code == 302:  # 重定向到登录页面
                print(f"✅ {user_data['username']}: 注册成功")

                # 测试登录
                login_response = requests.post(
                    f"{base_url}/",
                    data={
                        "username": user_data["username"],
                        "password": user_data["password"],
                    },
                    allow_redirects=False,
                )

                if login_response.status_code == 302:
                    print(f"✅ {user_data['username']}: 登录测试成功")
                else:
                    print(f"❌ {user_data['username']}: 登录测试失败")

            else:
                print(
                    f"❌ {user_data['username']}: 注册失败 - HTTP {response.status_code}"
                )

        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到服务器，请确保应用程序正在运行")
            break
        except Exception as e:
            print(f"❌ 注册请求失败: {e}")

    print("\n" + "=" * 50)
    print("🎯 注册测试完成")


def show_current_users():
    """显示当前用户列表"""
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from app import load_local_users

    print("\n👥 当前用户列表:")
    print("-" * 30)

    try:
        users = load_local_users()
        for username, user_data in users.items():
            print(f"👤 {username}")
            print(f"   📧 邮箱: {user_data.get('email', '未设置')}")
            print(f"   🎭 角色: {user_data.get('role', '未知')}")
            print(f"   📅 创建时间: {user_data.get('created_at', '未知')}")
            print()
    except Exception as e:
        print(f"❌ 无法加载用户列表: {e}")


if __name__ == "__main__":
    print("请确保Flask应用程序正在运行在 http://127.0.0.1:5000")
    input("按Enter键继续...")

    test_registration()
    show_current_users()
