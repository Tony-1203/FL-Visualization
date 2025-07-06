#!/usr/bin/env python3
"""
FL-Visualization项目完整测试套件
运行所有测试包括Supabase连接、用户认证、文件上传等核心功能
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    """检查测试依赖"""
    required_packages = [
        "pytest",
        "pytest-mock",
        "flask",
        "bcrypt",
        "supabase",
        "python-dotenv",
    ]

    missing = []
    for package in required_packages:
        try:
            if package == "python-dotenv":
                import dotenv
            elif package == "pytest-mock":
                import pytest_mock
            else:
                importlib.import_module(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ 缺少依赖包: {', '.join(missing)}")
        print("正在安装...")
        for package in missing:
            subprocess.run([sys.executable, "-m", "pip", "install", package])
        print("✅ 依赖安装完成")
    else:
        print("✅ 所有依赖包已安装")


def run_test_suite(test_type="all"):
    """运行测试套件"""

    # 检查依赖
    check_dependencies()

    print("\n" + "=" * 60)
    print("🧪 FL-Visualization 测试套件")
    print("=" * 60)

    test_commands = {
        "supabase": [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_supabase.py",
            "-v",
            "--tb=short",
        ],
        "auth": [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_authentication.py",
            "-v",
            "--tb=short",
        ],
        "upload": [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_file_upload.py",
            "-v",
            "--tb=short",
        ],
        "simple": [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_simple.py",
            "-v",
            "--tb=short",
        ],
        "all": [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--maxfail=10",
        ],
    }

    if test_type not in test_commands:
        print(f"❌ 未知的测试类型: {test_type}")
        print(f"可用类型: {', '.join(test_commands.keys())}")
        return False

    print(f"\n🚀 运行 {test_type} 测试...")
    print("-" * 40)

    try:
        result = subprocess.run(
            test_commands[test_type], cwd=project_root, capture_output=False, text=True
        )

        if result.returncode == 0:
            print(f"\n✅ {test_type} 测试全部通过!")
            return True
        else:
            print(f"\n❌ {test_type} 测试中有失败项")
            return False

    except Exception as e:
        print(f"❌ 运行测试时出错: {e}")
        return False


def run_quick_tests():
    """运行快速测试"""
    print("\n运行快速测试...")

    quick_tests = [
        (
            "Supabase连接",
            "tests/test_supabase.py::TestSupabaseIntegration::test_supabase_environment_variables",
        ),
        (
            "密码哈希",
            "tests/test_supabase.py::TestSupabaseIntegration::test_password_hashing",
        ),
        (
            "登录页面",
            "tests/test_authentication.py::TestUserAuthentication::test_login_page_access",
        ),
        (
            "访问控制",
            "tests/test_authentication.py::TestAccessControl::test_protected_route_without_login",
        ),
        (
            "文件上传页面",
            "tests/test_file_upload.py::TestFileUpload::test_upload_page_access_unauthenticated",
        ),
    ]

    passed = 0
    total = len(quick_tests)

    for test_name, test_path in quick_tests:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-v", "--tb=line"],
                cwd=project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(f"✅ {test_name}")
                passed += 1
            else:
                print(f"❌ {test_name}")

        except Exception as e:
            print(f"❌ {test_name} (错误: {e})")

    print(f"\n📊 快速测试结果: {passed}/{total} 通过")
    return passed == total


def show_test_coverage():
    """显示测试覆盖情况"""
    print("\n📋 测试覆盖情况:")
    print("-" * 40)

    test_areas = [
        (
            "🔗 Supabase数据库连接",
            "tests/test_supabase.py",
            [
                "环境变量配置",
                "用户查询功能",
                "用户保存功能",
                "连接错误处理",
                "密码哈希验证",
            ],
        ),
        (
            "🔐 用户认证功能",
            "tests/test_authentication.py",
            ["登录页面访问", "成功登录流程", "密码验证", "会话管理", "角色权限控制"],
        ),
        (
            "📁 文件上传功能",
            "tests/test_file_upload.py",
            [
                "文件上传权限",
                "文件格式验证",
                "文件大小限制",
                "路径安全检查",
                "并发上传处理",
            ],
        ),
        (
            "🧪 基础功能测试",
            "tests/test_simple.py",
            ["应用启动", "基本路由", "配置加载"],
        ),
    ]

    for area_name, test_file, features in test_areas:
        print(f"\n{area_name}")
        if os.path.exists(os.path.join(project_root, test_file)):
            print(f"  📄 测试文件: {test_file}")
            for feature in features:
                print(f"    • {feature}")
        else:
            print(f"  ❌ 测试文件不存在: {test_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="FL-Visualization测试套件")
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["all", "supabase", "auth", "upload", "simple", "quick", "coverage"],
        help="要运行的测试类型",
    )
    parser.add_argument("--quick", action="store_true", help="运行快速测试")
    parser.add_argument("--coverage", action="store_true", help="显示测试覆盖情况")

    args = parser.parse_args()

    if args.coverage or args.test_type == "coverage":
        show_test_coverage()
        return

    if args.quick or args.test_type == "quick":
        success = run_quick_tests()
    else:
        success = run_test_suite(args.test_type)

    if not success:
        sys.exit(1)

    print("\n🎉 测试完成!")


if __name__ == "__main__":
    main()
