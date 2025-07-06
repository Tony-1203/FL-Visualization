import pytest
import os
import tempfile
import sys
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app


@pytest.fixture
def client():
    """Flask test client fixture"""
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-key"
    app.config["WTF_CSRF_ENABLED"] = False

    with app.test_client() as client:
        with app.app_context():
            yield client


@pytest.fixture
def temp_upload_dir():
    """Temporary upload directory fixture"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil

    shutil.rmtree(temp_dir)


@pytest.fixture
def authenticated_client(client):
    """Authenticated client session"""
    with client.session_transaction() as session:
        session["username"] = "test_client"
        session["role"] = "client"
    return client


@pytest.fixture
def authenticated_server_client(client):
    """Authenticated server session"""
    with client.session_transaction() as session:
        session["username"] = "test_server"
        session["role"] = "server"
    return client


@pytest.fixture
def mock_supabase():
    """Mock Supabase client"""
    with patch("app.supabase") as mock:
        yield mock


@pytest.fixture
def test_user_data():
    """Test user data"""
    return {
        "username": "test_user",
        "email": "test@example.com",
        "password": "test_password_123",
        "role": "client",
    }


@pytest.fixture
def test_server_data():
    """Test server data"""
    return {
        "username": "test_server",
        "email": "server@example.com",
        "password": "server_password_123",
        "role": "server",
    }


@pytest.fixture
def authenticated_server(client):
    """Authenticated server session"""
    with client.session_transaction() as session:
        session["username"] = "server"
        session["role"] = "server"
    return client


@pytest.fixture
def mock_supabase():
    """Mock Supabase client fixture"""
    mock_client = Mock()
    mock_table = Mock()
    mock_client.table.return_value = mock_table
    return mock_client, mock_table
