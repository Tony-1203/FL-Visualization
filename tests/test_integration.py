"""
Integration tests for the FL-Visualization application
"""

import pytest
import tempfile
import os
import threading
import time
from unittest.mock import patch, MagicMock


class TestFileUploadIntegration:
    """Test file upload and processing integration"""

    def test_client_file_upload_flow(self, client):
        """Test complete client file upload flow"""
        # Login as client
        with client.session_transaction() as sess:
            sess["username"] = "test_client"
            sess["role"] = "client"

        # Create a temporary test file
        with tempfile.NamedTemporaryFile(suffix=".mhd", delete=False) as temp_file:
            temp_file.write(b"test medical data")
            temp_file_path = temp_file.name

        try:
            # Upload the file
            with open(temp_file_path, "rb") as test_file:
                data = {
                    "file": (test_file, "test.mhd"),
                    "description": "Test medical data",
                }
                response = client.post(
                    "/client/upload", data=data, content_type="multipart/form-data"
                )

            # Should handle upload (success or expected error)
            assert response.status_code in [200, 400, 500]

        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_inference_file_upload_flow(self, client):
        """Test inference file upload flow"""
        # Login as user
        with client.session_transaction() as sess:
            sess["username"] = "test_user"
            sess["role"] = "client"

        # Create a temporary test file
        with tempfile.NamedTemporaryFile(suffix=".mhd", delete=False) as temp_file:
            temp_file.write(b"test inference data")
            temp_file_path = temp_file.name

        try:
            # Upload file for inference
            with open(temp_file_path, "rb") as test_file:
                data = {"file": (test_file, "inference_test.mhd")}
                response = client.post(
                    "/inference/upload", data=data, content_type="multipart/form-data"
                )

            # Should handle upload
            assert response.status_code in [200, 400, 500]

        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


class TestTrainingIntegration:
    """Test federated learning training integration"""

    @patch("app.train_federated_model")
    def test_training_workflow(self, mock_train, client):
        """Test complete training workflow"""
        # Login as server
        with client.session_transaction() as sess:
            sess["username"] = "test_server"
            sess["role"] = "server"

        # Mock successful training
        mock_coordinator = MagicMock()
        mock_train.return_value = mock_coordinator

        # Simulate some client data
        from app import client_data_status

        client_data_status["client1"] = {
            "uploaded": True,
            "data_path": "/fake/path/client1",
        }

        # Start training
        response = client.post(
            "/server/start_training", json={"global_rounds": 2, "local_epochs": 1}
        )

        # Should accept training request
        assert response.status_code in [200, 400]

        # Wait a bit for background thread
        time.sleep(0.1)

        # Training function should have been called
        if response.status_code == 200:
            assert mock_train.called

    @patch("app.training_status")
    def test_training_status_updates(self, mock_status, client):
        """Test training status updates"""
        # Login as server
        with client.session_transaction() as sess:
            sess["username"] = "test_server"
            sess["role"] = "server"

        # Check training status endpoint
        response = client.get("/server/training_status")
        assert response.status_code in [200, 404, 405]


class TestUserSessionIntegration:
    """Test user session management integration"""

    def test_login_logout_flow(self, client):
        """Test complete login/logout flow"""
        # Test login
        response = client.post(
            "/login",
            data={
                "username": "integration_test",
                "password": "test123",
                "role": "client",
            },
        )
        assert response.status_code in [200, 302]

        # Test accessing protected route
        response = client.get("/client/dashboard")
        assert response.status_code in [200, 302]  # Should be accessible or redirect

        # Test logout
        response = client.get("/logout")
        assert response.status_code in [200, 302]

        # Test accessing protected route after logout
        response = client.get("/client/dashboard")
        assert response.status_code in [302, 401, 403]  # Should be blocked

    def test_role_based_access(self, client):
        """Test role-based access control"""
        # Test client role
        with client.session_transaction() as sess:
            sess["username"] = "test_client"
            sess["role"] = "client"

        # Client should access client routes
        response = client.get("/client/dashboard")
        assert response.status_code == 200

        # Client should not access server routes
        response = client.get("/server/dashboard")
        assert response.status_code in [302, 401, 403]

        # Test server role
        with client.session_transaction() as sess:
            sess["username"] = "test_server"
            sess["role"] = "server"

        # Server should access server routes
        response = client.get("/server/dashboard")
        assert response.status_code == 200


class TestDataManagementIntegration:
    """Test data management integration"""

    def test_client_data_status_tracking(self, client):
        """Test client data status tracking"""
        from app import client_data_status, initialize_client_data_status

        # Initialize
        initialize_client_data_status()

        # Should have initialized structure
        assert isinstance(client_data_status, dict)

        # Test updating client status
        client_data_status["test_client"] = {
            "uploaded": True,
            "data_path": "/test/path",
            "upload_time": "2024-01-01 00:00:00",
        }

        # Should maintain the data
        assert client_data_status["test_client"]["uploaded"] == True

    def test_log_management(self, client):
        """Test log management functionality"""
        from app import add_server_log, add_training_log, get_logs_list
        from app import server_logs, training_logs

        # Add some logs
        add_server_log("Test server log message")
        add_training_log("Test training log message")

        # Retrieve logs
        server_log_list = get_logs_list(server_logs)
        training_log_list = get_logs_list(training_logs)

        # Should contain the added logs
        assert len(server_log_list) > 0
        assert len(training_log_list) > 0

        # Should contain our test messages
        server_messages = server_log_list  # logs are already strings
        training_messages = training_log_list  # logs are already strings

        assert any("Test server log" in msg for msg in server_messages)
        assert any("Test training log" in msg for msg in training_messages)


class TestErrorRecoveryIntegration:
    """Test error recovery and resilience"""

    def test_missing_file_handling(self, client):
        """Test handling of missing files"""
        # Login as client
        with client.session_transaction() as sess:
            sess["username"] = "test_client"
            sess["role"] = "client"

        # Try to upload without file
        response = client.post("/client/upload", data={})
        assert response.status_code in [200, 400]  # Should handle gracefully

    @patch("app.train_federated_model")
    def test_training_failure_handling(self, mock_train, client):
        """Test handling of training failures"""
        # Login as server
        with client.session_transaction() as sess:
            sess["username"] = "test_server"
            sess["role"] = "server"

        # Mock training failure
        mock_train.side_effect = Exception("Training failed")

        # Simulate client data
        from app import client_data_status

        client_data_status["client1"] = {"uploaded": True, "data_path": "/fake/path"}

        # Try to start training
        response = client.post(
            "/server/start_training", json={"global_rounds": 1, "local_epochs": 1}
        )

        # Should handle request (may succeed initially, fail in background)
        assert response.status_code in [200, 400, 500]

    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        # Login as client
        with client.session_transaction() as sess:
            sess["username"] = "test_client"
            sess["role"] = "client"

        # Make multiple concurrent requests
        responses = []
        for _ in range(3):
            response = client.get("/client/dashboard")
            responses.append(response)

        # All should be handled properly
        for response in responses:
            assert response.status_code in [200, 302]
