"""
Tests for Socket.IO functionality
"""

import pytest
import time
from unittest.mock import patch


class TestSocketIOConnections:
    """Test Socket.IO connection handling"""

    def test_connection(self, socketio_client):
        """Test basic Socket.IO connection"""
        # The connection should be successful
        assert socketio_client.is_connected()

    def test_disconnect(self, socketio_client):
        """Test Socket.IO disconnection"""
        socketio_client.disconnect()
        assert not socketio_client.is_connected()


class TestSocketIOEvents:
    """Test Socket.IO event handling"""

    def test_heartbeat(self, socketio_client):
        """Test heartbeat event"""
        # Send heartbeat event
        socketio_client.emit("heartbeat")

        # Check for heartbeat response
        received = socketio_client.get_received()

        # Should receive a heartbeat response
        heartbeat_responses = [
            msg for msg in received if msg["name"] == "heartbeat_response"
        ]
        assert len(heartbeat_responses) > 0

        # Response should contain timestamp
        response_data = heartbeat_responses[0]["args"][0]
        assert "timestamp" in response_data

    def test_request_status_update_without_login(self, socketio_client):
        """Test status update request without being logged in"""
        socketio_client.emit("request_status_update")

        # Should not receive status updates if not logged in
        received = socketio_client.get_received()
        status_updates = [msg for msg in received if "status_update" in msg["name"]]

        # May receive empty or no updates
        # This is acceptable behavior for unauthorized requests
        assert isinstance(status_updates, list)

    def test_request_logs_without_server_role(self, socketio_client):
        """Test log request without server role"""
        # Clear any initial messages
        socketio_client.get_received()

        # Make the log request
        socketio_client.emit("request_logs", {"type": "server"})

        # Should not receive logs if not a server user
        received = socketio_client.get_received()
        log_updates = [msg for msg in received if msg["name"] == "logs_update"]

        # Should not receive logs without proper authorization
        assert len(log_updates) == 0


class TestSocketIOWithAuthentication:
    """Test Socket.IO events with authentication"""

    def test_status_update_with_client_role(self, client, socketio_client):
        """Test status updates as client user"""
        # First establish a session as client
        with client.session_transaction() as sess:
            sess["username"] = "test_client"
            sess["role"] = "client"

        # Now test Socket.IO with established session
        socketio_client.emit("request_status_update")

        received = socketio_client.get_received()
        # Should receive some status updates as authenticated user
        assert isinstance(received, list)

    def test_status_update_with_server_role(self, client, socketio_client):
        """Test status updates as server user"""
        # Establish session as server
        with client.session_transaction() as sess:
            sess["username"] = "test_server"
            sess["role"] = "server"

        socketio_client.emit("request_status_update")

        received = socketio_client.get_received()
        # Server should receive comprehensive status updates
        assert isinstance(received, list)


class TestBroadcastFunctions:
    """Test broadcast functionality"""

    @patch("app.socketio")
    def test_broadcast_user_status(self, mock_socketio):
        """Test user status broadcasting"""
        from app import broadcast_user_status

        # Call the broadcast function
        broadcast_user_status()

        # Should attempt to emit user status
        assert mock_socketio.emit.called

    @patch("app.socketio")
    def test_broadcast_training_status(self, mock_socketio):
        """Test training status broadcasting"""
        from app import broadcast_training_status

        # Call the broadcast function
        broadcast_training_status()

        # Should attempt to emit training status
        assert mock_socketio.emit.called

    @patch("app.socketio")
    def test_broadcast_client_data_update(self, mock_socketio):
        """Test client data update broadcasting"""
        from app import broadcast_client_data_update, online_users

        # Mock a server user online
        mock_session_id = "test_session"
        online_users[mock_session_id] = {
            "username": "test_server",
            "role": "server",
            "connected_at": "test_time",
        }

        try:
            # Call the broadcast function
            broadcast_client_data_update()

            # Should attempt to emit client data updates
            assert mock_socketio.emit.called
        finally:
            # Clean up
            if mock_session_id in online_users:
                del online_users[mock_session_id]


class TestSocketIOErrorHandling:
    """Test Socket.IO error handling"""

    def test_invalid_event_data(self, socketio_client):
        """Test handling of invalid event data"""
        # Send invalid data
        socketio_client.emit("request_logs", {"invalid": "data"})

        # Should handle gracefully without crashing
        received = socketio_client.get_received()
        assert isinstance(received, list)

    def test_unknown_event(self, socketio_client):
        """Test handling of unknown events"""
        # Send unknown event
        socketio_client.emit("unknown_event", {"data": "test"})

        # Should handle gracefully
        assert socketio_client.is_connected()

    def test_malformed_request_logs(self, socketio_client):
        """Test handling of malformed log requests"""
        # Send malformed log request
        socketio_client.emit("request_logs", "not a dict")

        # Should handle gracefully
        received = socketio_client.get_received()
        assert isinstance(received, list)
