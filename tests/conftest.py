"""
Test configuration for Flask application
"""

import os
import tempfile
import pytest
from app import app, socketio


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret-key"

    # Create a temporary directory for uploads during testing
    app.config["UPLOAD_FOLDER"] = tempfile.mkdtemp()

    with app.test_client() as client:
        with app.app_context():
            yield client


@pytest.fixture
def socketio_client(client):
    """Create a test client for Socket.IO with session."""
    app.config["TESTING"] = True

    # First establish a session using the Flask test client
    with client.session_transaction() as sess:
        sess["username"] = "test_user"
        sess["role"] = "client"

    # Create Socket.IO test client with the session context
    socketio_client = socketio.test_client(app, flask_test_client=client)

    return socketio_client
