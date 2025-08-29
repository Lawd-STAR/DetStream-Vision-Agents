"""
Integration tests for the TavusClient and TavusProcessor classes.

These tests make actual API calls to the Tavus API and require a valid TAVUS_KEY
environment variable to be set.
"""

import pytest
import os
from dotenv import load_dotenv
from unittest.mock import patch
import requests

from stream_agents.processors.tavus_processor import TavusClient, TavusProcessor


# Load environment variables
load_dotenv()


class TestTavusClientIntegration:
    """Integration tests for TavusClient that make actual API calls."""
    
    @pytest.fixture(scope="class")
    def api_key(self):
        """Get the Tavus API key from environment variables."""
        api_key = os.getenv("TAVUS_KEY")
        if not api_key:
            pytest.skip("TAVUS_KEY environment variable not set")
        return api_key
    
    @pytest.fixture(scope="class")
    def client(self, api_key):
        """Create a TavusClient instance for testing."""
        return TavusClient(api_key)
    
    @pytest.fixture(scope="class")
    def test_replica_id(self):
        """Test replica ID - you may need to update this with a valid replica ID."""
        # This is the example from the API docs - replace with your actual replica ID
        return os.getenv("TAVUS_TEST_REPLICA_ID", "rfe12d8b9597")
    
    @pytest.fixture(scope="class")
    def test_persona_id(self):
        """Test persona ID - you may need to update this with a valid persona ID."""
        # This is the example from the API docs - replace with your actual persona ID
        return os.getenv("TAVUS_TEST_PERSONA_ID", "pdced222244b")
    
    @pytest.mark.integration
    def test_client_initialization(self, api_key):
        """Test that TavusClient initializes correctly."""
        client = TavusClient(api_key)
        assert client.api_key == api_key
        assert client.base_url == "https://tavusapi.com/v2"
        assert "x-api-key" in client.session.headers
        assert client.session.headers["x-api-key"] == api_key
        assert client.session.headers["Content-Type"] == "application/json"
    
    @pytest.mark.integration
    def test_create_conversation_success(self, client, test_replica_id, test_persona_id):
        """Test creating a conversation with valid parameters."""
        try:
            conversation_data = client.create_conversation(
                replica_id=test_replica_id,
                persona_id=test_persona_id,
                conversation_name="Test Conversation - Integration Test"
            )
            
            # Verify response structure
            assert "conversation_id" in conversation_data
            assert "conversation_url" in conversation_data
            assert "status" in conversation_data
            assert "replica_id" in conversation_data
            assert "persona_id" in conversation_data
            
            # Verify the returned data matches what we sent
            assert conversation_data["replica_id"] == test_replica_id
            assert conversation_data["persona_id"] == test_persona_id
            
            # Store conversation_id for cleanup
            return conversation_data["conversation_id"]
            
        except requests.RequestException as e:
            # If this fails due to invalid replica/persona IDs, that's expected
            # Log the error and skip the test
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 400:
                    pytest.skip(f"Test skipped due to invalid test IDs: {e}")
                elif e.response.status_code == 401:
                    pytest.fail("Authentication failed - check your TAVUS_KEY")
                else:
                    pytest.fail(f"Unexpected API error: {e}")
            else:
                pytest.fail(f"Network error: {e}")
