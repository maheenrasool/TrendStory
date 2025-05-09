import unittest
from unittest.mock import patch, MagicMock
from client import get_trend_story  # Assuming client.py is in the same directory

class TestClient(unittest.TestCase):

    @patch('client.grpc.insecure_channel')
    @patch('client.TrendStory_pb2_grpc.TrendStoryServiceStub')
    def test_get_trend_story_success(self, MockTrendStoryServiceStub, MockInsecureChannel):
        """
        This test is designed to check if the get_trend_story function behaves as expected when the gRPC call is successful.
        This test ensures that when a valid gRPC connection is made, the function correctly 
        handles the successful response and returns the appropriate story

        """
        # Mock the gRPC response
        mock_channel = MagicMock()
        MockInsecureChannel.return_value = mock_channel
        
        # Create a mock for the TrendStoryServiceStub
        mock_stub = MagicMock()
        MockTrendStoryServiceStub.return_value = mock_stub
        
        # Mock the response from the GetStory method
        mock_response = MagicMock()
        mock_response.response = "Generated story based on provided style, theme, and category."
        mock_stub.GetStory.return_value = mock_response
        
        # Inputs for the get_trend_story function
        style = ["Minimal"]
        theme = ["Dark"]
        category = ["Political"]
        
        # Call the function
        result = get_trend_story(style, theme, category)
        
        # Test if the response is correct
        self.assertEqual(result, "Generated story based on provided style, theme, and category.")
        
        # Test if gRPC was called correctly
        MockInsecureChannel.assert_called_with('172.17.41.8:50052')
        MockTrendStoryServiceStub.assert_called_once()
        mock_stub.GetStory.assert_called_once()

    @patch('client.grpc.insecure_channel')
    @patch('client.TrendStory_pb2_grpc.TrendStoryServiceStub')
    def test_get_trend_story_failure(self, MockTrendStoryServiceStub, MockInsecureChannel):
        
        """This test is designed to check if the get_trend_story function handles errors 
        correctly when there is a failure in the gRPC communication.
        This test ensures that when an error occurs in the gRPC communication, the get_trend_story function 
        properly handles it and returns an appropriate error message to the user
        
        """
        # Mock the gRPC response
        mock_channel = MagicMock()
        MockInsecureChannel.return_value = mock_channel
        
        # Create a mock for the TrendStoryServiceStub
        mock_stub = MagicMock()
        MockTrendStoryServiceStub.return_value = mock_stub
        
        # Simulate an exception occurring during the gRPC call
        mock_stub.GetStory.side_effect = Exception("gRPC Error")
        
        # Inputs for the get_trend_story function
        style = ["Minimal"]
        theme = ["Dark"]
        category = ["Political"]
        
        # Call the function and check if it handles the exception
        result = get_trend_story(style, theme, category)
        
        # Test if the exception is caught and handled correctly
        self.assertEqual(result, "❌ gRPC Error: gRPC Error")
        
        # Test if gRPC was called correctly
        MockInsecureChannel.assert_called_with('172.17.41.8:50052')
        MockTrendStoryServiceStub.assert_called_once()
        mock_stub.GetStory.assert_called_once()

    def test_get_trend_story_invalid_input(self):
        """This test checks how the get_trend_story function behaves when it receives invalid input, 
        specifically when the style, theme, and category lists are empty
        This test ensures that when the user does not provide any valid selections, the function 
        correctly handles the invalid input and returns a suitable error message
        """
        # Call the function with invalid inputs
        result = get_trend_story([], [], [])
        
        # Test if the error message is correct
        self.assertEqual(result, "⚠️ Please select at least one option from each category.")

if __name__ == '__main__':
    unittest.main()
