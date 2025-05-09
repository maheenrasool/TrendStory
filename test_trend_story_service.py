# This is the test file we are using to test the server and client interaction logic.

#  It checks:
# 1. The client stub has the expected method.
# 2. The base service class handles unimplemented methods correctly.
# 3. The server binds correctly.
# 4. The client can make a call using the grpc.experimental API (if available).


import unittest
import grpc
from unittest.mock import Mock
from concurrent import futures

import TrendStory_pb2 as pb2
import TrendStory_pb2_grpc as pb2_grpc


class TestTrendStoryService(unittest.TestCase):

    def test_stub_creation(self):
        # Create a dummy channel (insecure, no real connection)
        """
        Creates a dummy gRPC channel to localhost:50051 (not connecting for real).
        Instantiates the client stub using that channel.
        Checks if the stub has a method called GetStory.
        This confirms that the protobuf compilation and stub generation worked correctly.
        """
        
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = pb2_grpc.TrendStoryServiceStub(channel)
            self.assertTrue(hasattr(stub, 'GetStory'))



    def test_servicer_base_class(self):
        # Create a mock request and context
        """
        Mocks a TrendStoryRequest and a gRPC context.
        Instantiates the base server-side class (not a custom implementation).
        Calls the GetStory() method, which is not implemented by default.
        Ensures:
        It raises a NotImplementedError.
        It sets the correct gRPC error code and details on the context.
        
        This validates the default fallback behavior for unimplemented server methods
        """
        
        request = Mock(spec=pb2.TrendStoryRequest)
        context = Mock()

        servicer = pb2_grpc.TrendStoryServiceServicer()

        with self.assertRaises(NotImplementedError):
            servicer.GetStory(request, context)
        
        context.set_code.assert_called_once_with(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details.assert_called_once_with('Method not implemented!')

    def test_add_servicer_to_server(self):
        """Creates a mock gRPC server.
        Mocks the service implementation (TrendStoryServiceServicer).
        Calls the function to bind the servicer to the server.
        Passes if no exception is raised.
        This confirms that the auto-generated code can bind the service properly, which is necessary to start a working gRPC server.
        """
        
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        servicer = Mock(spec=pb2_grpc.TrendStoryServiceServicer)

        try:
            pb2_grpc.add_TrendStoryServiceServicer_to_server(servicer, server)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"add_TrendStoryServiceServicer_to_server raised an exception: {e}")


    def test_experimental_get_story_method(self):
        """
        First checks if your version of grpc has the experimental module.
        If not, it skips the test.
        Otherwise, it uses the TrendStoryService.GetStory() function,
        which is a special static method defined in the gRPC stub (using grpc.experimental.unary_unary() internally).
        It tests that the call returns something (doesn’t raise or return None).
        """
        
        if not hasattr(grpc, "experimental"):
            self.skipTest("grpc.experimental is not available in this grpc version")

        request = Mock(spec=pb2.TrendStoryRequest)
        target = 'localhost:50051'

        call = pb2_grpc.TrendStoryService.GetStory(
            request=request,
            target=target,
            insecure=True,
            timeout=1
        )
        self.assertIsNotNone(call)



if __name__ == '__main__':
    unittest.main()
