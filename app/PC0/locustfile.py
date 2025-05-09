import time
import grpc
import grpc.experimental.gevent as grpc_gevent
from locust import User, task, between, events

import TrendStory_pb2
import TrendStory_pb2_grpc

# Enable gevent-compatible gRPC
grpc_gevent.init_gevent()


class GrpcClient:
    def __init__(self, target: str):
        self.channel = grpc.insecure_channel(target)
        self.stub = TrendStory_pb2_grpc.TrendStoryServiceStub(self.channel)

    def get_story(self, language: str = "eng", tones=None, themes=None, styles=None):
        if tones is None:
            tones = ["neutral"]
        if themes is None:
            themes = ["tragedy"]
        if styles is None:
            styles = ["short story"]

        request = TrendStory_pb2.TrendStoryRequest(
            language=language,
            tones=tones,
            themes=themes,
            styles=styles,
        )
        return self.stub.GetStory(request)


class TrendStoryUser(User):
    abstract = False
    wait_time = between(1, 2)

    host = "localhost:50052"  # gRPC server port

    def __init__(self, environment):
        super().__init__(environment)
        self.client = GrpcClient(self.host)

    @task(1)
    def get_story(self):
        start = time.time()
        try:
            response = self.client.get_story()
            elapsed = int((time.time() - start) * 1000)
            events.request.fire(
                request_type="grpc",
                name="GetStory",
                response_time=elapsed,
                response_length=len(response.response),
                exception=None,
            )
        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            events.request.fire(
                request_type="grpc",
                name="GetStory",
                response_time=elapsed,
                response_length=0,
                exception=e,
            )
