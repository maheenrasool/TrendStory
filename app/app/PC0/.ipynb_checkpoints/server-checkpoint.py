# PC0 - server.py
import grpc
from concurrent import futures
import TrendStory_pb2
import TrendStory_pb2_grpc
import requests
import StoryMaker
import json


# Server will listen on this port
SERVER_PORT = 50052  # 🔁 Change this if needed

# Define the service implementation
class TrendStoryServiceServicer(TrendStory_pb2_grpc.TrendStoryServiceServicer):
    def GetStory(self, request, context):

        with open('trends.json', 'r', encoding = 'utf-8') as f:
            trends = json.load(f)
        reply = StoryMaker.story_maker(trends, request.language, request.tones, request.themes, request.styles)
        image_path = "clean_trend_graph.png"
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
        # prompt = f"You are a world class story teller Write a story using :\nStyle: {request.tones}\nTheme: {request.themes}\nCategory: {request.styles}"

        # response = requests.post(
        #     'http://localhost:11434/api/generate',
        #     json={
        #         'model': 'llama3.2:latest',  # model name
        #         'prompt': prompt,
        #         'stream': False  
        #     }
        # )
        
        # ✅ Add your logic here to generate a story from request.tones, request.themes, and request.styles
        # reply = response.json()['response']
        
        return TrendStory_pb2.TrendStoryResponse(response=reply, image_data=image_bytes)

# Start gRPC server
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    TrendStory_pb2_grpc.add_TrendStoryServiceServicer_to_server(TrendStoryServiceServicer(), server)
    server.add_insecure_port(f'[::]:{SERVER_PORT}')
    server.start()
    print(f"✅ gRPC server is running on port {SERVER_PORT}...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
