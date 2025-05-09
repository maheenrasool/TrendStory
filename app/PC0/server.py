
# # **********************************CODE FOR RUNNING CONCURRENTLY WITHOUT ASYNC******************************
# # PC0 - server.py 
# import grpc
# from concurrent import futures
# import TrendStory_pb2
# import TrendStory_pb2_grpc
# import StoryMaker
# import json

# Server will listen on this port
# SERVER_PORT = 50052  # 🔁 Change this if needed

# # Define the service implementation
# class TrendStoryServiceServicer(TrendStory_pb2_grpc.TrendStoryServiceServicer):
#     def GetStory(self, request, context):
#         # Load trend data
#         with open('trends.json', 'r', encoding='utf-8') as f:
#             trends = json.load(f)

#         # Extract fields from request and prepare for story_maker()
#         language = request.language or "English"

#         # Convert CSV strings to the expected sets/strings
#         tones = (request.tones.split(",")[0] if request.tones else "Neutral")  # tone is a single value
#         themes = (request.themes.split(",")[0] if request.themes else "Tragedy")  # theme is a single value
#         styles = (request.styles.split(",")[0] if request.styles else "Short Story")  # style is a single value

#         categories = set(request.category.split(",")) if request.category else {"News & Politics"}
#         regions = set(request.region.split(",")) if request.region else {"PK"}

#         # Generate story and graph paths from StoryMaker
#         result = StoryMaker.story_maker(
#             trends=trends,
#             language=language,
#             regions=regions,
#             categories=categories,
#             tone=tones,
#             theme=themes,
#             style=styles,
#             image=True
#         )

#         story_text = result["story"]
#         path_messy = result["graph_logical"]
#         path_clean = result["graph_statistical"]

#         # Read image files
#         try:
#             with open(path_clean, "rb") as f1:
#                 image_bytes_clean = f1.read()
#         except FileNotFoundError:
#             image_bytes_clean = b""

#         try:
#             with open(path_messy, "rb") as f2:
#                 image_bytes_messy = f2.read()
#         except FileNotFoundError:
#             image_bytes_messy = b""

#         return TrendStory_pb2.TrendStoryResponse(
#             response=story_text,
#             image_data_clean=image_bytes_clean,
#             image_data_messy=image_bytes_messy
#         )

# # Start gRPC server
# def serve():
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#     TrendStory_pb2_grpc.add_TrendStoryServiceServicer_to_server(TrendStoryServiceServicer(), server)
#     server.add_insecure_port(f'[::]:{SERVER_PORT}')
#     server.start()
#     print(f"✅ gRPC server is running on port {SERVER_PORT}...")
#     server.wait_for_termination()

# if __name__ == '__main__':
#     serve()


# **********************************CODE FOR ASYNC******************************
import grpc
import grpc.aio
import TrendStory_pb2
import TrendStory_pb2_grpc
import StoryMaker
import json
import asyncio
import signal

SERVER_PORT = 50052

class TrendStoryServiceServicer(TrendStory_pb2_grpc.TrendStoryServiceServicer):

    async def GetStory(self, request, context):
        loop = asyncio.get_running_loop()

        # Validate input fields
        if not request.language or not request.tones or not request.themes or not request.styles:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Missing required fields: language, tones, themes, or styles.")
            return TrendStory_pb2.TrendStoryResponse()

        try:
            trends = await loop.run_in_executor(None, self._load_trends)

            # Extract and process input fields
            language = request.language or "English"
            tone = request.tones.split(",")[0] if request.tones else "Neutral"
            theme = request.themes.split(",")[0] if request.themes else "Tragedy"
            style = request.styles.split(",")[0] if request.styles else "Short Story"
            categories = set(request.category.split(",")) if request.category else {"News & Politics"}
            regions = set(request.region.split(",")) if request.region else {"PK"}

            result = await loop.run_in_executor(
                None,
                StoryMaker.story_maker,
                trends,
                language,
                regions,
                categories,
                tone,
                theme,
                style,
                True  # image=True
            )

            story_text = result.get("story", "")
            path_clean = result.get("graph_statistical", "")
            path_messy = result.get("graph_logical", "")

            image_bytes_clean = await loop.run_in_executor(None, self._read_file_safe, path_clean)
            image_bytes_messy = await loop.run_in_executor(None, self._read_file_safe, path_messy)

            return TrendStory_pb2.TrendStoryResponse(
                response=story_text,
                image_data_clean=image_bytes_clean,
                image_data_messy=image_bytes_messy
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Server error: {str(e)}")
            return TrendStory_pb2.TrendStoryResponse()

    def _load_trends(self):
        with open('trends.json', 'r', encoding='utf-8') as f:
            return json.load(f)

    def _read_file_safe(self, path):
        try:
            with open(path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return b""

async def serve():
    server = grpc.aio.server()
    TrendStory_pb2_grpc.add_TrendStoryServiceServicer_to_server(TrendStoryServiceServicer(), server)
    server.add_insecure_port(f'[::]:{SERVER_PORT}')
    await server.start()
    print(f"✅ Async gRPC server running on port {SERVER_PORT}. Press Ctrl+C to stop.")

    # Setup graceful shutdown handling
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()
    print("🔻 Stopping server...")
    await server.stop(grace=None)
    print("✅ Server shutdown complete.")

if __name__ == '__main__':
    asyncio.run(serve())



# ------------------non concurrent with non json output handling
# # PC0 - server.py
# import grpc
# from concurrent import futures
# import TrendStory_pb2
# import TrendStory_pb2_grpc
# import requests
# import StoryMaker
# import json

# # Server will listen on this port
# SERVER_PORT = 50052  # 🔁 Change this if needed

# # Define the service implementation
# class TrendStoryServiceServicer(TrendStory_pb2_grpc.TrendStoryServiceServicer):
#     def GetStory(self, request, context):

#         # Load trends
#         with open('trends.json', 'r', encoding='utf-8') as f:
#             trends = json.load(f)

#         # Convert input strings to appropriate types for story_maker
#         language = request.language or "English"
#         tone = request.tones.split(",")[0] if request.tones else "Neutral"
#         theme = request.themes.split(",")[0] if request.themes else "Tragedy"
#         style = request.styles.split(",")[0] if request.styles else "Short Story"
#         categories = set(request.category.split(",")) if request.category else {"News & Politics"}
#         regions = set(request.region.split(",")) if request.region else {"PK"}

#         # Generate story
#         reply_dict = StoryMaker.story_maker(
#             trends=trends,
#             language=language,
#             regions=regions,
#             categories=categories,
#             tone=tone,
#             theme=theme,
#             style=style,
#             image=True
#         )

#         print("DOES REPLY: ", reply_dict)

#         image_path_clean = reply_dict.get("graph_statistical", "clean_trend_graph.png")
#         image_path_messy = reply_dict.get("graph_logical", "messy_trend_graph.png")

#         try:
#             with open(image_path_clean, "rb") as f1:
#                 image_bytes_clean = f1.read()
#         except FileNotFoundError:
#             image_bytes_clean = b""

#         try:
#             with open(image_path_messy, "rb") as f2:
#                 image_bytes_messy = f2.read()
#         except FileNotFoundError:
#             image_bytes_messy = b""

#         return TrendStory_pb2.TrendStoryResponse(
#             response=reply_dict.get("story", ""),
#             image_data_clean=image_bytes_clean,
#             image_data_messy=image_bytes_messy
#         )

# # Start gRPC server
# def serve():
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#     TrendStory_pb2_grpc.add_TrendStoryServiceServicer_to_server(TrendStoryServiceServicer(), server)
#     server.add_insecure_port(f'[::]:{SERVER_PORT}')
#     server.start()
#     print(f"✅ gRPC server is running on port {SERVER_PORT}...")
#     server.wait_for_termination()

# if __name__ == '__main__':
#     serve()
