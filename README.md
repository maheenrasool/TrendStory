TrendStory Generator
A Python-based NLP application that extracts trending topics, generates a visual semantic graph, and produces a narrative from the connected trends. The application uses a gRPC server-client model for backend processing and features a Gradio-powered user interface for interaction.

🧠 Features
📈 Semantic trend graph generation using TF-IDF, metadata, and graph algorithms

📝 Story generation based on trend relationships

🖼️ Visualization of trend graphs using NetworkX and Matplotlib

⚡ gRPC-based backend communication

🧪 Postman test collection included for validating gRPC endpoints

💻 Simple Gradio UI with multi-select options (tone, theme, style, language)

📁 Project Structure
bash
Copy
Edit
.
├── NLP/
│   └── PC0/
│       ├── server.py               # gRPC server implementation
│       └── StoryMaker.py          # Graph building & story generation
│   └── PC1/
│       ├── client.py              # Gradio UI & gRPC client logic
│       ├── TrendStory_pb2.py      # Auto-generated protobuf class
│       └── TrendStory_pb2_grpc.py # Auto-generated gRPC stub
├── trends.json                    # Trend input data
├── requirements.txt               # All required packages
├── Dockerfile                     # Docker setup
└── README.md                      # You're here!
🚀 How to Run
Option 1: Manual Run
Server
bash
Copy
Edit
python NLP/PC0/server.py
Client
bash
Copy
Edit
python NLP/PC1/client.py
Option 2: Docker
bash
Copy
Edit
docker build -t trendstory-service .
docker run -p 50052:50052 trendstory-service
Ensure port 50052 is available and both systems are on the same network if running client and server on separate devices.

🔌 API Protocol (gRPC)
Method: GetStory

Request: TrendStoryRequest(tones, themes, styles, language)

Response: TrendStoryResponse(story: str, image_data: bytes)

🧪 Testing
Unit tests using unittest validate:

Stub creation

Service binding

Base servicer exceptions

Experimental static gRPC calls

Postman test cases available under /tests/collection.json

💻 UI Snapshot
The UI is built using Gradio with:

CheckboxGroups for tones, themes, styles, language

Button for triggering generation

Output: Text + Image

📦 Dependencies
Install with:

bash
Copy
Edit
pip install -r requirements.txt
Includes:

grpcio, grpcio-tools

feedparser, scikit-learn, matplotlib

gradio, Pillow, spacy, textblob, emoji, etc.

![image](https://github.com/user-attachments/assets/c420adcd-2273-4b8e-b6db-9c41c59d8711)

