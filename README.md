# TrendStory: gRPC-based AI Story Generator

**TrendStory** is a full-stack, AI-powered application that uses **gRPC** to deliver real-time story generation from trending data. It fetches and analyzes trends from **YouTube** and **Google News**, scores them using multiple metrics, and crafts creative narratives using **locally hosted LLMs** via **Ollama**. The interface is built with **Gradio**, and users can filter stories by **theme**, **tone**, **style**, **category**, and **region**.

---

## 🚀 Features

- **LLM-Driven**: Story generation powered by `llama3` via **Ollama** (runs locally, no API key required)
- **gRPC Architecture**: High-performance server–client communication using `grpc` and `protobuf`
- **Live Trend Mining**:
  - YouTube Data API
  - Google News RSS feeds
- **Language Processing**:
  - Language detection (`langdetect`)
  - Sentiment analysis (`nltk`, VADER)
  - Keyword extraction (`YAKE`, `TF-IDF`)
- **Graph Visualization**:
  - Logical graph (keyword/sentiment)
  - Statistical graph (TF-IDF/metadata-based)
- **Customizable Narratives**:
  - **Tones**: Comedy, Tragedy, Neutral
  - **Themes**: Mystery, Romance, Drama, etc.
  - **Styles**: Short Story, Poem, etc.
  - **Languages**: English / Roman Urdu
  - **Categories/Regions**: User-defined

---

## 📁 Project Structure

```
├── PC0/                      # Server-side code (gRPC server)
│   └── server.py
├── PC1/                      # Client-side code (Gradio UI)
│   └── client.py
├── TrendStory.proto          # gRPC service definition
├── TrendStory_pb2.py         # Generated protobuf classes
├── TrendStory_pb2_grpc.py    # Generated gRPC bindings
├── StoryMaker.py             # Story generation + graph creation
├── TrendExtrAnalyzer.py      # Trend fetching, scoring, and processing
├── TrendCleaner.py           # Keeps only top 3 trends for testing
├── trends.json               # Fetched trend data
├── Dockerfile                # Docker container configuration
├── requirements.txt          # Project dependencies
```

---

## 🛠 Additional Scripts

### `TrendCleaner.py`

Quick utility to trim the `trends.json` file and keep only the first 3 entries.

```python
with open('trends.json', 'r') as f:
    trends = json.load(f)
trends = trends[:3]
with open('trends.json', 'w') as f:
    json.dump(trends, f, indent=2)
```

### `TrendExtrAnalyzer.py`

Runs every hour via `cron` to fetch and analyze the latest trends:

```bash
0 * * * * /usr/bin/python /home/app/TrendExtrAnalyzer.py >> /home/app/cron_output.log 2>&1
```

Performs:
- YouTube + Google News trend extraction
- Translation & language detection
- Sentiment & keyword extraction
- TF-IDF + importance + info gain + relevance scoring
- Categorization and topic classification

---

## 🔁 gRPC Request Flow

1. User selects preferences via Gradio UI (`client.py`)
2. Request sent via `TrendStory_pb2.TrendStoryRequest`
3. Server (`server.py`) handles the request and invokes:
   - `StoryMaker.story_maker()` → filters trends, builds graphs, creates prompt
   - Generates images + sends prompt to **Ollama**
4. Returns a structured story + graph images

---

## 🐳 Running with Docker

```bash
docker build -t trendstory-app .
docker run -d -p 50052:50052 trendstory-app
```

---

## 🧪 Running Locally

```bash
pip install -r requirements.txt

# Start gRPC server
python PC0/server.py

# Start Gradio UI client
python PC1/client.py
```

---

## 📦 Technologies Used

- **gRPC** / **Protocol Buffers**
- **Python 3.9+**
- **Ollama** (`llama3` model, locally hosted)
- **Gradio** for the frontend UI
- **NetworkX** / **Matplotlib** for graph visualization
- **NLTK**, **YAKE**, **TF-IDF** for NLP
- **Asyncio** for concurrent server support
- **Docker** for containerization

---

## 📜 .proto Service Definition

```proto
service TrendStoryService {
  rpc GetStory (TrendStoryRequest) returns (TrendStoryResponse);
}

message TrendStoryRequest {
  string tones = 1;
  string themes = 2;
  string styles = 3;
  string language = 4;
  string category = 5;
  string region = 6;
}

message TrendStoryResponse {
  string response = 1;
  bytes image_data_clean = 2;
  bytes image_data_messy = 3;
}
```

### Generate bindings:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. TrendStory.proto
```

---

## 🧠 Architecture Overview
                             ┌─────────────────────┐
                             │     User (UI)       │
                             │ Gradio Frontend     │
                             └────────┬────────────┘
                                      │
                        Trend Preferences (gRPC Request)
                                      ▼
                        ┌────────────────────────────┐
                        │     gRPC Client (client.py)│
                        └────────┬───────────────────┘
                                 │
                                 ▼
                   ┌─────────────────────────────┐
                   │     gRPC Server (server.py) │
                   └────────┬────────────────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ StoryMaker.story_maker()   │
              │  • Filter Trends           │
              │  • Build Graphs            │
              │  • Create Prompt           │
              │  • Call Ollama (LLaMA3)    │
              └───────┬────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌─────────────────┐      ┌────────────────────────┐
│ TrendCleaner.py │      │ TrendExtrAnalyzer.py   │
│  • Keep top 3    │      │  • YouTube + News API  │
│  • Pre-testing   │      │  • Sentiment/Keywords  │
└─────────────────┘      │  • TF-IDF, Categorize  │
                         └────────────────────────┘

                      ▼
         ┌──────────────────────────┐
         │ Story + Graph Images     │
         └────────────┬─────────────┘
                      │ gRPC Response
                      ▼
            ┌────────────────────┐
            │    Gradio UI       │
            │  (Story + Graphs)  │
            └────────────────────┘

```
[PC1/client.py] ── gRPC ──> [PC0/server.py] ──> [StoryMaker.py]
                                  │
                                  ├── trends.json (input data)
                                  ├── TrendExtrAnalyzer.py (hourly fetch)
                                  └── TrendCleaner.py (testing utility)
```

---


