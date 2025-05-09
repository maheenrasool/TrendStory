#PC1 -client.py 
#The Ollama API is now available at 127.0.0.1:11434.
"""
📡 gRPC Network Setup Instructions (PC0 = Server, PC1 = Client)

STEP 1️⃣: Get Server IP Address (on PC0)

    On Windows:
        1. Open Command Prompt
        2. Type: ipconfig
        3. Copy the 'IPv4 Address' under your active adapter (e.g., Wi-Fi)
           Example:
               IPv4 Address. . . . . . . . . . . : 192.168.18.105

    On macOS/Linux:
        1. Open Terminal
        2. Type: ifconfig OR ip addr
        3. Find the 'inet' IP under your active adapter (e.g., wlan0 or en0)
           Example:
               inet 192.168.18.105

    ➤ Use this IP in your client code as `PC0_IP`

STEP 2️⃣: Set IP and Port in Code (on PC1)

    Example (client):
    -----------------
    PC0_IP = "192.168.18.105"      # Replace with IP from Step 1
    client_port_number = 50051     # This must match the server's port

    Example (server):
    -----------------
    server.add_insecure_port('[::]:50051')

STEP 3️⃣: Test Connectivity (Optional but Recommended)

    On PC1 (Client machine):

    ✅ Ping Test:
        ping 192.168.18.105

    ✅ Port Test:
        On Windows (after enabling Telnet Client):
            telnet 192.168.18.105 50051

        On macOS/Linux:
            nc -zv 192.168.18.105 50051

STEP 4️⃣: Run gRPC Server on PC0
    python server.py

STEP 5️⃣: Run gRPC Client (Gradio frontend or test script) on PC1
    python client.py or gradio_ui.py

⚠️ Notes:
    - Both PCs must be on the same local network (same Wi-Fi/router).
    - Disable firewall or allow port 50051 if connection is blocked.
    - Do not use ports below 1024 unless you know what you're doing.
"""

import gradio as gr
import grpc
import TrendStory_pb2
import TrendStory_pb2_grpc
from PIL import Image, ImageDraw, ImageFont
import io

# Replace with your actual server IP and port
PC0_IP = "192.168.0.104"
SERVER_PORT = 50053 #50052 for non docker version

# Helper: create dummy image with title text
def create_dummy_image(title, size=(300, 200), color=(30, 30, 30)):
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    font_size = 18
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    draw.text((x, y), title, fill="white", font=font)
    return img

# gRPC client function
def get_trend_story(tones, themes, styles, language, category, region):
    if not tones or not themes or not styles:
        return "⚠️ Please select at least one option from each category.", None, None

    try:
        channel = grpc.insecure_channel(f'{PC0_IP}:{SERVER_PORT}')
        stub = TrendStory_pb2_grpc.TrendStoryServiceStub(channel)

        request = TrendStory_pb2.TrendStoryRequest(
            tones=",".join(tones),
            themes=",".join(themes),
            styles=",".join(styles),
            language=",".join(language),
            category=",".join(category),
            region=",".join(region)
        )

        response = stub.GetStory(request)

        image_clean = Image.open(io.BytesIO(response.image_data_clean)) if response.image_data_clean else None
        image_messy = Image.open(io.BytesIO(response.image_data_messy)) if response.image_data_messy else None

        return response.response, image_clean, image_messy
    except Exception as e:
        return f"❌ gRPC Error: {str(e)}", None, None

# Dummy story data
popular_titles = [
    "AI Predicts Next Global Conflict", 
    "Fantasy World of Eloria Tops Charts", 
    "Dark Politics in the New Age",
    "Comedy in the Age of Robots", 
    "Vintage Adventures in Time",
    "Drama Unfolds in Space Trials",
    "Poetry Revives Lost Kingdoms",
    "Sci-Fi Tech Turns Sentient", 
    "Cyber Fights in Year 3025",
    "Royal Scandals of Mars Colony", 
    "AI Learns to Paint Dreams"
]

popular_stories = []
for i, title in enumerate(popular_titles):
    img = create_dummy_image(title)
    path = f"dummy_story_{i}.png"
    img.save(path)
    popular_stories.append({"title": title, "image": path})

# Optional: custom theme
purple_theme = gr.themes.Base().set(
    body_background_fill="#121212",
    body_text_color="#ffffff",
    input_background_fill="#8A2BE2",
    input_border_color="#cccccc",
    block_title_text_color="#ffffff",
    block_background_fill="#7B1FA2",
    block_border_color="#4B0082",
    button_primary_background_fill="#DDA0DD",
    button_primary_text_color="#000000",
    button_secondary_background_fill="#BA55D3",
    button_secondary_text_color="#000000"
)

# UI
with gr.Blocks(theme=purple_theme, title="gRPC TrendStory Generator") as demo:
    gr.HTML("""<style>button:hover { background-color: #E6CCE6 !important; color: #000 !important; transition: background-color 0.3s ease; }</style>""")
    gr.Markdown("<h2 style='text-align: center;'>📰 TrendStory Generator</h2>")
    gr.Markdown("Select options and generate a story using gRPC.")

    with gr.Row():
        tones = gr.CheckboxGroup(["Sarcastic","Comedic","Dramatic","Hopeful","Mysterious","Inspirational","Neutral"], label="Tones")
        themes = gr.CheckboxGroup(["Fantasy","Sci-Fi","Romance","Tragedy","Adventure"], label="Themes")
        styles = gr.CheckboxGroup(["Blog Post","News Article","Short Story","Diary Entry","Podcast Transcript","Monologue","Poetic Prose"], label="Styles")
        language = gr.CheckboxGroup(["Roman Urdu","English"], label="Language")

    with gr.Row():
        category = gr.CheckboxGroup(["Politics", "Tech", "Entertainment", "Health", "Economy"], label="Category")
        region = gr.CheckboxGroup(["Global", "Asia", "Europe", "Middle East", "Africa", "Americas"], label="Region")

    generate_btn = gr.Button("🎯 Generate via gRPC")
    story_output = gr.Textbox(label="Generated Story", lines=8, interactive=False)
    generated_image_clean = gr.Image(label="Clean Trend Graph")
    generated_image_messy = gr.Image(label="Messy Trend Graph")

    generate_btn.click(
        fn=get_trend_story,
        inputs=[tones, themes, styles, language, category, region],
        outputs=[story_output, generated_image_clean, generated_image_messy]
    )

    gr.Markdown("### 🔥 Popular & Previous TrendStories")
    scroll_html = "<div style='height: 500px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; padding: 10px;'>"
    for story in popular_stories:
        scroll_html += f"""
        <div style='text-align: center;'>
            <img src='{story["image"]}' style='width: 100%; max-width: 300px; height: auto; border-radius: 10px;' />
            <div style='color: white; margin-top: 5px;'>{story["title"]}</div>
        </div>
        """
    scroll_html += "</div>"
    gr.HTML(scroll_html)

demo.launch(share=False)
