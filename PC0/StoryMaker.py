#StoryMaker.py (backend)
import time
import requests
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from sklearn.metrics.pairwise import cosine_similarity
from dateutil import parser
from datetime import datetime, timezone

# for image gen
# from diffusers import StableDiffusionPipeline


# statistical trend graph builder
class StatisticalTrendGraphBuilder:
    def __init__(self, trends):
        self.trends = trends
        self.graph = nx.Graph()

    def build_graph(self, threshold=0.3):
        """Builds a graph based on trend similarity (text + metadata)."""
        try:
            if self.trends is None:
                with open('trends.json', 'r', encoding='utf-8') as f:
                    self.trends = json.load(f)

            texts = [
                (t.get('title', '') or '') + ' ' +
                (t.get('description', '') or '') + ' ' +
                (t.get('summary', '') or '') + ' ' +
                (t.get('category', '') or '')
                for t in self.trends
            ]

            # TF-IDF based text similarity
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Metadata boost: combine info_gain, importance, relevance
            info_gains = np.array([t.get('info_gain', 0) for t in self.trends])
            importances = np.array([t.get('importance', 0) for t in self.trends])
            relevances = np.array([t.get('relevance', 0) for t in self.trends])

            n = len(self.trends)
            for i in range(n):
                self.graph.add_node(i, trend=self.trends[i])

            for i in range(n):
                for j in range(i + 1, n):
                    sim = similarity_matrix[i, j]

                    # Metadata similarity boost
                    meta_sim = (
                        abs(info_gains[i] - info_gains[j]) +
                        abs(importances[i] - importances[j]) +
                        abs(relevances[i] - relevances[j])
                    )
                    meta_sim = 1 - (meta_sim / 3)  # Normalize 0-1, higher is more similar

                    total_similarity = (sim + meta_sim) / 2

                    if total_similarity > threshold:
                        self.graph.add_edge(i, j, weight=round(total_similarity, 3))

        except Exception as e:
            print(f"Error in build_graph: {e}")

    def plot_graph(self, story_path="clean_trend_graph.png"):
        """Plots the graph with node indices and edge colors based on source similarity."""
        try:
            pos = nx.spring_layout(self.graph, seed=42)
            plt.figure(figsize=(14, 9))

            # Draw nodes with index labels
            nx.draw_networkx_nodes(self.graph, pos, node_size=500, node_color='skyblue')
            labels = {i: str(i) for i in self.graph.nodes}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=10, font_weight='bold')

            # Color edges based on source similarity
            edge_colors = []
            edge_widths = []
            for u, v, data in self.graph.edges(data=True):
                source_u = self.graph.nodes[u]['trend'].get('source', '').lower()
                source_v = self.graph.nodes[v]['trend'].get('source', '').lower()

                if source_u == 'youtube' and source_v == 'youtube':
                    edge_colors.append('red')
                elif source_u == 'google' and source_v == 'google':
                    edge_colors.append('green')
                else:
                    edge_colors.append('blue')

                edge_widths.append(data['weight'] * 2)

            nx.draw_networkx_edges(self.graph, pos, width=edge_widths, edge_color=edge_colors, alpha=0.6)

            if isinstance(story_path, list):
                path_nodes = [i for i, t in enumerate(self.trends) if t in story_path]
                path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
                nx.draw_networkx_edges(
                    self.graph, pos,
                    edgelist=path_edges,
                    edge_color='black',
                    width=3,
                    style='dashed'
                )
                filename = "clean_trend_graph.png"
            else:
                filename = story_path  # Assume it's a filename string

            plt.title("Trend Graph (Red: YouTube, Green: Google, Blue: Inter-Source)\nDashed Line: Story Path", fontsize=12)
            plt.axis('off')
            plt.tight_layout()

            # Save and close
            plt.savefig(filename, format='png')
            plt.close()
            return os.path.abspath(filename)

        except Exception as e:
            print(f"Error in plot_graph: {e}")
            return None

    def extract_story_path(self):
        """Extracts a path that logically connects trends."""
        try:
            if not self.graph.nodes:
                return []

            # Start with the most important trend
            start_node = max(self.graph.nodes, key=lambda i: self.graph.nodes[i]['trend'].get('importance', 0))
            visited = {start_node}
            path = [start_node]

            current = start_node
            while True:
                neighbors = [
                    (nbr, self.graph[current][nbr]['weight'])
                    for nbr in self.graph.neighbors(current)
                    if nbr not in visited
                ]
                if not neighbors:
                    break

                # Pick the neighbor with highest edge weight
                next_node = max(neighbors, key=lambda x: x[1])[0]
                visited.add(next_node)
                path.append(next_node)
                current = next_node

            # Return the sequence of trend dicts
            return [self.graph.nodes[i]['trend'] for i in path]

        except Exception as e:
            print(f"Error in extract_story_path: {e}")
            top_5_trends = sorted(self.trends, key=lambda t: (
            t.get('importance', 0), 
            t.get('authenticity', 0), 
            t.get('relevance', 0), 
            t.get('info_gain', 0)
        ), reverse=True)[:5]
        return top_5_trends

    def get_graph(self):
        return self.graph

# LogicAl trend graph
class LogicalTrendGraphBuilder:
    def __init__(self, trends):
        self.trends = trends
        self.graph = nx.Graph()

    def parse_published_date(self, published_str):
        """Parses published date and makes it timezone-naive in UTC."""
        try:
            dt = parser.parse(published_str)
            if dt.tzinfo:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception as e:
            print(f"Date parsing error: {e}")
            return None

    def keyword_overlap_score(self, trend1, trend2):
        try:
            keywords1 = set(trend1.get('keywords', []))
            keywords2 = set(trend2.get('keywords', []))
            if not keywords1 or not keywords2:
                return 0
            return len(keywords1 & keywords2) / min(len(keywords1), len(keywords2))
        except Exception as e:
            print(f"Keyword overlap calculation error: {e}")
            return 0

    def category_match_bonus(self, trend1, trend2):
        try:
            return 1.0 if trend1.get('category') == trend2.get('category') else 0.0
        except Exception as e:
            print(f"Category match calculation error: {e}")
            return 0

    def date_proximity_score(self, trend1, trend2):
        try:
            date1 = self.parse_published_date(trend1.get('publishedAt') or trend1.get('published'))
            date2 = self.parse_published_date(trend2.get('publishedAt') or trend2.get('published'))
            if not date1 or not date2:
                return 0
            days_apart = abs((date1 - date2).days)
            return max(0, 1 - days_apart / 7)
        except Exception as e:
            print(f"Date proximity calculation error: {e}")
            return 0

    def sentiment_similarity_score(self, trend1, trend2):
        try:
            s1 = trend1.get('sentiment', 0)
            s2 = trend2.get('sentiment', 0)
            return max(0, 1 - abs(s1 - s2))
        except Exception as e:
            print(f"Sentiment similarity calculation error: {e}")
            return 0

    def calculate_connection_weight(self, trend1, trend2):
        try:
            return (
                self.keyword_overlap_score(trend1, trend2) * 2 +
                self.category_match_bonus(trend1, trend2) * 1.5 +
                self.date_proximity_score(trend1, trend2) * 1.2 +
                self.sentiment_similarity_score(trend1, trend2) * 1.0
            )
        except Exception as e:
            print(f"Error in calculating connection weight: {e}")
            return 0

    def determine_edge_color(self, trend1, trend2):
        """Assigns color based on source types."""
        try:
            source1 = trend1.get('source', '')
            source2 = trend2.get('source', '')
            if source1 == "YouTube" and source2 == "YouTube":
                return 'red'
            elif source1 == "Google News" and source2 == "Google News":
                return 'green'
            else:
                return 'blue'
        except Exception as e:
            print(f"Error determining edge color: {e}")
            return 'gray'  # default to gray if error

    def build_graph(self, threshold=1.5):
        """Builds a semantic graph of trends."""
        try:
            if self.trends is None:
                with open('trends.json', 'r', encoding='utf-8') as f:
                    self.trends = json.load(f)
            n = len(self.trends)
            for i in range(n):
                self.graph.add_node(i, trend=self.trends[i])

            for i in range(n):
                for j in range(i + 1, n):
                    weight = self.calculate_connection_weight(self.trends[i], self.trends[j])
                    if weight > threshold:
                        color = self.determine_edge_color(self.trends[i], self.trends[j])
                        self.graph.add_edge(i, j, weight=round(weight, 3), color=color)

        except Exception as e:
            print(f"Error building the graph: {e}")

    def plot_graph(self, filename=r"messy_trend_graph.png"):
        """Plots the graph with colored edges."""
        try:
            pos = nx.spring_layout(self.graph, seed=42)
            plt.figure(figsize=(14, 10))

            nx.draw_networkx_nodes(self.graph, pos, node_size=600, node_color='skyblue')

            edges = self.graph.edges(data=True)
            edge_colors = [d['color'] for (u, v, d) in edges]
            edge_weights = [d['weight'] for (u, v, d) in edges]

            nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, width=[w for w in edge_weights], alpha=0.7)

            labels = {i: str(i) for i in self.graph.nodes}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)

            plt.title("Trend Semantic Graph", fontsize=16)
            plt.axis('off')
            plt.tight_layout()

            # Save to file
            plt.savefig(filename, format='png')
            plt.close()
            return os.path.abspath(filename)

        except Exception as e:
            print(f"Error plotting the graph: {e}")
            return None

    def extract_story_path(self):
        """Extracts a connected semantic story path."""
        try:
            # if not self.graph.nodes:
            #     return []

            start_node = max(self.graph.nodes, key=lambda i: self.graph.nodes[i]['trend'].get('importance', 0))
            visited = {start_node}
            path = [start_node]

            current = start_node
            while True:
                neighbors = [
                    (nbr, self.graph[current][nbr]['weight'])
                    for nbr in self.graph.neighbors(current)
                    if nbr not in visited
                ]
                if not neighbors:
                    break

                next_node = max(neighbors, key=lambda x: x[1])[0]
                visited.add(next_node)
                path.append(next_node)
                current = next_node

            return [self.graph.nodes[i]['trend'] for i in path]

        except Exception as e:
            print(f"Error extracting story path: {e}")
            # If error occurs, return the top 5 trends based on multiple fields
            top_5_trends = sorted(self.trends, key=lambda t: (
                t.get('importance', 0), 
                t.get('authenticity', 0), 
                t.get('validity', 0), 
                t.get('info_gain', 0)
            ), reverse=True)[:5]
            return top_5_trends

    def get_graph(self):
        return self.graph

def interpret_sentiment(score):
    if score is None:
        return "was a neutral event"
    elif score > 0.5:
        return "was celebrated widely with overwhelming excitement."
    elif score > 0.2:
        return "received a warm and positive reception."
    elif score > -0.2:
        return "elicited a neutral or mixed response from the public."
    elif score > -0.5:
        return "faced mild criticism or disappointment."
    else:
        return "triggered significant backlash and negativity."

def build_writer_prompt(trend_chain,Language = "English", tone="Comedic", theme="Slice of Life", story_style="Analytical Essay"):
    """
    Build a structured prompt to generate a narrative, article, or styled write-up based on trending events.

    Parameters:
    - trend_chain: list of connected trends (ordered semantically or by time)
    - tone: 'Dramatic', 'Technical', 'Investigative', etc.
    - theme: 'Fantasy', 'Comedy', 'Informative', etc.
    - story_style: 'Blog Post', 'News Article', 'Analytical Essay', 'Dialogue', etc.

    Returns: a string prompt ready for LLM input.
    """


    if not trend_chain:
        return "No trends provided. Apologize to the reader for the inconvinience and give them a generic fun bonus story as compensation"

    category_map = {
        "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
        "15": "Pets & Animals", "17": "Sports", "18": "Short Movies",
        "19": "Travel & Events", "20": "Gaming", "21": "Videoblogging",
        "22": "People & Blogs", "23": "Comedy", "24": "Entertainment",
        "25": "News & Politics", "26": "Howto & Style", "27": "Education",
        "28": "Science & Technology", "29": "Nonprofits & Activism",
        "30": "Movies", "31": "Anime/Animation", "32": "Action/Adventure",
        "33": "Classics", "34": "Comedy", "35": "Documentary",
        "36": "Drama", "37": "Family", "38": "Foreign", "39": "Horror",
        "40": "Sci-Fi/Fantasy", "41": "Thriller", "42": "Shorts",
        "43": "Shows", "44": "Trailers"
    }
    tone_instructions = {
        "Comedic": "Crakc some good jokes, infuse the piece with humor. Make light-hearted observations, jokes, and puns to entertain the reader. The tone should be light and fun.",
        "Sarcastic": "Be cocky, mock people, be brutal, roast mode. The tone should be slightly critical, but in a witty way.",
        "Dramatic": "Use emotional highs and lows to build tension and suspense. The tone should be intense and focus on dramatic events.",
        "Hopeful": "Create a positive and uplifting tone. Focus on optimism and the potential for positive change, even in tough situations.",
        "Mysterious": "Write with an air of suspense, leaving some things to the imagination. Create a sense of curiosity and intrigue.",
        "Inspirational": "Use motivational language to encourage the reader. Focus on resilience, overcoming challenges, and finding strength in adversity.",
        "Neutral": "Adopt a factual, unbiased tone. Avoid emotional overtones and present the events in a straightforward, balanced manner."
    }

    # Story Style Instructions
    story_style_instructions = {
        "Blog Post": "Write as though you’re addressing an audience informally. Keep it conversational and engaging, often using personal experience or opinions.",
        "News Article": "Write formally and informatively. Stick to facts and provide clear, concise reporting on the event, with no personal opinions or humor.",
        "Short Story": "Craft a narrative with characters and plot. The story should have a beginning, middle, and end, with emotional depth or conflict.",
        "Diary Entry": "Write as if you are reflecting on your day. The tone should be introspective and personal, almost as if the reader is peeking into your private thoughts.",
        "Podcast Transcript": "Write in a conversational tone, as if you're speaking directly to the listener. The language should be relaxed but informative, with natural transitions and pacing.",
        "Monologue": "Write as if one character is speaking directly to an audience. The tone should be focused, revealing emotions or thoughts on a subject.",
        "Poetic Prose": "Write in a descriptive, lyrical style. Use rich language and imagery to evoke emotions and create vivid scenes."
    }

    # Theme Instructions
    theme_instructions = {
        "Fantasy": "Build a magical or otherworldly narrative, involving mythical creatures, magic, or fantastical landscapes. Let the reader escape into an imaginary world.",
        "Sci-Fi": "Create a futuristic or technological world. The narrative should involve advanced technology, space exploration, or dystopian societies.",
        "Romance": "Center the story around relationships and emotions. Focus on the connection between characters, their conflicts, and their journey towards love or resolution.",
        "Tragedy": "Write about loss, sorrow, and difficult circumstances. The tone should be somber, focusing on emotional pain and hardship.",
        "Adventure": "Focus on excitement, exploration, and risk-taking. Your characters should face challenges in unknown places, with an emphasis on courage and discovery."
    }

    tone_instruction = tone_instructions.get(tone, "Maintain a balanced tone.")
    story_style_instruction = story_style_instructions.get(story_style, "Write in a neutral and appropriate style for the event.")
    theme_instruction = theme_instructions.get(theme, "Write in a general style with no specific theme applied.")

    events_text = ""
    categories_used = set()

    importance_threshold = 0.2
    for idx, trend in enumerate(trend_chain, 1):
        title = trend.get('title', 'Unknown Event').title()
        desc = trend.get('description') or trend.get('summary', 'No Description Available')
        region = trend.get('region', 'Unknown Region')
        category = trend.get('category', 'Unknown Category')
        category_label = category_map.get(str(category), category)
        categories_used.add(category_label)
        source = trend.get('source', 'Unknown Source')
        sentiment = trend.get('sentiment', 0)
        lang = trend.get('lang', 'en')
        translation = trend.get('translation', None)

        if lang != 'en' and translation:
            desc += f" (Tarjuma: {translation})"

        sentiment_statement = interpret_sentiment(sentiment)

        if abs(sentiment) >= importance_threshold:
            importance_statement = "This event is considered significant."
        else:
            importance_statement = "This event holds minor significance."

        events_text += f"{idx}. [{source} - {region}] {title}: {desc}. It {sentiment_statement} {importance_statement}\n"

    regions = set(trend.get('region', 'Unknown Region') for trend in trend_chain)
    
    if Language.lower() != "english":
        prompt = f"""
Roman Urdu main likho (koi Arabic script nahi likhni).

aik kahani likho niche diye gaye world event per, roman urdu main:
**Event**:
    {events_text}

Instructions:
- 3-4 paragraph main mazey ki kahani likho 
- takhleeqi banooo
       
- Sirf kahani likho. 
- Koi explanations ya notes nahi likhne.
- Bas kahani start karo aur khatam karo.

kahani ke start aur end main "********" hamesha likho
at the start and end of the story always write "********"

Shuru karo:
    """
    
    
    else:      

        prompt = f"""
    You are a world-class **creative and technical writer AI**.
    
    You are capable of writing in multiple styles—**fiction, sarcasm, blog-style commentary, technical summaries, creative non-fiction, or news-style reporting**.
    
    Your task is to produce a well-structured, high-quality write-up based on recent real-world events. These events are summarized below:
    
    **Writing Directives**:
    - Style: {story_style} ({story_style_instruction})
    - Tone/Genre: {tone} ({tone_instruction})
    - Theme (if fictional): {theme} ({theme_instruction})
    - Target length: 3–5 paragraphs
    - Regional context to reflect: {', '.join(regions)}
    - Content categories involved: {', '.join(categories_used)}
    
    **Event Summary**:
    {events_text}
    
    **Writing Goals**:
    - Weave a coherent narrative or structured article across the events.
    - Maintain consistency in tone/style while handling emotional transitions.
    - Emphasize connections or contrasts between trends.
    - Creatively integrate cause-effect relationships, motivations, or analysis.
    - For news/technical styles: Be factual and structured.  
    - For fictional or blog styles: Include character/dialogue/world if suitable.

    Important Output Rules:
    - Only write the story or article.
    - Do not add any explanations.
    - Start the story immediately after this instruction.
    - No meta-comments, no introductions, no "Here's a story" text.
    
    Begin your piece below.
    """
    
    print("***************** PROMPT FOR WRITER MODEL *****************")
    print(prompt)
    return prompt.strip()


# def story_maker(trends, language="English", tone = "neutral", theme="Tragedy", style="Short Story"):
#     builder = LogicalTrendGraphBuilder(trends)
#     builder.build_graph(threshold=0.5)  # Threshold tunes connection strictness
#     messy_semantic = builder.plot_graph()
#     print("Graph saved at:", messy_semantic)
    
#     builder = StatisticalTrendGraphBuilder(trends[:])
#     builder.build_graph(threshold=0.5)  # You can tune threshold
#     clean_semantic = builder.plot_graph()
#     print("Graph saved at:", clean_semantic)
#     story = builder.extract_story_path()
#     prompt = build_writer_prompt(story[:5],language, tone, theme, style)
    
#     print("Prompt being sent to model:\n")
#     # print(prompt)
#     print("\nGenerating story...\n")
    
#     # Step 5: Measure time and send to Ollama
#     start_time = time.time()
    
#     response = requests.post(
#         'http://host.docker.internal:11434/api/generate',
#         json={
#             "model": "llama3",  # or whichever model you have
#             "prompt": prompt,
#             "stream": False  
#         }
#     )
    
#     end_time = time.time()
#     elapsed_time = end_time - start_time
    
#     # Step 6: Print the generated story
#     output = {"response": "❌ Failed to generate story."}
#     if response.status_code == 200:
#         output = response.json()
#         print("Generated Story:\n")
#         print(output['response'])
#         print(f"\n⏱️ Time taken to generate story: {elapsed_time:.2f} seconds")
#     else:
#         error_message = f"Error communicating with o-server! Status code: {response.status_code}."
#         print(f"❌ {error_message} Details: {response.text}")
        
#         # Assuming {E} is a placeholder for error information, you can use it like:
#         error_details = error_message.format(E=response.text)
#         print(f": {error_details}")
#     return output['response']



def story_maker(trends=None, language="English", regions={"PK"}, categories={"News & Politics"}, tone="Neutral", theme="Tragedy", style="Short Story", image=True):
    # If any of these are lists, convert to single values
    if isinstance(tone, list): tone = tone[0] if tone else "Neutral"
    if isinstance(theme, list): theme = theme[0] if theme else "Tragedy"
    if isinstance(categories, list): categories = set(categories)
    if isinstance(regions, list): regions = set(regions)
    if isinstance(style, list): style = style[0] if style else "Short Story"

    try:
        # Load trends if not provided
        if trends is None:
            with open('trends.json', 'r', encoding='utf-8') as f:
                trends = json.load(f)

        # Show available categories and regions
        available_categories = set(trend.get('category', 'Unknown') for trend in trends)
        available_regions = set(trend.get('region', 'Unknown') for trend in trends)
        print("Available categories:", available_categories)
        print("Available regions:", available_regions)

        # Filter by category if needed
        if "Hotest" not in categories:
            trends = [trend for trend in trends if trend.get('category') in categories]

        # Always filter by region (default {"US"})
        if "All" not in regions:  # Optional: you can add "All" if you want no region filtering
            trends = [trend for trend in trends if trend.get('region') in regions]

        print(f"Filtered {len(trends)} trends (categories: {categories}, regions: {regions})")
        

        builder = LogicalTrendGraphBuilder(trends)
        builder.build_graph(threshold=0.5)
        messy_semantic = builder.plot_graph()
        print("Graph saved at:", messy_semantic)
    except Exception as e:
        print(f"❌ Failed to generate logical graph: {e}")
        messy_semantic = "error.png"

    # Statistical graph
    try:
        builder = StatisticalTrendGraphBuilder(trends[:])
        builder.build_graph(threshold=0.5)
        clean_semantic = builder.plot_graph()
        print("Graph saved at:", clean_semantic)
    except Exception as e:
        print(f"❌ Failed to generate statistical graph: {e}")
        clean_semantic = "error.png"
        # return {"error": "Graph generation failed. Cannot proceed."}

    # Build story
    story = builder.extract_story_path()
    prompt = build_writer_prompt(story[:5], language, tone, theme, style)

    print("\nPrompt being sent to model...\n")

    # ****************************************************************** image verison only region and category DOES NTO WORK PIPE USED BEFORE REFERENCE ERROR

    try:
        start_time = time.time()
        response = requests.post(
           'http://host.docker.internal:11434/api/generate', #    'http://localhost:11434/api/generate
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            # timeout=160
        )
        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            output = response.json()
            generated_story = output['response']
            print("Generated Story:\n", generated_story)
            print(f"\n⏱️ Time taken: {elapsed_time:.2f}s")

            result = {
                "story": generated_story,
                "graph_logical": messy_semantic,
                "graph_statistical": clean_semantic
            }

            # --- IMAGE GENERATION PART ---
            if image:
                try:
                    # Load Stable Diffusion ONLY NOW
                    if pipe is None:
                        from diffusers import StableDiffusionPipeline
                        import torch
                        print("Loading Stable Diffusion model...")
                        pipe = StableDiffusionPipeline.from_pretrained(
                            "runwayml/stable-diffusion-v1-5",
                            torch_dtype=torch.float16
                        )
                        pipe = pipe.to("cuda")

                    # Collect all keywords from selected trends
                    all_keywords = []
                    for trend in story[:5]:
                        all_keywords.extend(trend.get("keywords", []))
                    all_keywords = list(set(all_keywords))  # remove duplicates

                    # Build image prompt
                    img_prompt_base = f"Write a short direct prompt to generate an image based on keywords : {', '.join(all_keywords)}"
                    # Take only part AFTER ":"
                    img_prompt = img_prompt_base.split(":", 1)[-1].strip()

                    print("\nSending prompt to Stable Diffusion...\n")

                    # Generate image
                    image_output = pipe(img_prompt).images[0]

                    # Save image
                    if not os.path.exists("generated_images"):
                        os.makedirs("generated_images")
                    image_path = f"generated_images/generated_image_{int(time.time())}.png"
                    image_output.save(image_path)

                    result["image_path"] = image_path

                    print(f" Image saved at {image_path}")
                    elapsed_time = time.time() - start_time
                    print(elapsed_time, "******************************************")

                except Exception as e_img:
                    print(f"❌Image generation failed: {e_img}")
                    result["image_error"] = str(e_img)

            return result

        else:
            print(f"❌ Model error: {response.status_code}, {response.text}")
            return {"error": "Model failed", "graph_logical": messy_semantic, "graph_statistical": clean_semantic}

    except Exception as e:
        print(f"❌ Model request failed: {e}")
        return {"error": "Model call exception", "graph_logical": messy_semantic, "graph_statistical": clean_semantic}

    except Exception as ex:
        print(f"❌ story_maker error: {ex}")
        return {"error": "story_maker internal error"}
    
    
    # ******************************************************************non image verison only region and category
    # Send prompt to model
#     try:
#         start_time = time.time()
#         response = requests.post(
#             'http://host.docker.internal:11434/api/generate',  #ollama model simple, no docker cmd 'http://localhost:11434/api/generate'
#             # 'http://localhost:11434/api/generate',
#             json={
#                 "model": "llama3",
#                 "prompt": prompt,
#                 "stream": False
#             },
#             timeout=160  # Optional: avoids hanging forever
#         )
#         elapsed_time = time.time() - start_time

#         if response.status_code == 200:
#             output = response.json()
#             if language != "English":
#                 text = output['response']
#                 # result = text.split(':', 1)[-1].strip()
#                 sections = []
#                 parts = text.split('********') 

#                 for i in range(1, len(parts), 2):  # Step by 2 to get between pairs
#                     if i < len(parts):
#                         sections.append(parts[i].strip())

#                 # Join all sections with newlines or process as needed
#                 result = '\n'.join(sections)
#                 print("Generated Story:\n", text)
#                 print(f"\n⏱️ Time taken: {elapsed_time:.2f}s")

#                 return {
#                 "story": result,
#                 "graph_logical": messy_semantic,
#                 "graph_statistical": clean_semantic
#             }

#             print("Generated Story:\n", output['response'])
#             print(f"\n⏱️ Time taken: {elapsed_time:.2f}s")
    
#             return {
#                 "story": output["response"],
#                 "graph_logical": messy_semantic,
#                 "graph_statistical": clean_semantic
#             }
#         else:
#             print(f"❌ Model error: {response.status_code}, {response.text}")
#             return {"error": "Model failed", "graph_logical": messy_semantic, "graph_statistical": clean_semantic}

#     except Exception as e:
#         print(f"❌ Model request failed: {e}")
#         return {"error": "Model call exception", "graph_logical": messy_semantic, "graph_statistical": clean_semantic}



if __name__ == '__main__':
    story_maker()

    
    