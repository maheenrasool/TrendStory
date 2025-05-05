import time
import requests
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dateutil import parser
from datetime import datetime, timezone

class CleanTrendGraphBuilder:
    def __init__(self, trends):
        self.trends = trends
        self.graph = nx.Graph()

    def build_graph(self, threshold=0.3):
        """Builds a graph based on trend similarity (text + metadata)."""
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

    def plot_graph(self, story_path="clean_trend_graph.png"):
        """Plots the graph with node indices and edge colors based on source similarity."""
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
        # plt.show()
    
        # Save and close
        plt.savefig(filename, format='png')
        plt.close()
        return os.path.abspath(filename)
        

    def extract_story_path(self):
        """Extracts a path that logically connects trends."""
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

    def get_graph(self):
        return self.graph


import networkx as nx
import matplotlib.pyplot as plt
from dateutil import parser
from datetime import datetime, timezone

class MessyTrendGraphBuilder:
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
        keywords1 = set(trend1.get('keywords', []))
        keywords2 = set(trend2.get('keywords', []))
        if not keywords1 or not keywords2:
            return 0
        return len(keywords1 & keywords2) / min(len(keywords1), len(keywords2))

    def category_match_bonus(self, trend1, trend2):
        return 1.0 if trend1.get('category') == trend2.get('category') else 0.0

    def date_proximity_score(self, trend1, trend2):
        date1 = self.parse_published_date(trend1.get('publishedAt') or trend1.get('published'))
        date2 = self.parse_published_date(trend2.get('publishedAt') or trend2.get('published'))
        if not date1 or not date2:
            return 0
        days_apart = abs((date1 - date2).days)
        return max(0, 1 - days_apart / 7)

    def sentiment_similarity_score(self, trend1, trend2):
        s1 = trend1.get('sentiment', 0)
        s2 = trend2.get('sentiment', 0)
        return max(0, 1 - abs(s1 - s2))

    def calculate_connection_weight(self, trend1, trend2):
        return (
            self.keyword_overlap_score(trend1, trend2) * 2 +
            self.category_match_bonus(trend1, trend2) * 1.5 +
            self.date_proximity_score(trend1, trend2) * 1.2 +
            self.sentiment_similarity_score(trend1, trend2) * 1.0
        )

    def determine_edge_color(self, trend1, trend2):
        """Assigns color based on source types."""
        source1 = trend1.get('source', '')
        source2 = trend2.get('source', '')
        if source1 == "YouTube" and source2 == "YouTube":
            return 'red'
        elif source1 == "Google News" and source2 == "Google News":
            return 'green'
        else:
            return 'blue'

    def build_graph(self, threshold=1.5):
        """Builds a semantic graph of trends."""
        n = len(self.trends)
        for i in range(n):
            self.graph.add_node(i, trend=self.trends[i])

        for i in range(n):
            for j in range(i + 1, n):
                weight = self.calculate_connection_weight(self.trends[i], self.trends[j])
                if weight > threshold:
                    color = self.determine_edge_color(self.trends[i], self.trends[j])
                    self.graph.add_edge(i, j, weight=round(weight, 3), color=color)

    def plot_graph(self, filename=r"messy_trend_graph.png"):
        """Plots the graph with colored edges."""
        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(14, 10))

        nx.draw_networkx_nodes(self.graph, pos, node_size=600, node_color='skyblue')

        edges = self.graph.edges(data=True)
        edge_colors = [d['color'] for (u, v, d) in edges]
        edge_weights = [d['weight'] for (u, v, d) in edges]

        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, width=[w for w in edge_weights], alpha=0.7)

        # labels = {i: self.graph.nodes[i]['trend'].get('title', f'Trend {i}')[:20] for i in self.graph.nodes}
        labels = {i: str(i) for i in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)

        plt.title("Trend Semantic Graph", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        # plt.show()
        plt.savefig(filename, format='png')
        plt.close()
        return os.path.abspath(filename)

    def extract_story_path(self):
        """Extracts a connected semantic story path."""
        if not self.graph.nodes:
            return []

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

    def get_graph(self):
        return self.graph


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
        return "No trends provided."

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

    events_text = ""
    categories_used = set()

    for idx, trend in enumerate(trend_chain, 1):
        title = trend.get('title', 'Unknown Event')
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
            desc += f" (Translated: {translation})"

        events_text += f"{idx}. [{source} - {region}] {title}: {desc} (Sentiment Score: {sentiment}) (Category: {category_label})\n"

    regions = set(trend.get('region', 'Unknown Region') for trend in trend_chain)
    
    if Language.lower() != "english":
        prompt = f"""
        Roman Urdu main likho, do not translate to English

aik kahani likho niche diye gaye world event per, roman urdu main:
**Event**:
    {events_text}

Instructions:
- 1-2 paragraph main mazey ki kahani likho 
- takhleeqi banooo
       
neechey likhna shuru kero
    """
    
    
    else:      

        prompt = f"""
    You are a world-class **creative and technical writer AI**.
    
    You are capable of writing in multiple styles—**fiction, blog-style commentary, technical summaries, creative non-fiction, or news-style reporting**.
    
    Your task is to produce a well-structured, high-quality write-up based on recent real-world events. These events are summarized below:
    
    **Writing Directives**:
    - Style: {story_style}
    - Tone/Genre: {tone}
    - Theme (if fictional): {theme}
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
    
    Begin your piece below.
    """
    
    print("***************** PROMPT FOR WRITER MODEL *****************")
    print(prompt)
    return prompt.strip()


def story_maker(trends, language, tone, theme, style):
    builder = MessyTrendGraphBuilder(trends)
    builder.build_graph(threshold=0.5)  # Threshold tunes connection strictness
    messy_semantic = builder.plot_graph()
    print("Graph saved at:", messy_semantic)
    
    builder = CleanTrendGraphBuilder(trends[:])
    builder.build_graph(threshold=0.5)  # You can tune threshold
    clean_semantic = builder.plot_graph()
    print("Graph saved at:", clean_semantic)
    story = builder.extract_story_path()
    prompt = build_writer_prompt(story[:5],language, tone, theme, style)
    
    print("Prompt being sent to model:\n")
    # print(prompt)
    print("\nGenerating story...\n")
    
    # Step 5: Measure time and send to Ollama
    start_time = time.time()
    
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            "model": "llama3.2:latest",  # or whichever model you have
            "prompt": prompt,
            "stream": False  
        }
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Step 6: Print the generated story
    output = {"response": "❌ Failed to generate story."}
    if response.status_code == 200:
        output = response.json()
        print("Generated Story:\n")
        print(output['response'])
        print(f"\n⏱️ Time taken to generate story: {elapsed_time:.2f} seconds")
    else:
        print("Error communicating with o-server!")
    return output['response']



    
    