import yake
import json
import feedparser
import re
import socket
import time
# import schedule
import datetime
# import spacy
from googleapiclient.discovery import build
from langdetect import detect
from googletrans import Translator
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from scipy.special import softmax
import numpy as np
import emoji
# from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
import os
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from dateutil import parser

# * * * * * /usr/bin/python /home/app/TrendExtrAnalyzer.py >> /home/app/cron_output.log 2>&1  cron job run every minute

# Load spaCy model
# nlp = spacy.load("en_core_web_sm")

# Load HuggingFace topic classifier
# tokenizer_hf = AutoTokenizer.from_pretrained("cardiffnlp/tweet-topic-21-multi")
# model_hf = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/tweet-topic-21-multi")
# hf_labels = model_hf.config.id2label

download('vader_lexicon')
translator = Translator()
with open("/home/app/cron_output.log", "a") as f:
    f.write("Script ran at: {}\n".format(__import__('datetime').datetime.now()))


# YouTube category map
YOUTUBE_CATEGORIES = {
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

YOUTUBE_API_KEY = 'AIzaSyCt8FPpasI--1HHAAuq3-8ijQDr-vmsh5M'
REGIONS = ['US', 'GB', 'PK']
TRENDS_FILE = 'trends.json'
FETCH_INTERVAL_MINUTES = 1

# ---- INITIALIZE ---- #
# emoji_pattern = re.compile(r'\p{Emoji}', flags=re.UNICODE)

# **************************************************************** HELPERS **************************************************************** #
def clean_text(text):
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        if not text:
            return ""

        text = re.sub(r'<.*?>', '', text)
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'&nbsp;?', ' ', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[\u2022\u2605\u25BA\u25AA️\U0001F3B6]', '', text)
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        text = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', text)
        text = re.sub(r'[^A-Za-z0-9.,;!?()\'" \n]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    except ValueError as ve:
        print(f"Error: {ve}")
        return ""

    except Exception as e:
        print(f"An unexpected error occurred in clean_text: {e}")
        return ""

def process_language(text):
    try:
        lang = detect(text)
    except:
        lang = 'unknown'
    return lang

def translate_if_needed(text, lang):
    if lang != 'en':
        try:
            return translator.translate(text, dest='en').text
            # return text
        except:
            return "[Translation failed]"
    return None

# def extract_keywords(text):
#     words = re.findall(r'\b[A-Z][a-z]+\b', text)
#     return list(set(words))[:5]

def extract_keywords(text, top_k=5):
    """Extract top K keywords based on TF-IDF scores."""
    if not text:
        return []

    # TF-IDF needs a corpus, so wrap text in list
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])

    scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    keywords = [word for word, score in sorted_scores[:top_k]]
    return keywords

def extract_keywords_yake(text, top_k=5):
    """Extract top K keywords using YAKE."""
    if not text:
        return []
    
    try:
        custom_kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=top_k)
        keywords = custom_kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]
    except Exception as e:
        print(f"YAKE error: {e}")
        return []
    

# def get_sentiment(text):
#     blob = TextBlob(text)
#     return blob.sentiment.polarity

# def calculate_sentiment(text):
#     """Simple sentiment scoring placeholder."""
#     if not text:
#         return 0
#     try:
#         blob = TextBlob(text)
#         return blob.sentiment.polarity  # -1 to 1
#     except Exception as e:
#         print(f"Sentiment error: {e}")
#         return 0

def calculate_sentiment(text):
    try:
        # Check if the text is valid
        if not text:
            raise ValueError("Text cannot be empty.")
        
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(text)['compound']
        
        # Return sentiment score
        return score
    
    except ValueError as ve:
        print(f"Error: {ve}")
        return None  # Return None or any other value indicating failure

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None 

# ---- INTERNET CHECKER ---- #
def is_internet_available():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False


def detect_and_translate(text):
    try:
        lang = detect(text)
        if lang != 'en':
            return translator.translate(text, dest='en').text
    except:
        return text
    return text

def classify_news_topic_cardiff(text):
    try:
        # inputs = tokenizer_hf(text, return_tensors="pt", truncation=True, padding=True)
        # outputs = model_hf(**inputs)
        # scores = softmax(outputs.logits.detach().numpy()[0])
        # top_index = int(np.argmax(scores))
        # label = hf_labels[top_index]
        label = "News & Politics"
        return label
    except Exception as e:
        print(f"Classification error: {e}")
        return "General"

# News topic classification using NER and keywords
# def categorize_news_with_spacy(text):
#     doc = nlp(text)
    
#     # Extract entities and categorize
#     ents = {ent.label_: ent.text.lower() for ent in doc.ents}
    
#     # Categorizing based on recognized entities
#     if "election" in ents.get("EVENT", []) or "government" in ents.get("ORG", []):
#         return "Politics"
#     elif "match" in ents.get("EVENT", []) or "sport" in ents.get("MISC", []):
#         return "Sports"
#     elif "concert" in ents.get("EVENT", []) or "music" in ents.get("MISC", []):
#         return "Music"
#     elif "AI" in ents.get("TECHNOLOGY", []):
#         return "Technology"
#     elif "fashion" in ents.get("MISC", []):
#         return "Fashion"
#     elif "movie" in ents.get("MISC", []) or "film" in ents.get("MISC", []):
#         return "Entertainment"
#     elif "war" in ents.get("MISC", []):
#         return "World Affairs"
#     elif "education" in ents.get("MISC", []):
#         return "Education"
#     return "General"

def normalize(value, max_value=1.5):
    return round(min(value / max_value, 1.0), 3)


# **************************************************ANALYZER**********************************************************
def calculate_importance_youtube(trend):
    """Importance based on views, likes, duration, HD definition."""
    try:
        # Safely get view count
        try:
            view_count = int(trend.get('viewCount', 0))
        except (ValueError, TypeError):
            view_count = 0

        # Safely get like count
        try:
            like_count = int(trend.get('likeCount', 0))
        except (ValueError, TypeError):
            like_count = 0

        # Safely parse duration
        duration = trend.get('duration') or 'PT0M0S'
        minutes = 0
        try:
            match = re.match(r'PT(\d+M)?(\d+S)?', duration)
            if match:
                minutes_part, seconds_part = match.groups()
                if minutes_part:
                    minutes += int(minutes_part.replace('M', ''))
                if seconds_part:
                    minutes += int(seconds_part.replace('S', '')) / 60
        except Exception:
            minutes = 0

        # Check HD definition
        definition = trend.get('definition', 'sd')
        hd_bonus = 1.2 if definition == 'hd' else 1.0

        # Calculate scores
        basic_score = (view_count + 2 * like_count) / 1000
        duration_bonus = min(minutes / 10, 1.5)
        importance = basic_score * duration_bonus * hd_bonus
        return round(importance, 3)
    
    except Exception as e:
        print(f"Error in calculate_importance_youtube: {e}")
        return 0


def parse_published_date(published_str):
    """Parses published date and makes it timezone-naive in UTC."""
    try:
        dt = parser.parse(published_str)
        if dt.tzinfo:
            # Convert to UTC and make naive
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception as e:
        print(f"Date parsing error: {e}")
        return None

def calculate_importance_news(trend):
    """Importance based on how fresh the news article is."""
    published_str = trend.get('published')
    if not published_str:
        return 1.0  # Default if no date

    published_dt = parse_published_date(published_str)
    if not published_dt:
        return 1.0

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    age_in_hours = (now - published_dt).total_seconds() / 3600

    if age_in_hours < 1:
        return 1.5  # Very fresh
    elif age_in_hours < 6:
        return 1.2
    elif age_in_hours < 24:
        return 1.0
    else:
        return 0.8  # Older news


def calculate_relevance(text):
    """Keyword density -> relevance."""
    try:
        if not text:
            return 0
        words = text.split()
        keywords = extract_keywords_yake(text)  # Assuming this function extracts keywords.
        if not words:
            return 0
        relevance = len(keywords) / len(words)
        return round(relevance * 10, 3)  # Scale up a bit
    except Exception as e:
        print(f"Error calculating relevance: {e}")
        return 0  # Return a default value in case of error


def calculate_authenticity(source):
    """Simple source-based authenticity."""
    if source == 'YouTube':
        return 0.5
    elif source == 'Google News':
        return 0.9
    return 0.7

def calculate_info_gain_tfidf(text):
    """Info gain based on total TF-IDF richness."""
    try:
        if not text:
            return 0

        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform([text])
        total_tfidf_score = X.sum()
        normalized_score = min(total_tfidf_score / 5, 1.0)  # Scale to [0,1], 5 is an arbitrary richness
        return round(normalized_score, 3)
    
    except Exception as e:
        print(f"Error in calculate_info_gain_tfidf: {e}")
        return 0
    
def calculate_info_gain(text):
    """Info gain based on text richness and length."""
    try:
        if not text:
            return 0
        words = re.findall(r'\b\w+\b', text)
        unique_words = set(words)

        richness = len(unique_words) / (len(words) + 1)  # Avoid divide by zero
        length_factor = min(len(words) / 100, 1.0)  # Cap at 1

        info_gain = richness * length_factor
        return round(info_gain, 3)
    except Exception as e:
        print(f"Error in calculate_info_gain: {e}")
        return 0


# **********************************************************FETCHING TRENDS *************************************************************

# Fetch YouTube Trends
def fetch_youtube_trends(start_id, region='US'):
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.videos().list(
            part='snippet,statistics,contentDetails',
            chart='mostPopular',
            maxResults=10,
            regionCode=region
        )
        response = request.execute()

        trends = []
        i = 0
        for item in response.get('items', []):
            try:
                snippet = item['snippet']
                statistics = item.get('statistics', {})
                content = item.get('contentDetails', {})
                
                title = clean_text(snippet.get('title', ''))
                desc = clean_text(snippet.get('description', ''))
                category_id = snippet.get('categoryId', '0')
                category = YOUTUBE_CATEGORIES.get(category_id, "Unknown")
                combined_text = f"{title} {desc}"
                
                language = process_language(combined_text)
                translated = translate_if_needed(combined_text, language)

                trend = {
                    'Id': start_id + i,
                    'source': 'YouTube',
                    'region': region,
                    'title': title,
                    'description': desc,
                    'category': category,
                    'publishedAt': snippet.get('publishedAt'),
                    'channelTitle': snippet.get('channelTitle'),
                    'viewCount': statistics.get('viewCount'),
                    'likeCount': statistics.get('likeCount'),
                    'commentCount': statistics.get('commentCount'),
                    'duration': content.get('duration'),
                    'definition': content.get('definition'),
                    'language': language,
                    **({'translated': translated} if translated else {})
                }

                trend['sentiment'] = calculate_sentiment(combined_text)
                trend['importance'] = normalize(calculate_importance_youtube(trend))
                trend['relevance'] = normalize(calculate_relevance(combined_text))
                trend['authenticity'] = normalize(calculate_authenticity(trend['source']))
                trend['info_gain'] = normalize(calculate_info_gain(combined_text))
                trend['keywords'] = extract_keywords_yake(combined_text)

                trends.append(trend)
                i += 1

            except Exception as e:
                print(f"Error processing trend {i}: {e}")
                continue  # Skip to next item if there is an error in the current trend

        return trends

    except Exception as e:
        print(f"Error fetching YouTube trends: {e}")
        return []  # Return empty list if the API call or outer processing fails


# Fetch Google News Trends
def fetch_google_news_trends(start_id, region='US'):
    try:
        rss_url = f'https://news.google.com/rss?hl=en-{region}&gl={region}&ceid={region}:en'
        feed = feedparser.parse(rss_url)
        
        # If the feed fails to load or is malformed, return an empty list
        if feed.bozo:
            print("Error: Malformed RSS feed.")
            return []

        trends = []
        for i, entry in enumerate(feed.entries[:10]):
            try:
                title = detect_and_translate(clean_text(entry.get('title', '')))
                summary = detect_and_translate(clean_text(entry.get('summary', '')))
                category = classify_news_topic_cardiff(title + ' ' + summary)
                combined_text = f"{title} {summary}"

                trend = {
                    'Id': start_id + i,
                    'source': 'Google News',
                    'region': region,
                    'title': title,
                    'summary': summary,
                    'category': category,
                    'link': entry.get('link'),
                    'published': entry.get('published')
                }

                trend['sentiment'] = calculate_sentiment(combined_text)
                trend['importance'] = normalize(calculate_importance_news(trend))
                trend['relevance'] = normalize(calculate_relevance(combined_text))
                trend['authenticity'] = normalize(calculate_authenticity(trend['source']))
                trend['info_gain'] = normalize(calculate_info_gain(combined_text))
                trend['keywords'] = extract_keywords_yake(combined_text)

                trends.append(trend)
            except Exception as e:
                print(f"Error processing entry {i}: {e}")

        return trends

    except Exception as e:
        print(f"Error fetching Google News trends: {e}")
        return []  # Return an empty list if there's an error at the top level

# Save Trends
def save_trends(trends):
    new_trends_count = 0  # To track how many new trends are added
    existing_trends = []

    try:
        # Check if the trends file exists and try to load the existing trends
        if os.path.exists(TRENDS_FILE):
            with open(TRENDS_FILE, 'r', encoding='utf-8') as f:
                try:
                    existing_trends = json.load(f)
                    # Check if existing_trends is a valid list, if not initialize it as an empty list
                    if not isinstance(existing_trends, list):
                        existing_trends = []
                except json.JSONDecodeError:
                    print("Error decoding the JSON file. Initializing an empty list for existing trends.")
                    existing_trends = []  # If there's a decode error, assume it's empty

        # Set for tracking seen trends based on title and URL
        seen = set()
        unique = []

        # Loop through both existing and new trends
        for trend in existing_trends + trends:
            try:
                # Assuming 'title' and 'url' are the main identifiers for uniqueness
                key = (trend['title'].lower(), trend.get('url', '').lower())  # Case-insensitive comparison
                
                if key not in seen:
                    seen.add(key)
                    unique.append(trend)
                    if trend not in existing_trends:  # If the trend is not already in existing, count it as new
                        new_trends_count += 1
            except KeyError as e:
                print(f"Missing expected key {e} in trend: {trend}")
            except Exception as e:
                print(f"Unexpected error occurred while processing trend: {e}")

        # Try saving the unique trends back to the file
        try:
            with open(TRENDS_FILE, 'w', encoding='utf-8') as f:
                json.dump(unique, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error writing to file {TRENDS_FILE}: {e}")
        except Exception as e:
            print(f"Unexpected error occurred while saving trends: {e}")
    
    except Exception as e:
        print(f"Error occurred while processing trends: {e}")
        return 0
    
    return new_trends_count  # Return the number of new trends added



# HELPER FOR TREND ID LOADING
# Function to load the last trend ID with error handling
def load_last_trend_id():
    trend_id_file = '/home/app/last_trend_id.txt'
    try:
        # Try to read the last trend ID from the file
        if os.path.exists(trend_id_file):
            with open(trend_id_file, 'r') as file:
                return int(file.read().strip())
        return 0  # Return 0 if no previous ID is found (first run)
    except Exception as e:
        print(f"Error loading last trend ID: {e}")
        return 0  # In case of any error, return 0 and proceed

# Function to save the last trend ID with error handling
def save_last_trend_id(trend_id):
    trend_id_file = '/home/app/last_trend_id.txt'
    try:
        # Try to save the current trend ID to the file
        with open(trend_id_file, 'w') as file:
            file.write(str(trend_id))
    except Exception as e:
        print(f"Error saving last trend ID: {e}")

# Main fetcher
def fetch_trends_task():
    try:
        if not is_internet_available():
            print("No internet. Retrying later.")
            return

        all_trends = []
        trend_id = load_last_trend_id()

        for region in REGIONS:
            try:
                yt_trends = fetch_youtube_trends(trend_id, region)
                trend_id += len(yt_trends)
                all_trends.extend(yt_trends)
            except Exception as e:
                print(f"Error fetching YouTube trends for region {region}: {e}")

            try:
                news_trends = fetch_google_news_trends(trend_id, region)
                trend_id += len(news_trends)
                all_trends.extend(news_trends)
            except Exception as e:
                print(f"Error fetching Google News trends for region {region}: {e}")

        try:
            new_trends_num = save_trends(all_trends)
        except Exception as e:
            print(f"Error saving trends: {e}")
            return

        try:
            save_last_trend_id(trend_id)
        except Exception as e:
            print(f"Error saving last trend ID: {e}")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Saved {new_trends_num} new trends.")

    except Exception as e:
        print(f"Error in fetch_trends_task: {e}")


# Scheduler
# def run_scheduler():
#     schedule.every(FETCH_INTERVAL_MINUTES).minutes.do(fetch_trends_task)
#     print(f"Running every {FETCH_INTERVAL_MINUTES} minutes...")
#     fetch_trends_task()
#     schedule.run_pending()
    # time.sleep(10)
    # break


# Start
if __name__ == "__main__":
    # run_scheduler()
    fetch_trends_task()
