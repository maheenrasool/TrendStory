import json

# Load existing trends
with open('trends.json', 'r', encoding='utf-8') as f:
    trends = json.load(f)

# Keep only first 3
trends = trends[:3]

# Save back the trimmed list
with open('trends.json', 'w', encoding='utf-8') as f:
    json.dump(trends, f, ensure_ascii=False, indent=2)

print(f"Kept {len(trends)} trends in trends.json.")
