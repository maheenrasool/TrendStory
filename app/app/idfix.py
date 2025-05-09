import json

TRENDS_FILE = 'trends.json'

def renumber_ids(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        trends = json.load(f)

    if not isinstance(trends, list):
        raise ValueError("Expected a list of trends in JSON.")

    for i, trend in enumerate(trends, start=1):
        trend['Id'] = i

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(trends, f, indent=2, ensure_ascii=False)

    print(f"Renumbered {len(trends)} trend IDs.")

if __name__ == "__main__":
    renumber_ids(TRENDS_FILE)
