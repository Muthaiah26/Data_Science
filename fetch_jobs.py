# fetch_jobs.py
import requests
import time
import re
from pymongo import MongoClient
from indexing import add_job_to_index  # optional: to auto-update FAISS

# === Mongo Setup ===
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "job_recommender"
COLLECTION = "jobs"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
jobs_col = db[COLLECTION]

# === Adzuna API ===
ADZUNA_APP_ID = "b5815081"
ADZUNA_APP_KEY = "c30e1f7e14294dcae4d292dc6225460a"
BASE_URL = "https://api.adzuna.com/v1/api/jobs/in/search/1"

# === Utility functions ===
def clean_text(text):
    text = re.sub(r"<[^>]+>", "", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_skills(text, skills_db):
    found_skills = set()
    cleaned_text = text.lower()
    for skill in skills_db:
        # Use regex word boundary to avoid partial matches (e.g., "react" in "reactivate")
        if re.search(r'\b' + re.escape(skill) + r'\b', cleaned_text):
            found_skills.add(skill)
    return list(found_skills)

# === Fetch Adzuna ===
def fetch_adzuna(query="", location="India", results_per_page=25):
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": query,
        "where": location,
        "results_per_page": results_per_page,
        "content-type": "application/json"
    }
    print(f"Fetching {query} jobs from Adzuna...")
    r = requests.get(BASE_URL, params=params, timeout=10)
    r.raise_for_status()
    results = r.json().get("results", [])
    print(f"Fetched {len(results)} jobs from Adzuna.")
    return results

# === Insert into MongoDB ===
def ingest_jobs_to_mongo(jobs,master_skills_list):
    count = 0
    for j in jobs:
        doc = {
            "_id": j.get("id"),
            "title": j.get("title"),
            "company": j.get("company", {}).get("display_name"),
            "location": j.get("location", {}).get("display_name"),
            "description": j.get("description"),
            "description_clean": clean_text(j.get("description", "")),
            "posted_timestamp": time.time(),
            "source": "adzuna",
            "skills_list": extract_skills(j.get("description", ""), master_skills_list)
        }
        jobs_col.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        add_job_to_index(doc)  # update FAISS index live
        count += 1
    print(f"Inserted or updated {count} jobs into MongoDB.")

# === Run manually ===
if __name__ == "__main__":
    try:
        from app import load_master_skills
        print("Loading skills for manual run...")
        manual_skills_db = load_master_skills('jobs_skills.csv')
    except ImportError:
        print("Could not load master skills list. Using small default list.")
        manual_skills_db = ["python", "java", "sql", "aws", "machine learning"]

    jobs = fetch_adzuna("data scientist", "India")
    ingest_jobs_to_mongo(jobs, manual_skills_db) # <-- Pass the skills list
    print("Job ingestion complete.")
