import fitz 
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import os
import tempfile
import traceback
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time 
import math
from search_utils import find_similar_jobs_by_embedding
import joblib
from indexing import build_faiss_index, add_job_to_index
from fetch_jobs import ingest_jobs_to_mongo, fetch_adzuna




RANKER = joblib.load("ranker_model.pkl")


API_KEY = "AIzaSyDYIJOkAkz7Rlijxi4EuAUYUdPFjTEhzNw"
try:
    genai.configure(api_key=API_KEY)
    
    
    generation_config = {
      "temperature": 0.2,
      "top_p": 1,
      "top_k": 1,
      "max_output_tokens": 2048,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    
    llm_model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-09-2025",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)
    print("Gemini model loaded successfully.")

except Exception as e:
    print(f"Error configuring Generative AI: {e}")
    llm_model = None



app = Flask(__name__, template_folder='templates')


SKILLS_DB = []

print("🔄 Building FAISS index from MongoDB jobs...")
build_faiss_index()
print("✅ FAISS index built successfully and ready for search.")

print("Loading sentence embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")


def get_full_job_description(job_url, headers):
    """
    Visits a single job URL and scrapes its full description text.
    Uses the selector you found earlier!
    """
    try:
        response = requests.get(job_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # This is the selector you were finding before!
        # It targets the main description box on the right.
        desc_container = soup.select_one(".jobs-description__container")
        
        if desc_container:
            # Use .get_text() to extract all text, joining with a space
            return desc_container.get_text(separator=" ", strip=True)
        
        # Fallback if the first selector fails
        desc_fallback = soup.select_one(".description__text")
        if desc_fallback:
             return desc_fallback.get_text(separator=" ", strip=True)

        print(f"Warning: Could not find description container for {job_url}")
        return ""  # Return empty if no description is found
        
    except requests.exceptions.RequestException as e:
        print(f"Error scraping full description for {job_url}: {e}")
        return ""


def load_master_skills(csv_path='jobs_skills.csv'):
    """
    Loads the master list of all skills from a separate CSV file.
    Handles a 'Skills' column with semicolon-separated values.
    """
    try:
        
        skills_df = pd.read_csv(csv_path)
        
        
        if 'Skills' in skills_df.columns:
            skills_column = skills_df['Skills']
        elif 'skills' in skills_df.columns:
            skills_column = skills_df['skills']
        else:
            skills_column = skills_df.iloc[:, 0]
            print(f"Warning: 'Skills' or 'skills' column not found in '{csv_path}'. Falling back to first column.")
        
        all_skills_set = set()
        
     
        for skill_string in skills_column.dropna():
            
            skills_in_row = str(skill_string).split(';')
            
            
            for skill in skills_in_row:
                cleaned_skill = skill.strip().lower()
                if cleaned_skill: 
                    all_skills_set.add(cleaned_skill)
        
        skills_list = list(all_skills_set)
        print(f"Dynamically loaded {len(skills_list)} unique skills from '{csv_path}'.")
        return skills_list
    except FileNotFoundError:
        print(f"Warning: Master skills file '{csv_path}' not found. SKILLS_DB will be empty. Resume skill extraction will not work.")
        return []
    except Exception as e:
        print(f"Error reading master skills CSV '{csv_path}': {e}")
        print("Assuming single column, no header. Please check the file format.")
        return []

def load_and_preprocess_jobs(csv_path='jobs.csv'):
    try:
        jobs_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: '{csv_path}' not found. Please create it.")
        return pd.DataFrame(), None 
    except Exception as e:
        print(f"Error reading CSV '{csv_path}': {e}")
        return pd.DataFrame(), None  

   
    if 'description' not in jobs_df.columns:
        print("Error: 'jobs.csv' must have 'description' column.")
        return pd.DataFrame(), None
        
    print(f"Loading and preprocessing {len(jobs_df)} jobs from '{csv_path}'...")
    
    jobs_df['description'] = jobs_df['description'].astype(str)
    jobs_df['description_clean'] = jobs_df['description'].apply(clean_text)
    
    print("Extracting skills from job descriptions using master skills list...")
    jobs_df['skills_list'] = jobs_df['description_clean'].apply(lambda x: extract_skills(x, SKILLS_DB))
    
    print("Computing job embeddings...")
    job_embeddings = model.encode(jobs_df['description_clean'].tolist(), show_progress_bar=True)
    
    print("Job processing complete.")
    return jobs_df, job_embeddings

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.-]', '', text)
    return text.strip()

def extract_skills(text, skills_list):
    """
    Extracts skills from resume text *based on the dynamically loaded skills_list*.
    """
    found_skills = set()
    cleaned_text = clean_text(text) # Clean the resume text once
    for skill in skills_list:
        
        if re.search(r'\b' + re.escape(skill) + r'\b', cleaned_text, re.IGNORECASE):
            found_skills.add(skill)
    return list(found_skills)

def extract_text_from_pdf(file_storage):
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            file_storage.save(temp_file)
            temp_file_path = temp_file.name

        with fitz.open(temp_file_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    
    except Exception as e:
        print(f"---!! PDF EXTRACTION FAILED !!---")
        print(traceback.format_exc())
        return None
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def extract_text_from_txt(file_storage):
    try:
        return file_storage.read().decode('utf-8')
    except UnicodeDecodeError:
        try:
            file_storage.seek(0)
            return file_storage.read().decode('latin-1')
        except Exception as e:
            print(f"---!! TXT EXTRACTION FAILED !!---")
            print(traceback.format_exc())
            return None

def calculate_skill_match_score(resume_skills, job_skills):
    if not resume_skills and not job_skills:
        return 1.0
    if not resume_skills or not job_skills:
        return 0.0

    resume_set = set(resume_skills)
    job_set = set(job_skills)
    
    intersection = len(resume_set.intersection(job_set))
    union = len(resume_set.union(job_set))
    
    return intersection / union if union > 0 else 0.0

def get_gemini_response(prompt_text):
    if not llm_model:
        return "Gemini model is not loaded. Please check API key and configuration."
    
    max_retries = 5
    delay = 1
    for i in range(max_retries):
        try:
            prompt_parts = [prompt_text]
            response = llm_model.generate_content(prompt_parts)
            return response.text
        except Exception as e:
            print(f"Gemini API call failed (attempt {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                return f"Error communicating with Gemini API after {max_retries} attempts."

SKILLS_DB = load_master_skills('jobs_skills.csv')


jobs_df, job_embeddings = load_and_preprocess_jobs() 




@app.route('/')
def home():
    return render_template('index.html')



@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # --- 1. Receive Resume ---
        if 'resume' not in request.files:
            return jsonify({"error": "No resume file provided"}), 400
        file = request.files['resume']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        file_name = file.filename.lower()
        if file_name.endswith('.pdf'):
            resume_text = extract_text_from_pdf(file)
        elif file_name.endswith('.txt'):
            resume_text = extract_text_from_txt(file)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        if not resume_text:
            return jsonify({"error": "Failed to extract resume text"}), 500

        # --- 2. Extract skills ---
        resume_skills = extract_skills(resume_text, SKILLS_DB)
        resume_clean = clean_text(resume_text)
        resume_vec = model.encode([resume_clean])

        if resume_skills:
            # --- 1. Get the top 3 skills to query individually ---
            # We take 3 to get a good variety without too many API calls
            top_skills_to_query = resume_skills[:3] 
            print(f"🕵️  Dynamic fetch: Will query for top 3 skills: {top_skills_to_query}")

            all_new_jobs = [] # A list to hold all jobs from all queries
            
            # --- 2. Loop over each skill and fetch jobs for it ---
            for skill_query in top_skills_to_query:
                print(f"--- Querying Adzuna for '{skill_query}' ---")
                try:
                    # Fetch 5-10 jobs for each skill
                    new_jobs_for_skill = fetch_adzuna(query=skill_query, location="India", results_per_page=10) 
                    
                    if new_jobs_for_skill:
                        print(f"📥 Found {len(new_jobs_for_skill)} jobs for '{skill_query}'.")
                        all_new_jobs.extend(new_jobs_for_skill) # Add them to the main list
                    else:
                        print(f"No new jobs found for query: '{skill_query}'")
                
                except Exception as e:
                    # Don't fail the whole request if one skill query fails
                    print(f"Error during Adzuna fetch for '{skill_query}': {e}")

            # --- 3. Ingest all collected jobs at once ---
            if all_new_jobs:
                # IMPORTANT: De-duplicate the list, as two skills might return the same job
                unique_jobs_dict = {job['id']: job for job in all_new_jobs}
                unique_jobs_list = list(unique_jobs_dict.values())
                
                print(f"Total new jobs found: {len(all_new_jobs)}")
                print(f"Total unique new jobs: {len(unique_jobs_list)}. Ingesting...")
                
                # Ingest the unique jobs into Mongo and FAISS
                ingest_jobs_to_mongo(unique_jobs_list, SKILLS_DB)
            else:
                print("No new jobs found for any of the top skills.")
        
        else:
            print("No resume skills found, skipping dynamic job fetch.")

        # --- 3. Find candidate jobs using FAISS ---
        print("🔍 Searching FAISS for similar jobs...")
        candidates = find_similar_jobs_by_embedding(resume_clean, top_k=50)
        if not candidates:
            print("❌ Still no candidates found even after rebuild.")
            return jsonify({"error": "No similar jobs found even after FAISS rebuild"}), 404
        print(f"👍 Found {len(candidates)} candidate jobs from FAISS.")

        # --- 4. Compute ranking features ---
        def compute_features(resume_skills, job_doc, resume_text):
            # skill overlap ratio
            job_skills = job_doc.get("skills_list", [])
            skill_intersection = len(set(resume_skills).intersection(set(job_skills)))
            skill_union = len(set(resume_skills).union(set(job_skills))) or 1
            skill_ratio = skill_intersection / skill_union

            # vector similarity score returned from FAISS
            vec_score = job_doc.get("vector_score", 0.0)

            # recency (days)
            created_ts = job_doc.get("posted_timestamp")
            recency_days = 365
            if created_ts:
                recency_days = (time.time() - created_ts) / (24*3600)
            recency_score = max(0, 1 - recency_days / 90)

            # location match (1 or 0)
            loc_score = 0
            user_loc = "India"  # later can detect automatically from resume
            if user_loc and job_doc.get("location") and user_loc.lower() in job_doc.get("location").lower():
                loc_score = 1

            return [vec_score, skill_ratio, recency_score, loc_score]

        # --- 5. Rank the candidates using ML model ---
        X = [compute_features(resume_skills, j, resume_text) for j in candidates]
        X=np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.size == 0:
            print("⚠️ Warning: No features to predict. Returning empty list.")
            return jsonify({"matched_jobs": []})

        preds = RANKER.predict_proba(X)[:, 1] if hasattr(RANKER, "predict_proba") else RANKER.predict(X)

        for j, score in zip(candidates, preds):
            j["final_score"] = float(score)

        candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
        top_jobs = candidates[:10]
       

        # --- 6. AI summary via Gemini ---
        summary_prompt = f"Here is a resume:\n\n{resume_text}\n\nAct as a recruiter. Write a 3-sentence summary of this candidate."
        ai_summary = get_gemini_response(summary_prompt)

        # --- 7. AI explanation for top job match ---
        ai_match_explanation = ""
        if top_jobs:
            top_job = top_jobs[0]
            full_desc_for_ai = top_job.get("description", "")
            match_prompt = f"Resume:\n{resume_text}\n\nJob Description:\n{full_desc_for_ai}\n\nExplain 3 key reasons why this candidate fits this job."
            ai_match_explanation = get_gemini_response(match_prompt)

        # --- 8. Final Response ---
        return jsonify({
            "resume_skills": resume_skills,
            "matched_jobs": top_jobs,
            "ai_summary": ai_summary,
            "ai_match_explanation": ai_match_explanation
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)

