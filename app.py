import fitz 
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile
import traceback
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time 
from linkedin_scraper import scrape_linkedin_jobs
import math



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

print("Loading sentence embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

def scrape_linkedin_job_list(query, location, headers, num_jobs=10):
    """
    Scrapes the LinkedIn "hidden" API for a list of job postings.
    Returns a list of dicts, each with 'title', 'company', and 'link'.
    """
    base_url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
    
    # LinkedIn's API paginates by 25
    num_pages = math.ceil(num_jobs / 25)
    jobs_list = []
    
    for page in range(num_pages):
        start_param = page * 25
        params = {
            "keywords": query,
            "location": location,
            "start": start_param,
            "trk": "public_jobs_jobs-search-bar_search-submit", # This param seems to be required
            "position": 1,
            "pageNum": 0
        }
        
        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()  # Raise an error for bad status codes
            soup = BeautifulSoup(response.text, "html.parser")
            
            # The API returns <li> items directly
            job_cards = soup.find_all("li")
            
            if not job_cards:
                break  # Stop if no more jobs are returned

            for job_card in job_cards:
                # Stop once we've collected the number of jobs we want
                if len(jobs_list) >= num_jobs:
                    break
                    
                title_el = job_card.find("h3", class_="base-search-card__title")
                company_el = job_card.find("h4", class_="base-search-card__subtitle")
                link_el = job_card.find("a", class_="base-card__full-link")
                
                if title_el and company_el and link_el:
                    jobs_list.append({
                        "title": title_el.get_text(strip=True),
                        "company": company_el.get_text(strip=True),
                        "link": link_el["href"],
                        "description": ""  # We'll fill this in the next step
                    })
        
        except requests.exceptions.RequestException as e:
            print(f"Error scraping job list page {page}: {e}")
            break # Stop if we get an error

    return jobs_list[:num_jobs] # Return only the number of jobs requested


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

        # --- 3. Scrape LinkedIn Jobs ---
        
        # Create the search query
        query = " ".join(resume_skills[:5]) or "software engineer"
        
        # Use the realistic headers that work
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Connection": "keep-alive",
            "Referer": "https.www.google.com/",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site"
        }

        # === NEW METHOD (STEP 1: Get Job List) ===
        print(f"Scraping job list for query: {query}")
        jobs = scrape_linkedin_job_list(query, "India", headers, num_jobs=10)
        
        if not jobs:
            return jsonify({"error": "No jobs found or scraper was blocked"}), 404

        # === NEW METHOD (STEP 2: Get Full Description for Each Job) ===
        print(f"Found {len(jobs)} jobs. Fetching full descriptions...")
        for job in jobs:
            # We must scrape each job's page to get its description
            job['description'] = get_full_job_description(job['link'], headers)
            # Add a small delay to avoid getting rate-limited (blocked)
            time.sleep(0.5) 
        print("All descriptions fetched.")

        # --- 4. Skill matching + embedding comparison ---
        # This section will now work, because job['description'] is full!
        
        job_texts = [clean_text(j['description']) for j in jobs]
        job_embeddings = model.encode(job_texts)
        text_similarities = cosine_similarity(resume_vec, job_embeddings)[0]

        # Extract skills from the full description we just scraped
        for job in jobs:
            job['skills_list'] = extract_skills(job['description'], SKILLS_DB)

        skill_scores = [
            calculate_skill_match_score(resume_skills, job['skills_list'])
            for job in jobs
        ]

        text_weight = 0.7
        skill_weight = 0.3
        final_scores = (text_weight * text_similarities) + (skill_weight * np.array(skill_scores))
        final_scores = final_scores.astype(float)

        # --- 5. Combine results ---
        results = []
        for i, job in enumerate(jobs):
            # Only include jobs where we successfully found a description
            if job['description']:
                results.append({
                    "title": job['title'],
                    "company": job['company'],
                    "score": float(final_scores[i]) * 100,
                    # This calculation will now work!
                    "matched_skills": list(set(resume_skills).intersection(set(job['skills_list']))),
                    "missing_skills": list(set(job['skills_list']).difference(set(resume_skills))),
                    # We send the first 500 chars of the desc to the frontend
                    "job_description": job['description'][:500] + "...", 
                    "link": job['link']
                })

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        # --- 6. AI summary via Gemini ---
        summary_prompt = f"Here is a resume:\n\n{resume_text}\n\nAct as a recruiter. Write a 3-sentence summary of this candidate."
        ai_summary = get_gemini_response(summary_prompt)

        if results:
            top_job = results[0]
            # Find the full description for the top job to send to Gemini
            full_desc_for_ai = ""
            for job in jobs:
                if job['link'] == top_job['link']:
                    full_desc_for_ai = job['description']
                    break
            
            match_prompt = f"Resume:\n{resume_text}\n\nJob Description:\n{full_desc_for_ai}\n\nExplain 3 key reasons why this candidate fits this job."
            ai_match_explanation = get_gemini_response(match_prompt)
        else:
            ai_match_explanation = ""

        return jsonify({
            "resume_skills": resume_skills,
            "jobs": results,
            "ai_summary": ai_summary,
            "ai_match_explanation": ai_match_explanation
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500






if __name__ == '__main__':
    app.run(debug=True)

