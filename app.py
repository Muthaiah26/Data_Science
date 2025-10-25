import fitz 
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile
import traceback
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time 


API_KEY = "AIzaSyCYDkTZgh13GV7d1-QR8Bq3YbjNNvcllmY"
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
        if 'resume' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['resume']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # --- 1. Extract Resume Text ---
        resume_text = ""
        file_name = file.filename.lower()
        
        if file_name.endswith('.pdf'):
            resume_text = extract_text_from_pdf(file)
        elif file_name.endswith('.txt'):
            resume_text = extract_text_from_txt(file)
        else:
            return jsonify({"error": "Unsupported file type. Please upload a .pdf or .txt file."}), 400

        if not resume_text:
            return jsonify({"error": "Could not extract text from resume."}), 500

       
        resume_skills = extract_skills(resume_text, SKILLS_DB)
        resume_clean = clean_text(resume_text) # Clean text for embedding
        resume_vec = model.encode([resume_clean])
        
        if jobs_df.empty or job_embeddings is None:
            return jsonify({"error": "Job data is not loaded on the server."}), 500


        text_similarities = cosine_similarity(resume_vec, job_embeddings)[0]
        
       
        skill_scores = [calculate_skill_match_score(resume_skills, job_skills) for job_skills in jobs_df['skills_list']]
        
        text_weight = 0.7
        skill_weight = 0.3
        
        final_scores = (text_weight * text_similarities) + (skill_weight * np.array(skill_scores))
        
        jobs_df['match_score'] = final_scores
        
        # --- 4. Get Top Matches ---
        top_n = 10
        top_matches_df = jobs_df.sort_values(by='match_score', ascending=False).head(top_n)

        results = []
        for _, row in top_matches_df.iterrows():
            # Get skills from the pre-processed list
            job_skill_list = row['skills_list'] 
            results.append({
                "title": row['title'],
                "company": row['company'],
                "score": row['match_score'] * 100,
                "matched_skills": list(set(resume_skills).intersection(set(job_skill_list))),
                "missing_skills": list(set(job_skill_list).difference(set(resume_skills))),
                "job_description": row['description'] 
            })
            
        # --- 5. Call Gemini API for AI-powered insights ---
        
        summary_prompt = f"Here is a resume:\n\n{resume_text}\n\nAct as a professional recruiter. Write a concise 3-sentence summary of this person's key strengths and professional profile."
        ai_summary = get_gemini_response(summary_prompt)
        
        ai_match_explanation = ""
        if results:
            top_job = results[0]
            match_prompt = f"Resume:\n{resume_text}\n\nJob Description:\n{top_job['job_description']}\n\nAct as a hiring manager. In 3 bullet points, explain the *top 3 reasons* why the person with this resume is a strong fit for this specific job. Focus on matching skills and experience."
            ai_match_explanation = get_gemini_response(match_prompt)

        return jsonify({
            "resume_skills": resume_skills,
            "jobs": results,
            "ai_summary": ai_summary,
            "ai_match_explanation": ai_match_explanation
        })

    except Exception as e:
        print(f"---!! UNEXPECTED ERROR IN /analyze !!---")
        print(traceback.format_exc())
        return jsonify({"error": "An unexpected server error occurred. Please check the server logs."}), 500


if __name__ == '__main__':
    app.run(debug=True)

