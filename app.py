import fitz  # PyMuPDF
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile # <-- Import tempfile

# Use standard Flask template folder
app = Flask(__name__, template_folder='templates')

# --- CONFIGURATION & MODEL LOADING ---
SKILLS_DB = [
    'python', 'java', 'c++', 'javascript', 'react', 'angular', 'vue',
    'node.js', 'sql', 'mysql', 'postgresql', 'mongodb', 'docker',
    'kubernetes', 'aws', 'azure', 'gcp', 'terraform', 'ansible',
    'machine learning', 'deep learning', 'tensorflow', 'pytorch',
    'scikit-learn', 'pandas', 'numpy', 'data analysis',
    'project management', 'agile', 'scrum', 'product management',
    'ui/ux design', 'figma', 'sketch', 'adobe xd', 'git',
    'jira', 'communication', 'teamwork', 'leadership'
]

print("Loading sentence embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# (load_and_preprocess_jobs, clean_text, extract_skills, calculate_skill_match_score remain the same)
def load_and_preprocess_jobs(csv_path='jobs.csv'):
    """
    Loads job data from CSV, cleans it, extracts skills,
    and pre-computes text embeddings.
    """
    try:
        jobs_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: '{csv_path}' not found. Please create it.")
        return pd.DataFrame(), None, {}

    if 'description' not in jobs_df.columns:
        print("Error: 'description' column not in jobs.csv")
        return pd.DataFrame(), None, {}
        
    print(f"Loading and preprocessing {len(jobs_df)} jobs from '{csv_path}'...")
    
    jobs_df['description'] = jobs_df['description'].astype(str)
    jobs_df['description_clean'] = jobs_df['description'].apply(clean_text)
    jobs_df['skills'] = jobs_df['description_clean'].apply(lambda x: extract_skills(x, SKILLS_DB))
    job_embeddings = model.encode(jobs_df['description_clean'].tolist(), show_progress_bar=True)
    
    print("Job processing complete.")
    return jobs_df, job_embeddings

def clean_text(text):
    """A simple text cleaning function."""
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

def extract_skills(text, skills_list):
    """Extracts a set of skills from text based on a skills list."""
    found_skills = set()
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            found_skills.add(skill)
    return list(found_skills)

# ---!!! NEW, MORE ROBUST PDF EXTRACTION ---
def extract_text_from_pdf(file_storage):
    """
    Saves the file storage object to a temp file and reads it.
    This is more robust than reading from a stream.
    """
    temp_file_path = None
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            file_storage.save(temp_file) # Save the uploaded file to the temp file
            temp_file_path = temp_file.name

        # Now, open the temp file with fitz
        with fitz.open(temp_file_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def extract_text_from_txt(file_storage):
    """Extracts text from a TXT file stream."""
    try:
        return file_storage.read().decode('utf-8')
    except UnicodeDecodeError:
        try:
            # Fallback for different encoding
            file_storage.seek(0) # Reset stream pointer
            return file_storage.read().decode('latin-1')
        except Exception as e:
            print(f"Error reading TXT: {e}")
            return None

def calculate_skill_match_score(resume_skills, job_skills):
    """
    Calculates a Jaccard similarity score between two lists of skills.
    """
    if not resume_skills and not job_skills:
        return 1.0  # Both empty, perfect match
    if not resume_skills or not job_skills:
        return 0.0  # One is empty, no match

    resume_set = set(resume_skills)
    job_set = set(job_skills)
    
    intersection = len(resume_set.intersection(job_set))
    union = len(resume_set.union(job_set))
    
    return intersection / union if union > 0 else 0.0

# --- Load data on app start ---
jobs_df, job_embeddings = load_and_preprocess_jobs()

# --- FLASK ROUTES ---

@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyzes the uploaded resume (PDF or TXT) against the pre-processed jobs.
    """
    if 'resume' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['resume']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # --- 1. Extract Resume Text ---
    resume_text = ""
    file_name = file.filename.lower()
    
    if file_name.endswith('.pdf'):
        # ---!!! USE THE NEW ROBUST FUNCTION ---
        resume_text = extract_text_from_pdf(file)
        if resume_text is None:
            return jsonify({"error": "Could not read PDF file. File may be corrupt or unreadable."}), 500
    elif file_name.endswith('.txt'):
        resume_text = extract_text_from_txt(file)
        if resume_text is None:
            return jsonify({"error": "Could not read text file."}), 500
    else:
        return jsonify({"error": "Unsupported file type. Please upload a .pdf or .txt file."}), 400

    if not resume_text:
        return jsonify({"error": "Could not extract text from resume. The file might be empty."}), 500

    # --- 2. Process Resume ---
    resume_clean = clean_text(resume_text)
    resume_skills = extract_skills(resume_clean, SKILLS_DB)
    resume_vec = model.encode([resume_clean]) # Shape (1, embedding_dim)
    
    if jobs_df.empty or job_embeddings is None:
        return jsonify({"error": "Job data is not loaded on the server."}), 500

    # --- 3. Calculate Scores ---
    
    text_similarities = cosine_similarity(resume_vec, job_embeddings)[0]
    skill_scores = [calculate_skill_match_score(resume_skills, job_skills) for job_skills in jobs_df['skills']]
    
    text_weight = 0.7
    skill_weight = 0.3
    
    final_scores = (text_weight * text_similarities) + (skill_weight * np.array(skill_scores))
    
    jobs_df['match_score'] = final_scores
    
    # --- 4. Get Top Matches ---
    top_n = 10
    top_matches_df = jobs_df.sort_values(by='match_score', ascending=False).head(top_n)

    results = []
    for _, row in top_matches_df.iterrows():
        results.append({
            "title": row['title'],
            "company": row['company'],
            "score": row['match_score'] * 100, # Convert to percentage
            "matched_skills": list(set(resume_skills).intersection(set(row['skills']))),
            "missing_skills": list(set(row['skills']).difference(set(resume_skills)))
        })

    # Return the resume's extracted skills and the top job matches
    return jsonify({
        "resume_skills": resume_skills,
        "jobs": results
    })

if __name__ == '__main__':
    app.run(debug=True)

