
from apscheduler.schedulers.background import BackgroundScheduler
from fetch_jobs import fetch_adzuna, ingest_jobs_to_mongo
from indexing import build_faiss_index

def job_fetcher():
    jobs = fetch_adzuna("data scientist", "India", 50)
    ingest_jobs_to_mongo(jobs)
    build_faiss_index()

scheduler = BackgroundScheduler()
scheduler.add_job(job_fetcher, 'interval', minutes=30)
scheduler.start()
