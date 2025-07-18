import os
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache
import torch

# Optimized model loading with caching
@lru_cache(maxsize=1)
def load_model():
    # Force CPU and reduce threads for Render's free tier
    torch.set_num_threads(1)
    return SentenceTransformer(
        'all-MiniLM-L6-v2',
        device='cpu',
        cache_folder='/tmp/models'  # Prevent re-downloads
    )

model = load_model()

def get_top_matches(resume_text, query, top_k=3):
    """Optimized for Render's environment"""
    try:
        # Free up memory before processing
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Process in inference mode for better performance
        with torch.inference_mode():
            # Encode with smaller batch size
            resume_embedding = model.encode(
                resume_text,
                convert_to_tensor=True,
                batch_size=8,
                show_progress_bar=False
            )
            
            jobs = fetch_jobs_from_api(query)
            if not jobs:
                return []

            job_texts = [j.get("job_description", "") for j in jobs]
            job_embeddings = model.encode(
                job_texts,
                convert_to_tensor=True,
                batch_size=8,
                show_progress_bar=False
            )

            # Calculate similarities
            similarities = util.pytorch_cos_sim(resume_embedding, job_embeddings)[0]
            top_indices = similarities.argsort(descending=True)[:top_k]

            return [format_job(jobs[i], similarities[i]) for i in top_indices]

    except Exception as e:
        print(f"⚠️ ML processing error: {str(e)}")
        return []

def format_job(job, similarity_score):
    """Consolidated job formatting"""
    desc = job.get("job_description", "")
    return {
        "title": job.get("job_title", "No title"),
        "company": job.get("employer_name", "Unknown"),
        "location": f"{job.get('job_city', '')}, {job.get('job_country', '')}".strip(", "),
        "description": desc[:300] + ("..." if len(desc) > 300 else ""),
        "score": round(float(similarity_score) * 100, 2),
        "apply_link": job.get("job_apply_link") or "No link",
        "posted_at": job.get("job_posted_at_datetime_utc", "Unknown")
    }

def fetch_jobs_from_api(query, num_results=10):
    """Added retries and timeout handling"""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))

    try:
        response = session.get(
            "https://jsearch.p.rapidapi.com/search",
            headers={
                "X-RapidAPI-Key": os.getenv("JSEARCH_API_KEY"),
                "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
            },
            params={"query": query, "num_pages": 1},
            timeout=15
        )
        response.raise_for_status()
        return response.json().get("data", [])[:num_results]
        
    except Exception as e:
        print(f"⚠️ JSearch API error: {str(e)}")
        return []