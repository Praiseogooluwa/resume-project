from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
import requests
import os
import io
import fitz  # PyMuPDF
import asyncio
import httpx
import uuid
from typing import Dict, List
from datetime import datetime, timedelta

# Create FastAPI app
app = FastAPI(
    title="Resume Matcher API",
    description="AI-powered job matching system",
    version="2.2"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (use Redis/DB for production)
job_status: Dict[str, dict] = {}

# Cleanup old jobs periodically
def cleanup_old_jobs():
    cutoff_time = datetime.now() - timedelta(hours=1)
    expired_jobs = [
        job_id for job_id, job_data in job_status.items()
        if datetime.fromisoformat(job_data.get('created_at', '2000-01-01T00:00:00')) < cutoff_time
    ]
    for job_id in expired_jobs:
        del job_status[job_id]

@app.get("/")
async def root():
    return {
        "message": "Resume Matcher API is running",
        "version": "2.2",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "match_jobs": "/match-jobs/",
            "match_jobs_async": "/match-jobs-async/",
            "job_status": "/job-status/{job_id}",
            "get_jobs": "/get-jobs/"
        }
    }

def extract_text_from_pdf(uploaded_file: UploadFile) -> str:
    try:
        uploaded_file.file.seek(0)
        pdf_data = uploaded_file.file.read()
        
        if len(pdf_data) > 5_000_000:
            return "Error: PDF exceeds 5MB limit"

        pdf_stream = io.BytesIO(pdf_data)
        with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

async def call_ml_service_async(resume_text: str, query: str) -> dict:
    """Async call to ML service with longer timeout"""
    ml_url = os.getenv("RENDER_ML_URL")
    if not ml_url:
        raise HTTPException(status_code=500, detail="ML service URL not configured")

    async with httpx.AsyncClient(timeout=120.0) as client:  # 2 minutes timeout
        try:
            response = await client.post(
                f"{ml_url.rstrip('/')}/predict",
                json={
                    "resume_text": resume_text,
                    "query": query.strip()
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="ML service timeout")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"ML service error: {str(e)}")

def process_resume_job_sync(job_id: str, resume_text: str, query: str):
    """Background task to process resume matching"""
    try:
        # Update status to processing
        job_status[job_id]["status"] = "processing"
        job_status[job_id]["progress"] = "Calling ML service..."
        
        # Make synchronous call to ML service with longer timeout
        ml_url = os.getenv("RENDER_ML_URL")
        if not ml_url:
            job_status[job_id] = {
                "status": "error",
                "error": "ML service URL not configured",
                "created_at": job_status[job_id]["created_at"]
            }
            return

        response = requests.post(
            f"{ml_url.rstrip('/')}/predict",
            json={
                "resume_text": resume_text,
                "query": query.strip()
            },
            timeout=300,  # 5 minutes timeout for background task
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        # Update with success
        job_status[job_id].update({
            "status": "completed",
            "result": result,
            "completed_at": datetime.now().isoformat()
        })
        
    except requests.exceptions.Timeout:
        job_status[job_id].update({
            "status": "error",
            "error": "ML service timeout after 5 minutes"
        })
    except requests.exceptions.RequestException as e:
        job_status[job_id].update({
            "status": "error",
            "error": f"ML service error: {str(e)}"
        })
    except Exception as e:
        job_status[job_id].update({
            "status": "error",
            "error": f"Processing error: {str(e)}"
        })

@app.post("/match-jobs-async/")
async def match_jobs_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF resume file"),
    query: str = Form(..., description="Job search query")
):
    """Match resume with jobs using async processing"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    resume_text = extract_text_from_pdf(file)
    if "Error" in resume_text:
        raise HTTPException(status_code=400, detail=resume_text.replace("Error: ", ""))

    # Create job ID and initialize status
    job_id = str(uuid.uuid4())
    job_status[job_id] = {
        "status": "queued",
        "progress": "Initializing...",
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None
    }
    
    # Add background task
    background_tasks.add_task(process_resume_job_sync, job_id, resume_text, query)
    
    # Cleanup old jobs
    cleanup_old_jobs()
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Job submitted successfully. Use /job-status/{job_id} to check progress.",
        "estimated_time": "2-5 minutes"
    }

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of an async job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status[job_id]

@app.post("/match-jobs/")
async def match_jobs(
    file: UploadFile = File(..., description="PDF resume file"),
    query: str = Form(..., description="Job search query")
):
    """Match resume with jobs using ML service (quick timeout)"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    resume_text = extract_text_from_pdf(file)
    if "Error" in resume_text:
        raise HTTPException(status_code=400, detail=resume_text.replace("Error: ", ""))

    try:
        # Try with shorter timeout first
        result = await asyncio.wait_for(
            call_ml_service_async(resume_text, query),
            timeout=20.0  # 20 seconds for quick response
        )
        return result
        
    except asyncio.TimeoutError:
        # If timeout, suggest async endpoint
        raise HTTPException(
            status_code=504, 
            detail="Request timeout. For longer processing, use /match-jobs-async/ endpoint"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-jobs/")
async def get_jobs(
    query: str = Query(..., min_length=2, max_length=100, description="Job search keywords")
):
    """Get jobs from JSearch API"""
    try:
        api_key = os.getenv("JSEARCH_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="JSearch API key not configured")

        async with httpx.AsyncClient(timeout=25.0) as client:
            response = await client.get(
                "https://jsearch.p.rapidapi.com/search",
                headers={
                    "X-RapidAPI-Key": api_key,
                    "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
                },
                params={
                    "query": query.strip(),
                    "num_pages": "1",
                    "page": "1"
                }
            )
            response.raise_for_status()

        jobs = response.json().get("data", [])
        return {
            "jobs": [{
                "title": job.get("job_title", "No title"),
                "company": job.get("employer_name", "Unknown company"),
                "location": f"{job.get('job_city', '')}, {job.get('job_country', '')}".strip(', '),
                "description": (job.get("job_description") or "")[:300] + ("..." if len(job.get("job_description", "")) > 300 else ""),
                "apply_link": job.get("job_apply_link") or "#",
                "posted_at": job.get("job_posted_at_datetime_utc", "Unknown")
            } for job in jobs]
        }

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="JSearch API timeout")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"JSearch API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.2",
        "active_jobs": len(job_status),
        "services": {
            "ml_service": bool(os.getenv("RENDER_ML_URL")),
            "jsearch_api": bool(os.getenv("JSEARCH_API_KEY"))
        },
        "environment_variables": {
            "RENDER_ML_URL": "configured" if os.getenv("RENDER_ML_URL") else "missing",
            "JSEARCH_API_KEY": "configured" if os.getenv("JSEARCH_API_KEY") else "missing"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    print("Resume Matcher API starting up...")
    # Pre-warm any models or services here if needed

# Vercel handler
handler = Mangum(app)
