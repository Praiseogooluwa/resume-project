from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from mangum import Mangum
import requests
import os
import io
import fitz  # PyMuPDF
from typing import Dict, List
from pathlib import Path

app = FastAPI(
    title="Resume Matcher API",
    description="AI-powered job matching system",
    version="2.1",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json"
)

# Mount static directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-OpenAPI-Schema"],
    max_age=600
)

# Custom OpenAPI
@app.on_event("startup")
async def customize_openapi():
    if not app.openapi_schema:
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        for path in openapi_schema["paths"].values():
            for method in path.values():
                method["responses"].update({
                    "400": {"description": "Validation Error"},
                    "404": {"description": "Not Found"},
                    "500": {"description": "Internal Server Error"},
                    "502": {"description": "Service Unavailable"},
                    "504": {"description": "Gateway Timeout"}
                })
        app.openapi_schema = openapi_schema
        app.openapi = lambda: app.openapi_schema

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return RedirectResponse(url="/static/favicon.ico")

def extract_text_from_pdf(uploaded_file: UploadFile) -> str:
    try:
        pdf_data = uploaded_file.file.read(5_000_000)
        if uploaded_file.file.read(1):
            return "Error: PDF exceeds 5MB limit (max 5MB allowed)"

        pdf_stream = io.BytesIO(pdf_data)
        with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except fitz.FileDataError:
        return "Error: Invalid or corrupted PDF file"
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

@app.post("/match-jobs/", response_model=Dict[str, List[Dict]])
async def match_jobs(
    request: Request,
    file: UploadFile = File(..., description="PDF resume file (max 5MB)"),
    query: str = Form(..., min_length=2, description="Job search query")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Only PDF files are accepted")

    resume_text = extract_text_from_pdf(file)
    if not resume_text or "Error" in resume_text:
        raise HTTPException(400, detail=resume_text.replace("Error: ", ""))

    try:
        ml_url = os.getenv("RENDER_ML_URL")
        if not ml_url:
            raise RuntimeError("ML service URL not configured")

        response = requests.post(
            f"{ml_url.strip('/')}/predict",
            json={
                "resume_text": resume_text,
                "query": query.strip(),
                "client_ip": request.client.host
            },
            timeout=12,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Forwarded-For": request.client.host or ""
            }
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        raise HTTPException(504, detail="ML service timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(502, detail=f"ML service error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, detail=f"Processing error: {str(e)}")

@app.get("/get-jobs/", response_model=Dict[str, List[Dict]])
async def get_jobs(
    request: Request,
    query: str = Query(..., min_length=2, max_length=100, description="Job search keywords")
):
    try:
        api_key = os.getenv("JSEARCH_API_KEY")
        if not api_key:
            raise RuntimeError("API key not configured")

        response = requests.get(
            "https://jsearch.p.rapidapi.com/search",
            headers={
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
                "X-Forwarded-For": request.client.host or ""
            },
            params={
                "query": query.strip(),
                "num_pages": "1",
                "page": "1"
            },
            timeout=10
        )
        response.raise_for_status()

        jobs = response.json().get("data", [])
        return {
            "jobs": [{
                "title": job.get("job_title", "No title"),
                "company": job.get("employer_name", "Unknown company"),
                "location": ", ".join(filter(None, [job.get("job_city"), job.get("job_country")])),
                "description": (job.get("job_description") or "")[:300] + ("..." if len(job.get("job_description", "")) > 300 else ""),
                "apply_link": job.get("job_apply_link") or "#",
                "posted_at": job.get("job_posted_at_datetime_utc", "Unknown")
            } for job in jobs]
        }

    except requests.exceptions.Timeout:
        raise HTTPException(504, detail="JSearch API timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(502, detail=f"JSearch API error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, detail=f"Processing error: {str(e)}")

@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "healthy",
        "version": app.version,
        "services": {
            "ml_service": os.getenv("RENDER_ML_URL") is not None,
            "jsearch_api": os.getenv("JSEARCH_API_KEY") is not None
        }
    }

# Required for Vercel
handler = Mangum(app)
