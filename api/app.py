from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from pathlib import Path
import logging
from typing import Dict, Any, List
import json
import sys

# Ensure the parent directory is in sys.path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from main import FlickdAIEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Flickd AI Engine API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance and job status
engine = None
processing_status: Dict[str, Any] = {}

class CaptionRequest(BaseModel):
    caption: str

@app.on_event("startup")
async def startup_event():
    """Initialize the AI engine and set up directories on startup."""
    global engine
    # Ensure required directories exist
    for d in ["frames", "images", "data", "embeddings", "outputs", "models", "temp"]:
        Path(d).mkdir(exist_ok=True)
    try:
        logger.info("Initializing FlickdAIEngine...")
        engine = FlickdAIEngine()
        logger.info("Building product index...")
        engine.build_product_index("data/products.csv", "images")
        logger.info("API startup complete")
    except Exception as e:
        logger.error(f"Failed to initialize engine: {str(e)}")
        raise

@app.post("/process-video")
async def process_video(
    video: UploadFile = File(...),
    caption: str = ""
):
    """Process a video and return structured output (synchronous)."""
    try:
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        video_path = temp_dir / video.filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        result = engine.process_video(str(video_path), caption or "")
        os.remove(video_path)
        return result
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-video-async")
async def process_video_async(
    video: UploadFile = File(...),
    caption: str = "",
    background_tasks: BackgroundTasks = None
):
    """Process a video asynchronously and return a job ID."""
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    video_path = temp_dir / video.filename
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving video: {str(e)}")
    job_id = f"job_{len(processing_status)}"
    processing_status[job_id] = {
        "status": "processing",
        "progress": 0,
        "result": None,
        "error": None
    }
    def process_video_background():
        try:
            result = engine.process_video(str(video_path), caption)
            processing_status[job_id].update({
                "status": "completed",
                "progress": 100,
                "result": result
            })
        except Exception as e:
            processing_status[job_id].update({
                "status": "failed",
                "error": str(e)
            })
        finally:
            if video_path.exists():
                os.remove(video_path)
    if background_tasks:
        background_tasks.add_task(process_video_background)
    else:
        process_video_background()
    return {"job_id": job_id, "status": "processing"}

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of an asynchronous processing job."""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return processing_status[job_id]

@app.post("/classify-vibe")
async def classify_vibe(request: CaptionRequest):
    """Classify the vibe of a caption."""
    try:
        vibes = engine.classify_vibe(request.caption)
        return {"vibes": vibes}
    except Exception as e:
        logger.error(f"Error classifying vibe: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vibe-keywords")
async def get_vibe_keywords():
    """Get the list of vibe keywords used for classification."""
    return engine.vibe_keywords if engine else {}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "engine_initialized": engine is not None}

@app.get("/")
def read_root():
    return {"message": "Flickd AI Engine API"} 