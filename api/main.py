# The FastAPI entrypoint is now api/app.py. This file is deprecated.

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from pathlib import Path
import json
from main import FlickdAIEngine
from typing import Dict, Any
import uvicorn
from pydantic import BaseModel
from typing import List
import sys

sys.path.append("..")

app = FastAPI(
    title="Flickd AI Engine API",
    description="API for processing fashion videos and detecting items",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the engine
engine = FlickdAIEngine()

# Store processing status
processing_status: Dict[str, Any] = {}

class VideoAnalysisResult(BaseModel):
    video_id: str
    detected_products: List[int]
    vibe: str
    confidence_score: float

@app.post("/process-video")
async def process_video(
    video: UploadFile = File(...),
    caption: str = "",
    background_tasks: BackgroundTasks = None
):
    """Process a video and return results."""
    # Create temporary directory for video
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Save uploaded video
    video_path = temp_dir / video.filename
    try:
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving video: {str(e)}")
    
    try:
        # Process video
        result = engine.process_video(str(video_path), caption)
        
        # Clean up
        os.remove(video_path)
        
        return result
    except Exception as e:
        # Clean up on error
        if video_path.exists():
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/process-video-async")
async def process_video_async(
    video: UploadFile = File(...),
    caption: str = "",
    background_tasks: BackgroundTasks = None
):
    """Process a video asynchronously and return a job ID."""
    # Create temporary directory for video
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Save uploaded video
    video_path = temp_dir / video.filename
    try:
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving video: {str(e)}")
    
    # Generate job ID
    job_id = f"job_{len(processing_status)}"
    
    # Store initial status
    processing_status[job_id] = {
        "status": "processing",
        "progress": 0,
        "result": None,
        "error": None
    }
    
    def process_video_background():
        try:
            # Process video
            result = engine.process_video(str(video_path), caption)
            
            # Update status
            processing_status[job_id].update({
                "status": "completed",
                "progress": 100,
                "result": result
            })
        except Exception as e:
            # Update status with error
            processing_status[job_id].update({
                "status": "failed",
                "error": str(e)
            })
        finally:
            # Clean up
            if video_path.exists():
                os.remove(video_path)
    
    # Add background task
    background_tasks.add_task(process_video_background)
    
    return {"job_id": job_id, "status": "processing"}

@app.post("/analyze_video/")
async def analyze_video(video: UploadFile, caption: str):
    # Save uploaded video temporarily
    temp_video = Path("temp") / video.filename
    temp_video.parent.mkdir(exist_ok=True)
    
    with temp_video.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # Process video
    frames_dir = Path("frames") / video.filename.split(".")[0]
    frames_dir.mkdir(exist_ok=True)
    
    frame_paths = engine.extract_frames(str(temp_video), str(frames_dir))
    
    all_detections = []
    all_matches = []
    
    # Process each frame
    for frame_path in frame_paths:
        detections = engine.detect_fashion_items(str(frame_path))
        matches = engine.match_products(str(frame_path), detections)
        all_matches.extend(matches)
    
    # Get unique product matches
    unique_products = list(set(all_matches))
    
    # Classify vibe
    vibe = engine.classify_vibe(caption)
    
    # Clean up
    temp_video.unlink()
    
    return VideoAnalysisResult(
        video_id=video.filename,
        detected_products=unique_products,
        vibe=vibe,
        confidence_score=0.85  # Example confidence score
    )

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of an asynchronous processing job."""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_status[job_id]

@app.get("/health")
async def health_check():
    """Check if the API is healthy."""
    return {"status": "healthy"}

@app.get("/vibe-keywords")
async def get_vibe_keywords():
    """Get the list of vibe keywords used for classification."""
    return engine.vibe_keywords

@app.get("/")
def read_root():
    return {"message": "Flickd AI Engine API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)