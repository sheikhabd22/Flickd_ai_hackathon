import os
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """Download and save required models."""
    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True)
    
    # Download YOLOv8
    logger.info("Downloading YOLOv8 model...")
    yolo_model = YOLO('yolov8n.pt')
    yolo_model.save(str(models_dir / "yolov8n.pt"))
    
    # Download CLIP
    logger.info("Downloading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Save CLIP model
    clip_model.save_pretrained(str(models_dir / "clip"))
    clip_processor.save_pretrained(str(models_dir / "clip"))
    
    logger.info("All models downloaded successfully!")

if __name__ == "__main__":
    download_models() 