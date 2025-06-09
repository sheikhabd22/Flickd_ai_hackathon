import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import faiss
from PIL import Image
from typing import List, Dict, Any
import logging
from utils.embedding_manager import EmbeddingManager
from utils.clip_matcher import ClipMatcher
from utils.audio_processor import AudioProcessor
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Flickd AI Engine API")

# Initialize the engine globally
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    try:
        engine = FlickdAIEngine()
        # Build product index
        logger.info("Building product index...")
        products_csv = "data/products.csv"
        images_dir = "images"
        
        if not os.path.exists(products_csv):
            raise FileNotFoundError(f"Products CSV not found at {products_csv}")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found at {images_dir}")
            
        # Verify we have matching images for products
        products_df = pd.read_csv(products_csv)
        missing_images = []
        for product_id in products_df['id']:
            img_path = os.path.join(images_dir, f"{product_id}.jpg")
            if not os.path.exists(img_path):
                missing_images.append(product_id)
        
        if missing_images:
            logger.warning(f"Missing images for {len(missing_images)} products")
            products_df = products_df[~products_df['id'].isin(missing_images)]
            products_df.to_csv(products_csv, index=False)
            logger.info(f"Updated products.csv with {len(products_df)} products that have images")
        
        engine.build_product_index(products_csv, images_dir)
        logger.info("Product index built successfully")
    except Exception as e:
        logger.error(f"Failed to initialize engine: {str(e)}")
        raise

@app.post("/process-video")
async def process_video(
    video: UploadFile = File(...),
    caption: str = None
):
    try:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            shutil.copyfileobj(video.file, temp_file)
            temp_path = temp_file.name

        # Process the video
        result = engine.process_video(temp_path, caption or "")
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

class FlickdAIEngine:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize models with caching
        logger.info("Initializing YOLOv8 model...")
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            # Download model if not present
            if not os.path.exists('yolov8n.pt'):
                logger.info("Downloading YOLOv8 model...")
                self.yolo_model = YOLO('yolov8n.pt')
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8: {str(e)}")
            raise
        
        logger.info("Initializing audio processor...")
        self.audio_processor = AudioProcessor()
        
        logger.info("Initializing Whisper model for audio transcription...")
        self.whisper_model = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        logger.info("Initializing CLIP matcher...")
        self.clip_matcher = ClipMatcher()
        
        logger.info("Initializing BART model for vibe classification...")
        self.vibe_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            cache_dir=self.model_dir / "bart"
        )
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager()
        
        # Initialize FAISS index
        self.product_index = None
        self.product_metadata = None
        
        # Create outputs directory
        os.makedirs("outputs", exist_ok=True)
        
        # Define fashion-related classes for YOLO with confidence thresholds
        self.fashion_classes = {
            'person': 0.3,
            'backpack': 0.4,
            'handbag': 0.4,
            'suitcase': 0.4,
            'umbrella': 0.4,
            'shoe': 0.4,
            'boot': 0.4,
            'hat': 0.4,
            'helmet': 0.4,
            'glasses': 0.4,
            'sunglasses': 0.4,
            'tie': 0.4,
            'scarf': 0.4,
            'glove': 0.4,
            'dress': 0.35,
            'shirt': 0.35,
            't-shirt': 0.35,
            'pants': 0.35,
            'shorts': 0.35,
            'skirt': 0.35,
            'jacket': 0.35,
            'coat': 0.35,
            'sweater': 0.35,
            'hoodie': 0.35,
            'socks': 0.4,
            'belt': 0.4,
            'watch': 0.4,
            'necklace': 0.4,
            'bracelet': 0.4,
            'ring': 0.4,
            'earring': 0.4
        }
        
        # Define vibe keywords and hashtags (expanded)
        self.vibe_keywords = {
            "Coquette": {
                "keywords": ["coquette", "feminine", "blush", "romantic", "delicate", "soft", "pastel", "pink", "lace", "ribbon", "bow", "frilly", "sweet"],
                "hashtags": ["#coquette", "#coquetteaesthetic", "#coquettestyle", "#coquettefashion", "#coquetteoutfit"]
            },
            "Clean Girl": {
                "keywords": ["clean", "minimal", "glow", "simple", "elegant", "natural", "fresh", "neutral", "beige", "white", "cream", "minimalist"],
                "hashtags": ["#cleangirl", "#cleangirlaesthetic", "#minimalstyle", "#minimalistfashion", "#neutralaesthetic"]
            },
            "Cottagecore": {
                "keywords": ["cottagecore", "floral", "vintage", "rustic", "nature", "garden", "whimsical", "mushroom", "fairy", "forest", "cottage", "vintage"],
                "hashtags": ["#cottagecore", "#cottagecoreaesthetic", "#cottagecorestyle", "#cottagecorefashion", "#cottagecoreoutfit"]
            },
            "Streetcore": {
                "keywords": ["street", "baggy", "urban", "edgy", "casual", "streetwear", "urban", "oversized", "grunge", "punk", "skate", "hiphop"],
                "hashtags": ["#streetcore", "#streetwear", "#urbanstyle", "#streetfashion", "#streetwearstyle"]
            },
            "Y2K": {
                "keywords": ["y2k", "retro", "low-rise", "nostalgic", "90s", "2000s", "vintage", "butterfly", "sparkle", "glitter", "juicy", "britney"],
                "hashtags": ["#y2k", "#y2kaesthetic", "#y2kstyle", "#y2kfashion", "#y2koutfit"]
            },
            "Boho": {
                "keywords": ["boho", "earthy", "flowy", "bohemian", "hippie", "free-spirited", "natural", "fringe", "tassel", "macrame", "tribal", "ethnic"],
                "hashtags": ["#boho", "#bohostyle", "#bohemian", "#bohofashion", "#bohochic"]
            },
            "Party Glam": {
                "keywords": ["party", "glam", "sparkle", "glitter", "dressy", "elegant", "festive", "sequin", "metallic", "shiny", "glittery", "fancy"],
                "hashtags": ["#partystyle", "#glam", "#partyglam", "#partyfashion", "#glamstyle"]
            }
        }
        
        # Define candidate labels for zero-shot classification
        self.vibe_candidates = list(self.vibe_keywords.keys())
        
        # Ensure product_metadata is always a DataFrame after loading
        index, metadata = self.embedding_manager.load_index()
        if index is not None and metadata is not None:
            self.product_index = index
            if isinstance(metadata, dict):
                self.product_metadata = pd.DataFrame(metadata)
            else:
                self.product_metadata = metadata
    
    def extract_frames(self, video_path: str, output_dir: str, frame_interval: int = 30) -> List[str]:
        """Extract frames from video at specified intervals."""
        logger.info(f"Extracting frames from {video_path}")
        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
            frame_count += 1
            
        cap.release()
        logger.info(f"Extracted {len(frame_paths)} frames")
        return frame_paths
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video and transcribe it using Whisper."""
        logger.info(f"Extracting and transcribing audio from {video_path}")
        try:
            # Extract audio using ffmpeg
            audio_path = os.path.join("temp", f"{Path(video_path).stem}.wav")
            os.makedirs("temp", exist_ok=True)
            
            # Extract audio using ffmpeg
            os.system(f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}" -y')
            
            # Transcribe audio using Whisper
            result = self.whisper_model(audio_path)
            transcription = result["text"]
            
            # Clean up temporary audio file
            os.remove(audio_path)
            
            return transcription
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {str(e)}")
            return ""
    
    def detect_fashion_items(self, frame_path: str) -> List[Dict[str, Any]]:
        """Detect fashion items in a frame using YOLOv8 with improved detection."""
        logger.info(f"Detecting fashion items in {frame_path}")
        
        # Run YOLO detection with lower confidence threshold
        results = self.yolo_model(frame_path, conf=0.25)  # Lower threshold for more detections
        
        detections = []
        detected_classes = set()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_name = result.names[int(box.cls)]
                confidence = float(box.conf)
                
                # Only process fashion-related classes with class-specific thresholds
                if class_name.lower() in self.fashion_classes:
                    threshold = self.fashion_classes[class_name.lower()]
                    if confidence > threshold:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detected_classes.add(class_name)
                        
                        # Add padding to the bounding box
                        padding = 0.1  # 10% padding
                        width = x2 - x1
                        height = y2 - y1
                        
                        x1 = max(0, x1 - width * padding)
                        y1 = max(0, y1 - height * padding)
                        x2 = min(result.orig_shape[1], x2 + width * padding)
                        y2 = min(result.orig_shape[0], y2 + height * padding)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class': class_name
                        })
                        
                        # If we detect a person, also try to detect their clothing
                        if class_name.lower() == 'person':
                            # Add additional detections for upper body and lower body
                            person_height = y2 - y1
                            # Upper body (shirt, jacket, etc.)
                            upper_body = {
                                'bbox': [x1, y1, x2, y1 + person_height * 0.6],
                                'confidence': confidence,
                                'class': 'upper_body'
                            }
                            # Lower body (pants, skirt, etc.)
                            lower_body = {
                                'bbox': [x1, y1 + person_height * 0.4, x2, y2],
                                'confidence': confidence,
                                'class': 'lower_body'
                            }
                            detections.extend([upper_body, lower_body])
        
        logger.info(f"Detected {len(detections)} fashion items. Classes: {list(detected_classes)}")
        if not detections:
            logger.warning("No fashion items detected in frame")
        return detections
    
    def crop_detected_items(self, frame_path: str, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Crop detected items from the frame with padding."""
        logger.info(f"Cropping detected items from {frame_path}")
        frame = Image.open(frame_path)
        cropped_items = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Add padding to the crop
            padding = 0.1  # 10% padding
            width = x2 - x1
            height = y2 - y1
            
            x1 = max(0, x1 - width * padding)
            y1 = max(0, y1 - height * padding)
            x2 = min(frame.width, x2 + width * padding)
            y2 = min(frame.height, y2 + height * padding)
            
            try:
                cropped = frame.crop((x1, y1, x2, y2))
                cropped_items.append({
                    'detection': detection,
                    'cropped_image': cropped
                })
            except Exception as e:
                logger.warning(f"Failed to crop item: {str(e)}")
                continue
        
        return cropped_items
    
    def generate_clip_embeddings(self, cropped_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate CLIP embeddings for cropped items."""
        return self.clip_matcher.generate_clip_embeddings(cropped_items)
    
    def build_product_index(self, products_csv: str, images_dir: str):
        """Build or load FAISS index for product images (cosine similarity)."""
        self.clip_matcher.build_product_index(products_csv, images_dir)
    
    def match_products(self, items_with_embeddings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Match detected items to product catalog using CLIP + FAISS with improved matching."""
        logger.info("Matching products")
        matches = []
        
        for item in items_with_embeddings:
            try:
                query_embedding = {
                    'image_embedding': item['image_embedding'],
                    'clip_text_embedding': item['clip_text_embedding']  # Fixed key name
                }
                
                # Get more matches with lower threshold
                product_matches = self.clip_matcher.match_embedding(
                    query_embedding,
                    top_k=10,  # Increased from 5 to 10
                    similarity_threshold=0.4  # Lowered from 0.6 to 0.4
                )
                
                if product_matches:
                    matches.extend(product_matches)
            except Exception as e:
                logger.warning(f"Failed to match item: {str(e)}")
                continue
        
        # Sort matches by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Remove duplicates while preserving order
        seen_ids = set()
        unique_matches = []
        for match in matches:
            if match['matched_product_id'] not in seen_ids:
                unique_matches.append(match)
                seen_ids.add(match['matched_product_id'])
        
        # Take top matches
        return unique_matches[:8]  # Increased from 4 to 8 matches
    
    def classify_vibe(self, caption: str) -> List[str]:
        """Classify the vibe of the video using hybrid approach (rule-based + zero-shot)."""
        logger.info("Classifying video vibe")
        caption = caption.lower()
        vibe_scores = {}
        
        # 1. Rule-based keyword matching
        for vibe, data in self.vibe_keywords.items():
            keyword_matches = sum(1 for kw in data['keywords'] if kw in caption)
            hashtag_matches = sum(1 for ht in data['hashtags'] if ht.lower() in caption)
            if keyword_matches > 0 or hashtag_matches > 0:
                rule_score = (keyword_matches + hashtag_matches) / (len(data['keywords']) + len(data['hashtags']))
                vibe_scores[vibe] = rule_score
        
        # 2. Zero-shot classification using BART with smart prompts
        if caption.strip():
            # Create context-aware prompts
            prompts = [
                f"The vibe of this outfit is: {caption}",
                f"This fashion style represents: {caption}",
                f"The aesthetic of this look is: {caption}"
            ]
            
            # Get predictions for each prompt
            prompt_scores = {}
            for prompt in prompts:
                zero_shot_results = self.vibe_classifier(
                    prompt,
                    candidate_labels=self.vibe_candidates,
                    multi_label=True
                )
                
                # Combine scores from different prompts
                for vibe, score in zip(zero_shot_results['labels'], zero_shot_results['scores']):
                    if vibe in prompt_scores:
                        prompt_scores[vibe].append(score)
                    else:
                        prompt_scores[vibe] = [score]
            
            # Average scores across prompts
            for vibe, scores in prompt_scores.items():
                zero_shot_score = sum(scores) / len(scores)
                if vibe in vibe_scores:
                    # Combine rule-based and zero-shot scores with equal weight
                    vibe_scores[vibe] = (vibe_scores[vibe] + zero_shot_score) / 2
                else:
                    vibe_scores[vibe] = zero_shot_score
        
        # Sort vibes by score and take top 3
        sorted_vibes = sorted(vibe_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Ensure we return at least one vibe
        if not sorted_vibes:
            return ["Clean Girl"]
        
        # Return vibes with scores above threshold
        return [vibe for vibe, score in sorted_vibes if score > 0.3]
    
    def validate_input(self, video_path: str, caption: str) -> None:
        """Validate input video and caption."""
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        if not video_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            raise ValueError("Unsupported video format. Supported formats: .mp4, .mov, .avi, .mkv")
        
        if not caption or not isinstance(caption, str):
            raise ValueError("Invalid caption. Must be a non-empty string")

    def process_video(self, video_path: str, caption: str) -> Dict[str, Any]:
        """Process a video and return structured output with enhanced matching."""
        try:
            # Validate input
            self.validate_input(video_path, caption)
            
            logger.info(f"Processing video: {video_path}")
            
            # Process audio
            audio_result = self.audio_processor.process_video_audio(video_path)
            audio_transcription = audio_result['text']
            logger.info(f"Audio transcription: {audio_transcription}")
            
            # Combine caption with audio transcription for better context
            combined_text = f"{caption} {audio_transcription}"
            
            # Extract frames
            frames_dir = os.path.join("frames", Path(video_path).stem)
            os.makedirs(frames_dir, exist_ok=True)
            frame_paths = self.extract_frames(video_path, frames_dir, frame_interval=15)  # More frequent frames
            
            if not frame_paths:
                raise ValueError("No frames extracted from video")
            
            # Process each frame
            all_matches = []
            for frame_path in frame_paths:
                try:
                    # Detect fashion items
                    detections = self.detect_fashion_items(frame_path)
                    
                    if not detections:
                        continue
                    
                    # Crop detected items
                    cropped_items = self.crop_detected_items(frame_path, detections)
                    
                    # Generate CLIP embeddings
                    items_with_embeddings = self.generate_clip_embeddings(cropped_items)
                    
                    # Match to products with lower threshold
                    matches = self.match_products(items_with_embeddings)
                    if matches:
                        all_matches.extend(matches)
                except Exception as e:
                    logger.warning(f"Failed to process frame {frame_path}: {str(e)}")
                    continue
            
            # Classify vibe using combined text
            vibes = self.classify_vibe(combined_text)
            
            # Prepare final result
            result = {
                'video_id': Path(video_path).stem,
                'vibes': vibes,
                'products': all_matches,
                'audio': {
                    'transcription': audio_transcription,
                    'segments': audio_result['segments'],
                    'language': audio_result['language']
                }
            }
            
            # Save to JSON file
            output_path = os.path.join("outputs", f"{result['video_id']}.json")
            
            # Create archive directory if it doesn't exist
            archive_dir = os.path.join("outputs", "archive")
            os.makedirs(archive_dir, exist_ok=True)
            
            # Archive old output if it exists
            if os.path.exists(output_path):
                archive_path = os.path.join(archive_dir, f"{result['video_id']}_{int(time.time())}.json")
                os.rename(output_path, archive_path)
                logger.info(f"Archived old output to {archive_path}")
            
            # Save current output
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Successfully processed video: {video_path}")
            logger.info(f"Found {len(all_matches)} product matches")
            return result
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            raise

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 