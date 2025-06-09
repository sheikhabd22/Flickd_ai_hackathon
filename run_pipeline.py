import os
from pathlib import Path
import json
from utils.download_images import download_product_images
from main import FlickdAIEngine

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "frames",
        "images",
        "data",
        "embeddings",
        "outputs",
        "models",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def run_pipeline():
    """Run the complete pipeline."""
    print("Setting up directories...")
    setup_directories()
    
    # Download product images
    print("\nDownloading product images...")
    download_product_images("data/products.csv", "images")
    
    # Initialize engine
    print("\nInitializing AI engine...")
    engine = FlickdAIEngine()
    
    # Build product index
    print("\nBuilding product index...")
    engine.build_product_index("data/products.csv", "images")
    
    # Process videos
    print("\nProcessing videos...")
    videos_dir = "data/videos"
    with open("data/captions.json", 'r') as f:
        captions = json.load(f)
    
    results = {}
    for video_file in os.listdir(videos_dir):
        if video_file.endswith(('.mp4', '.mov')):
            print(f"\nProcessing {video_file}...")
            video_path = os.path.join(videos_dir, video_file)
            video_id = Path(video_file).stem
            
            # Extract frames
            frames_dir = Path("frames") / video_id
            frames_dir.mkdir(exist_ok=True)
            frame_paths = engine.extract_frames(video_path, str(frames_dir))
            
            # Process frames
            all_matches = []
            for frame_path in frame_paths:
                detections = engine.detect_fashion_items(str(frame_path))
                matches = engine.match_products(str(frame_path), detections)
                all_matches.extend(matches)
            
            # Get video caption and classify vibe
            caption = captions.get(video_id, "")
            vibe = engine.classify_vibe(caption)
            
            # Store results
            results[video_id] = {
                "detected_products": list(set(all_matches)),
                "vibe": vibe,
                "confidence_score": 0.85
            }
    
    # Save results
    with open("outputs/results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_pipeline()