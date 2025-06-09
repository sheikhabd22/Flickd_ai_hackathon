import pandas as pd
import requests
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_image(args):
    """Download a single image."""
    image_url, save_path = args
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        logger.error(f"Failed to download {image_url}: {str(e)}")
        return False

def download_product_images(images_csv: str, output_dir: str):
    """Download all product images from the catalog."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Read product catalog
    logger.info(f"Reading image links from {images_csv}")
    df = pd.read_csv(images_csv)
    
    # Pick the first image for each product id
    first_images = df.groupby('id').first().reset_index()
    
    # Prepare download tasks
    download_tasks = []
    for _, row in first_images.iterrows():
        image_url = row['image_url']
        save_path = output_path / f"{row['id']}.jpg"
        if not save_path.exists():
            download_tasks.append((image_url, save_path))
    
    if not download_tasks:
        logger.info("All images already downloaded!")
        return
    
    # Download images in parallel
    logger.info(f"Downloading {len(download_tasks)} images...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(
            executor.map(download_image, download_tasks),
            total=len(download_tasks),
            desc="Downloading images"
        ))
    
    # Log results
    successful = sum(results)
    failed = len(results) - successful
    logger.info(f"Download complete: {successful} successful, {failed} failed")

if __name__ == "__main__":
    download_product_images("data/images.csv", "images") 