import os
import json
from main import FlickdAIEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline():
    """Test the complete pipeline with a sample video."""
    try:
        # Initialize engine
        logger.info("Initializing FlickdAIEngine...")
        engine = FlickdAIEngine()
        
        # Build product index
        logger.info("Building product index...")
        engine.build_product_index("data/products.csv", "images")
        
        # Test video processing
        test_video = "data/videos/video1.mp4"  # Replace with your test video
        test_caption = "Clean girl aesthetic with minimal white top and jeans #cleangirl #minimalstyle"
        
        if os.path.exists(test_video):
            logger.info(f"Processing test video: {test_video}")
            result = engine.process_video(test_video, test_caption)
            
            # Print results
            logger.info("\nResults:")
            logger.info(f"Video ID: {result['video_id']}")
            logger.info(f"Detected Vibes: {result['vibes']}")
            logger.info("\nMatched Products:")
            for product in result['products']:
                logger.info(f"- Type: {product['type']}")
                logger.info(f"  Color: {product['color']}")
                logger.info(f"  Product ID: {product['matched_product_id']}")
                logger.info(f"  Match Type: {product['match_type']}")
                logger.info(f"  Confidence: {product['confidence']:.2f}")
        else:
            logger.error(f"Test video not found: {test_video}")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_pipeline() 