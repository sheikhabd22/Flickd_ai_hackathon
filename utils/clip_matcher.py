import os
import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import logging
from sklearn.preprocessing import normalize
import re
from sklearn.metrics.pairwise import cosine_similarity
import json

logger = logging.getLogger(__name__)

class ClipMatcher:
    def __init__(self):
        # Initialize CLIP model for image-text matching
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize indices and metadata
        self.product_index = None
        self.product_metadata = None
        self.index_path = "models/product_index.faiss"
        self.metadata_path = "models/product_metadata.json"
        self.text_index_path = "models/product_text_index.faiss"
        self.audio_index_path = "models/product_audio_index.faiss"
        
        # Enhanced weights for different matching components
        self.weights = {
            'image': 0.5,      # Weight for image similarity (increase since no ST)
            'text': 0.5,       # Weight for text similarity (CLIP only)
            'audio': 0.0,      # Not used
            'title': 0.4,
            'desc': 0.3,
            'tags': 0.3,
            'color': 0.2,
            'style': 0.2,
            'price': 0.1
        }
        
        # Dynamic similarity thresholds
        self.similarity_thresholds = {
            'exact': 0.85,
            'similar': 0.65,
            'loose': 0.45
        }
        
        # Load fashion-specific attributes
        self.fashion_attributes = self._load_fashion_attributes()

    def _load_fashion_attributes(self) -> Dict[str, List[str]]:
        """Load fashion-specific attributes for better matching."""
        attributes = {
            'styles': [
                'casual', 'formal', 'sporty', 'elegant', 'vintage', 'modern',
                'bohemian', 'minimalist', 'streetwear', 'classic', 'trendy'
            ],
            'colors': [
                'black', 'white', 'red', 'blue', 'green', 'yellow', 'purple',
                'pink', 'orange', 'brown', 'gray', 'beige', 'navy', 'burgundy'
            ],
            'patterns': [
                'solid', 'striped', 'floral', 'plaid', 'checkered', 'polka dot',
                'geometric', 'abstract', 'animal print', 'tie-dye'
            ],
            'materials': [
                'cotton', 'silk', 'wool', 'linen', 'denim', 'leather', 'suede',
                'polyester', 'nylon', 'velvet', 'lace', 'knit'
            ]
        }
        return attributes

    def _truncate_text(self, text: str, max_length: int = 75) -> str:
        """Enhanced text truncation with semantic preservation."""
        if not isinstance(text, str):
            return ""
        
        # Tokenize the text
        tokens = self.tokenizer.tokenize(text)
        
        if len(tokens) <= max_length:
            return text
            
        # Smart truncation: keep important parts
        # 1. Keep the beginning (context)
        # 2. Keep the end (conclusion)
        # 3. Keep any important keywords in the middle
        keep_start = max_length // 3
        keep_end = max_length // 3
        keep_middle = max_length - keep_start - keep_end
        
        # Extract important keywords
        important_tokens = []
        for token in tokens[keep_start:-keep_end]:
            if any(attr in token.lower() for attr_list in self.fashion_attributes.values() for attr in attr_list):
                important_tokens.append(token)
        
        # Combine tokens
        final_tokens = (
            tokens[:keep_start] +
            important_tokens[:keep_middle] +
            tokens[-keep_end:]
        )
        
        return self.tokenizer.convert_tokens_to_string(final_tokens)

    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing with fashion-specific handling."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important fashion-related ones
        text = re.sub(r'[^a-z0-9\s\-/]', ' ', text)
        
        # Normalize fashion-specific terms
        fashion_terms = {
            't-shirt': 'tshirt',
            't shirt': 'tshirt',
            'tank top': 'tanktop',
            'crop top': 'croptop',
            'high waist': 'highwaist',
            'low waist': 'lowwaist'
        }
        for term, replacement in fashion_terms.items():
            text = text.replace(term, replacement)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate to fit CLIP's token limit
        return self._truncate_text(text)

    def _extract_product_features(self, product: pd.Series) -> Tuple[str, Dict[str, Any]]:
        """Enhanced product feature extraction with fashion-specific attributes."""
        # Preprocess text fields
        title = self._preprocess_text(product['title'])
        description = self._preprocess_text(product['description'])
        tags = self._preprocess_text(product['product_tags'])
        
        # Extract attributes
        attributes = {
            'color': 'unknown',
            'style': 'unknown',
            'pattern': 'unknown',
            'material': 'unknown',
            'price_range': 'unknown'
        }
        
        # Extract color
        for color in self.fashion_attributes['colors']:
            if color in tags.lower() or color in description.lower():
                attributes['color'] = color
                break
        
        # Extract style
        for style in self.fashion_attributes['styles']:
            if style in tags.lower() or style in description.lower():
                attributes['style'] = style
                break
        
        # Extract pattern
        for pattern in self.fashion_attributes['patterns']:
            if pattern in tags.lower() or pattern in description.lower():
                attributes['pattern'] = pattern
                break
        
        # Extract material
        for material in self.fashion_attributes['materials']:
            if material in tags.lower() or material in description.lower():
                attributes['material'] = material
                break
        
        # Determine price range
        try:
            price = float(product['price_display_amount'])
            if price < 50:
                attributes['price_range'] = 'budget'
            elif price < 100:
                attributes['price_range'] = 'mid-range'
            else:
                attributes['price_range'] = 'premium'
        except (ValueError, TypeError):
            pass
        
        # Combine text fields with weights
        combined_text = f"{title} {tags} {description}"
        
        return combined_text, attributes

    def generate_clip_embeddings(self, cropped_items: List[Dict[str, Any]], audio_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate enhanced CLIP embeddings with audio integration."""
        logger.info("Generating CLIP embeddings")
        items_with_embeddings = []

        for item in cropped_items:
            try:
                # Ensure image is in RGB format
                if isinstance(item['cropped_image'], Image.Image):
                    image = item['cropped_image'].convert('RGB')
                else:
                    image = Image.fromarray(item['cropped_image']).convert('RGB')
                
                # Generate image embedding
                inputs = self.clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_embedding = self.clip_model.get_image_features(**inputs)
                    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                    image_embedding = image_embedding.squeeze(0).numpy().astype("float32")
                
                # Generate text embedding based on detection class
                detection_class = item['detection']['class'].lower()
                if detection_class == 'upper_body':
                    text = "a shirt or top or jacket or sweater or hoodie"
                elif detection_class == 'lower_body':
                    text = "pants or jeans or skirt or shorts"
                else:
                    text = f"a {detection_class}"
                
                # Add audio context if available
                if audio_text:
                    text = f"{text} {audio_text}"
                
                # Use CLIP for text
                text_inputs = self.clip_processor(text=text, return_tensors="pt")
                with torch.no_grad():
                    clip_text_embedding = self.clip_model.get_text_features(**text_inputs)
                    clip_text_embedding = clip_text_embedding / clip_text_embedding.norm(dim=-1, keepdim=True)
                    clip_text_embedding = clip_text_embedding.squeeze(0).numpy().astype("float32")
                
                items_with_embeddings.append({
                    'detection': item['detection'],
                    'image_embedding': image_embedding,
                    'clip_text_embedding': clip_text_embedding
                })
                logger.info(f"Generated embeddings for {detection_class}")
            except Exception as e:
                logger.error(f"Failed to generate embedding for item: {str(e)}")
                continue

        logger.info(f"Generated embeddings for {len(items_with_embeddings)} items")
        return items_with_embeddings

    def match_embedding(self, query_embedding: Dict[str, np.ndarray], top_k=5, similarity_threshold=None) -> List[Dict[str, Any]]:
        """Matching using only CLIP embeddings (image and text)."""
        if self.product_index is None or self.product_metadata is None:
            raise ValueError("Product index not built. Call build_product_index first.")

        if similarity_threshold is None:
            similarity_threshold = self.similarity_thresholds['similar']

        # Search in image index
        D_img, I_img = self.product_index.search(
            query_embedding['image_embedding'].reshape(1, -1), 
            k=top_k * 2
        )
        
        # Search in text indices
        D_clip_text, I_clip_text = self.product_text_index.search(
            query_embedding['clip_text_embedding'].reshape(1, -1), 
            k=top_k * 2
        )
        
        matches = []
        seen_products = set()
        for idx in range(len(D_img[0])):
            product_idx = I_img[0][idx]
            if product_idx in seen_products:
                continue
            product = self.product_metadata.iloc[product_idx]
            combined_text, attributes = self._extract_product_features(product)
            # Calculate weighted scores
            image_score = D_img[0][idx] * self.weights['image']
            clip_text_score = D_clip_text[0][idx] * self.weights['text']
            # Calculate attribute scores
            color_score = 1.0 if attributes['color'] != 'unknown' else 0.0
            style_score = 1.0 if attributes['style'] != 'unknown' else 0.0
            price_score = 1.0 if attributes['price_range'] != 'unknown' else 0.0
            # Combine all scores
            weighted_score = (
                image_score +
                clip_text_score +
                color_score * self.weights['color'] +
                style_score * self.weights['style'] +
                price_score * self.weights['price']
            )
            if weighted_score >= similarity_threshold:
                # Determine match type
                if weighted_score >= self.similarity_thresholds['exact']:
                    match_type = "exact"
                elif weighted_score >= self.similarity_thresholds['similar']:
                    match_type = "similar"
                else:
                    match_type = "loose"
                matches.append({
                    "type": product["product_type"].lower(),
                    "color": attributes['color'],
                    "style": attributes['style'],
                    "pattern": attributes['pattern'],
                    "material": attributes['material'],
                    "price_range": attributes['price_range'],
                    "matched_product_id": str(product["id"]),
                    "title": product["title"],
                    "description": product["description"],
                    "price": float(product["price_display_amount"]),
                    "match_type": match_type,
                    "confidence": float(weighted_score),
                    "match_source": "combined",
                    "attributes": attributes
                })
                seen_products.add(product_idx)
        # Sort matches by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        # Apply diversity ranking
        diverse_matches = []
        seen_types = set()
        seen_colors = set()
        for match in matches:
            if len(diverse_matches) >= top_k:
                break
            # Check if this match adds diversity
            if (match['type'] not in seen_types or 
                match['color'] not in seen_colors):
                diverse_matches.append(match)
                seen_types.add(match['type'])
                seen_colors.add(match['color'])
        return diverse_matches

    def build_product_index(self, products_csv: str, images_dir: str):
        """Build or load FAISS index for product images and text with enhanced preprocessing."""
        logger.info("Building/loading product index")

        # Try loading existing indices
        if (os.path.exists(self.index_path) and 
            os.path.exists(self.metadata_path) and 
            os.path.exists(self.text_index_path)):
            logger.info("Loading existing FAISS indices and metadata")
            try:
                self.product_index = faiss.read_index(self.index_path)
                self.product_text_index = faiss.read_index(self.text_index_path)
                self.product_metadata = pd.read_json(self.metadata_path)
                logger.info(f"Successfully loaded indices with {len(self.product_metadata)} products")
                return
            except Exception as e:
                logger.error(f"Failed to load existing indices: {str(e)}")
                logger.info("Will rebuild indices from scratch")

        logger.info(f"Reading products from {products_csv}")
        products_df = pd.read_csv(products_csv)
        logger.info(f"Found {len(products_df)} products in CSV")
        
        image_embeddings = []
        text_embeddings = []
        valid_indices = []
        failed_products = []

        for idx, row in products_df.iterrows():
            img_path = os.path.join(images_dir, f"{row['id']}.jpg")
            if os.path.exists(img_path):
                try:
                    # Process image
                    image = Image.open(img_path).convert("RGB")
                    image_inputs = self.clip_processor(images=image, return_tensors="pt")
                    
                    # Process text with enhanced preprocessing
                    combined_text, attributes = self._extract_product_features(row)
                    text_inputs = self.clip_processor(text=combined_text, return_tensors="pt")
                    
                    with torch.no_grad():
                        # Get image features
                        image_features = self.clip_model.get_image_features(**image_inputs)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        image_features = image_features.squeeze(0).numpy().astype("float32")
                        
                        # Get text features
                        text_features = self.clip_model.get_text_features(**text_inputs)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        text_features = text_features.squeeze(0).numpy().astype("float32")
                    
                    image_embeddings.append(image_features)
                    text_embeddings.append(text_features)
                    valid_indices.append(idx)
                except Exception as e:
                    failed_products.append(row['id'])
                    logger.warning(f"Failed to process product {row['id']}: {str(e)}")

        if not image_embeddings:
            raise ValueError("No valid product images found")

        logger.info(f"Successfully processed {len(valid_indices)} products")
        if failed_products:
            logger.warning(f"Failed to process {len(failed_products)} products: {failed_products}")

        # Stack embeddings
        image_embeddings = np.vstack(image_embeddings)
        text_embeddings = np.vstack(text_embeddings)
        metadata = products_df.iloc[valid_indices].reset_index(drop=True)

        logger.info(f"Building FAISS indices with {len(valid_indices)} products")
        # Build FAISS indices for cosine similarity
        self.product_index = faiss.IndexFlatIP(image_embeddings.shape[1])
        self.product_index.add(image_embeddings)
        
        self.product_text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
        self.product_text_index.add(text_embeddings)

        # Save indices and metadata
        os.makedirs("models", exist_ok=True)
        faiss.write_index(self.product_index, self.index_path)
        faiss.write_index(self.product_text_index, self.text_index_path)
        metadata.to_json(self.metadata_path)

        self.product_metadata = metadata
        logger.info(f"FAISS indices built successfully with {len(valid_indices)} products") 