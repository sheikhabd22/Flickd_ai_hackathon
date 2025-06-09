import psycopg2
from psycopg2.extras import Json
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "flickd"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
        self.create_tables()
    
    def create_tables(self):
        """Create necessary tables if they don't exist."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_results (
                    video_id TEXT PRIMARY KEY,
                    caption TEXT,
                    vibe_classification JSONB,
                    product_matches JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()
    
    def store_results(self, video_id: str, caption: str, vibe: Dict[str, float], 
                     product_matches: Dict[str, Any]):
        """Store video processing results in the database."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO video_results (video_id, caption, vibe_classification, product_matches)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (video_id) DO UPDATE
                SET caption = EXCLUDED.caption,
                    vibe_classification = EXCLUDED.vibe_classification,
                    product_matches = EXCLUDED.product_matches,
                    created_at = CURRENT_TIMESTAMP
            """, (video_id, caption, Json(vibe), Json(product_matches)))
            self.conn.commit()
    
    def get_results(self, video_id: str) -> Dict[str, Any]:
        """Retrieve results for a specific video."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM video_results WHERE video_id = %s
            """, (video_id,))
            result = cur.fetchone()
            if result:
                return {
                    'video_id': result[0],
                    'caption': result[1],
                    'vibe_classification': result[2],
                    'product_matches': result[3],
                    'created_at': result[4]
                }
            return None
    
    def close(self):
        """Close the database connection."""
        self.conn.close() 