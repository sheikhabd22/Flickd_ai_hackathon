import pandas as pd
import os

def load_and_process_data():
    # Read CSV files
    products_df = pd.read_csv('data/products.csv')
    images_df = pd.read_csv('images.csv')
    
    # Group images by product id to handle multiple images per product
    grouped_images = images_df.groupby('id')['image_url'].apply(list).reset_index()
    
    # Merge product data with image data
    merged_df = pd.merge(
        products_df,
        grouped_images,
        on='id',
        how='left'
    )
    
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Save processed data
    merged_df.to_json('outputs/processed_data.json', orient='records')
    
    return merged_df

def get_product_with_images(product_id):
    merged_df = pd.read_json('outputs/processed_data.json')
    return merged_df[merged_df['id'] == product_id].to_dict('records')[0]

if __name__ == "__main__":
    processed_data = load_and_process_data()
    print(f"Processed {len(processed_data)} products with their images")