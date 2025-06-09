import pandas as pd
import os

def convert_excel_to_csv(excel_path: str, output_dir: str):
    """Convert Excel file to CSV format."""
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    output_path = os.path.join(output_dir, "products.csv")
    df.to_csv(output_path, index=False)
    print(f"Converted Excel to CSV: {output_path}")

if __name__ == "__main__":
    # Convert the catalog Excel file
    excel_path = "../../catalog.xlsx"
    output_dir = "../data"
    convert_excel_to_csv(excel_path, output_dir) 