import pandas as pd
import pyarrow.parquet as pq
import argparse
import os
from pathlib import Path

def convert_json_to_parquet(input_dir, output_dir):
    """
    Convert all JSON files in input_dir to Parquet files in output_dir.
    
    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to save Parquet files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files in input directory
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON file(s) to convert")
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        
        # Read JSON into a Pandas DataFrame
        df = pd.read_json(json_file)
        
        # Create output filename (same name but with .parquet extension)
        output_file = output_path / f"{json_file.stem}.parquet"
        
        # Write the DataFrame to a Parquet file
        df.to_parquet(output_file)
        print(f"Saved to: {output_file}")
        
        # Read back and display info
        df_read = pd.read_parquet(output_file)
        print(f"Rows: {len(df_read)}, Columns: {len(df_read.columns)}")
        print(f"Columns: {list(df_read.columns)}")
        print("---")

def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON files to Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing JSON files"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for Parquet files"
    )
    
    args = parser.parse_args()
    
    convert_json_to_parquet(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()