import pandas as pd
import pyarrow.parquet as pq
import argparse
import os
from pathlib import Path

def keep_single_image_token(text: str) -> str:
    # 如果没有 <image>，原样返回
    if "<image>" not in text:
        return text
    # 只保留第一个，把后面的都删掉
    first = True
    parts = []
    for seg in text.split("<image>"):
        if first:
            parts.append(seg)
            parts.append("<image>")
            first = False
        else:
            parts.append(seg)
    return "".join(parts)

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

    print(f"Found {len(json_files)} JSON file(s) to convert and concatenate")

    # Collect all dataframes
    df_list = []
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        df = pd.read_json(json_file)

        # 对每一行 sample 清洗 turns 里的 content，将多个 <image> 压缩为 1 个
        for idx, row in df.iterrows():
            turns = row.get("turns")
            if not isinstance(turns, list):
                continue
            for turn in turns:
                inputs = turn.get("inputs", [])
                for inp in inputs:
                    c = inp.get("content", "")
                    if isinstance(c, str):
                        inp["content"] = keep_single_image_token(c)
            # 显式写回这一行（虽然上面已经原地改了，这里保证 df 一致性）
            df.at[idx, "turns"] = turns

        # Optionally keep track of source file
        df["__source_file__"] = json_file.name
        df_list.append(df)

    # Concatenate all JSON files into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    # Single output Parquet file
    output_file = output_path / "combined.parquet"

    # Write the combined DataFrame to a Parquet file
    combined_df.to_parquet(output_file)
    print(f"\nSaved combined parquet to: {output_file}")

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
        "--input_dir",
        type=str,
        help="Input directory containing JSON files",
        default="/home/ubuntu/Yidan/ARLArena/checkpoints/sft",
        required=False
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for Parquet files",
        default="/home/ubuntu/Yidan/ARLArena/checkpoints/sft/parquet",
        required=False
    )
    
    args = parser.parse_args()
    
    convert_json_to_parquet(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()