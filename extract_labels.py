import os
import csv
import argparse

def extract_labels(image_dir, full_csv_path, out_csv_path):
    # 1. Get the list of IDs from the image directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    # Assuming the ID is the filename without the extension
    target_ids = set()
    for f in image_files:
        # e.g., '1020551475201546.jpg' -> '1020551475201546'
        base_name = os.path.splitext(f)[0]
        target_ids.add(base_name)
        
    print(f"Found {len(target_ids)} target IDs in '{image_dir}'")
    
    # 2. Stream the huge CSV to extract matching rows
    matched_data = []
    
    print(f"Reading '{full_csv_path}'... this might take a minute since the file is large.")
    try:
        with open(full_csv_path, mode='r', encoding='utf-8') as fin:
            reader = csv.DictReader(fin)
            
            # Verify columns exist
            if not reader.fieldnames or 'id' not in reader.fieldnames or 'country' not in reader.fieldnames:
                raise ValueError(f"CSV must contain 'id' and 'country' columns. Found headers: {reader.fieldnames}")
                
            for row in reader:
                current_id = row['id']
                if current_id in target_ids:
                    # We found a match, capture the needed columns
                    matched_data.append({
                        "id": current_id, 
                        "country": row['country']
                    })
                    
                    # Optimization: If we found all of them, we can stop reading the 2.9GB file early!
                    if len(matched_data) == len(target_ids):
                        print(f"Found all {len(target_ids)} target IDs early! Stopping search.")
                        break
                        
    except FileNotFoundError:
        print(f"Error: The source CSV file '{full_csv_path}' was not found. Please check the path.")
        return

    print(f"Matched {len(matched_data)} out of {len(target_ids)} IDs.")
    
    if len(matched_data) == 0:
        print("No matches found. Skipping file writing.")
        return

    # 3. Write purely the matched data to the new lightweight CSV
    print(f"Writing results to '{out_csv_path}'...")
    with open(out_csv_path, mode='w', encoding='utf-8', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=["id", "country"])
        writer.writeheader()
        writer.writerows(matched_data)
        
    print("Label extraction complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract country labels for a specific set of image IDs.")
    parser.add_argument("--image_dir", type=str, default="osv5m_sampled_1000", help="Directory containing the target images (the IDs).")
    parser.add_argument("--source_csv", type=str, default="train.csv", help="Path to the original massive CSV.")
    parser.add_argument("--out_csv", type=str, default="sampled_labels.csv", help="Path for the output CSV.")
    
    args = parser.parse_args()
    
    extract_labels(args.image_dir, args.source_csv, args.out_csv)
