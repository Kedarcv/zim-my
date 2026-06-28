#!/usr/bin/env python3
"""
Manual download script for datasets that failed in the main download script.
Handles corrupted/malformed files with error recovery.
"""

import os
import json
import zipfile
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import Dataset

# Create directories
os.makedirs("data/raw", exist_ok=True)

print("=" * 60)
print("Manual Download: Failed Datasets")
print("=" * 60)

# Dataset 1: Zimbabwe Agriculture
print("\n[1/2] Downloading Zimbabwe Agriculture Dataset...")
try:
    # Download the zip file manually
    zip_path = hf_hub_download(
        repo_id="sairos/Zimbabwe_agriculture_dataset",
        filename="final_dataset.zip",
        repo_type="dataset"
    )
    
    print(f"  Downloaded to: {zip_path}")
    
    # Extract and parse with error handling
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List all files in the zip
        file_list = zip_ref.namelist()
        print(f"  Files in zip: {len(file_list)}")
        
        # Find CSV/JSON files
        data_files = [f for f in file_list if f.endswith(('.csv', '.json', '.jsonl'))]
        print(f"  Data files found: {data_files}")
        
        all_records = []
        
        for data_file in data_files:
            try:
                print(f"  Processing: {data_file}")
                with zip_ref.open(data_file) as f:
                    content = f.read().decode('utf-8')
                    
                    if data_file.endswith('.csv'):
                        # Try reading CSV with error handling
                        try:
                            df = pd.read_csv(f, on_bad_lines='skip')
                            records = df.to_dict('records')
                            all_records.extend(records)
                            print(f"    ✓ Loaded {len(records)} records from CSV")
                        except Exception as e:
                            print(f"    ✗ CSV error: {e}")
                            # Try as JSON lines
                            lines = content.strip().split('\n')
                            for line in lines:
                                try:
                                    record = json.loads(line)
                                    all_records.append(record)
                                except:
                                    pass
                            print(f"    ✓ Loaded {len(records)} records as JSONL")
                    
                    elif data_file.endswith('.json'):
                        data = json.loads(content)
                        if isinstance(data, list):
                            all_records.extend(data)
                            print(f"    ✓ Loaded {len(data)} records from JSON")
                        elif isinstance(data, dict):
                            all_records.append(data)
                            print(f"    ✓ Loaded 1 record from JSON")
                    
                    elif data_file.endswith('.jsonl'):
                        lines = content.strip().split('\n')
                        for line in lines:
                            try:
                                record = json.loads(line)
                                all_records.append(record)
                            except:
                                pass
                        print(f"    ✓ Loaded {len(all_records)} records from JSONL")
            
            except Exception as e:
                print(f"    ✗ Error processing {data_file}: {e}")
                continue
        
        if all_records:
            # Filter out non-dict records
            valid_records = [r for r in all_records if isinstance(r, dict)]
            if valid_records:
                # Convert to HuggingFace Dataset
                ds = Dataset.from_list(valid_records)
                ds.save_to_disk("data/raw/zimbabwe_agriculture")
                print(f"  ✓ Saved {len(ds)} records to data/raw/zimbabwe_agriculture")
            else:
                print("  ✗ No valid records (all were non-dict)")
        else:
            print("  ✗ No records could be extracted")

except Exception as e:
    print(f"  ✗ Failed to download: {e}")

# Dataset 2: Zimbabwe History & Heritage
print("\n[2/2] Downloading Zimbabwe History & Heritage Dataset...")
try:
    # Try to list all files in the repo
    from huggingface_hub import list_repo_files
    
    files = list_repo_files("Ruramai/zimbabwe_history_heritage", repo_type="dataset")
    print(f"  Files in repo: {len(files)}")
    
    # Download all JSON/JSONL files
    all_records = []
    
    for filename in files:
        if filename.endswith(('.json', '.jsonl', '.json.gz', '.jsonl.gz')):
            try:
                print(f"  Downloading: {filename}")
                file_path = hf_hub_download(
                    repo_id="Ruramai/zimbabwe_history_heritage",
                    filename=filename,
                    repo_type="dataset"
                )
                
                # Read the file
                if filename.endswith('.gz'):
                    import gzip
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        content = f.read()
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                # Parse based on format
                if '.jsonl' in filename:
                    lines = content.strip().split('\n')
                    for line in lines:
                        try:
                            record = json.loads(line)
                            all_records.append(record)
                        except:
                            pass
                    print(f"    ✓ Loaded {len(lines)} lines from JSONL")
                
                elif '.json' in filename:
                    try:
                        data = json.loads(content)
                        if isinstance(data, list):
                            all_records.extend(data)
                            print(f"    ✓ Loaded {len(data)} records from JSON")
                        elif isinstance(data, dict):
                            all_records.append(data)
                            print(f"    ✓ Loaded 1 record from JSON")
                    except:
                        # Try as JSONL
                        lines = content.strip().split('\n')
                        for line in lines:
                            try:
                                record = json.loads(line)
                                all_records.append(record)
                            except:
                                pass
                        print(f"    ✓ Loaded {len(all_records)} records as JSONL")
            
            except Exception as e:
                print(f"    ✗ Error with {filename}: {e}")
                continue
    
    if all_records:
        # Filter and normalize records
        valid_records = []
        for r in all_records:
            if isinstance(r, dict):
                # Normalize all values to strings
                normalized = {}
                for k, v in r.items():
                    if isinstance(v, (list, dict)):
                        normalized[k] = json.dumps(v, ensure_ascii=False)
                    elif v is None:
                        normalized[k] = ""
                    else:
                        normalized[k] = str(v)
                valid_records.append(normalized)
            elif isinstance(r, str):
                valid_records.append({"text": r})
        
        if valid_records:
            # Convert to HuggingFace Dataset
            ds = Dataset.from_list(valid_records)
            ds.save_to_disk("data/raw/zimbabwe_history")
            print(f"  ✓ Saved {len(ds)} records to data/raw/zimbabwe_history")
        else:
            print("  ✗ No valid records found")
    else:
        print("  ✗ No records could be extracted")

except Exception as e:
    print(f"  ✗ Failed to download: {e}")

print("\n" + "=" * 60)
print("Manual Download Complete!")
print("=" * 60)

# Show final dataset sizes
print("\nDataset sizes:")
for dataset_dir in Path("data/raw").iterdir():
    if dataset_dir.is_dir():
        size = sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file())
        print(f"  {dataset_dir.name}: {size / 1024 / 1024:.1f} MB")
