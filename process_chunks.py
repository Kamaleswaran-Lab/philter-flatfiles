import pandas as pd
import requests
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def create_session():
    """Create a requests session with connection pooling and retry strategy"""
    session = requests.Session()
    
    # Retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    # Mount adapter with retry strategy
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=20,
        pool_maxsize=20
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def process_single_note(session, note_text, row_index, max_retries=3):
    """Process a single note with error handling"""
    for attempt in range(max_retries):
        try:
            # Encode text as UTF-8 to handle Unicode characters
            encoded_text = note_text.encode('utf-8')
            
            response = session.post(
                'http://philter:8080/api/filter',
                data=encoded_text,
                headers={'Content-Type': 'text/plain; charset=utf-8'},
                params={'p': 'default'},
                timeout=120
            )
            response.raise_for_status()
            return row_index, response.text
            
        except UnicodeEncodeError as e:
            print(f"Unicode encoding error for row {row_index}: {e}")
            return row_index, f"ERROR: Unicode encoding error: {str(e)}"
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed to process row {row_index} after {max_retries} attempts: {e}")
                return row_index, f"ERROR: {str(e)}"
            time.sleep(2 ** attempt)
    
    return row_index, "ERROR: Max retries exceeded"

def process_chunk_parallel(df_chunk, chunk_id, note_column='DOC_TEXT', max_workers=10):
    """Process a single chunk in parallel"""
    print(f"Processing chunk {chunk_id} with {len(df_chunk)} rows...")
    
    # Create sessions for workers
    sessions = [create_session() for _ in range(max_workers)]
    
    # Prepare tasks
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        for idx, (index, row) in enumerate(df_chunk.iterrows()):
            note_text = str(row[note_column]) if pd.notna(row[note_column]) else ""
            session = sessions[idx % max_workers]
            
            future = executor.submit(process_single_note, session, note_text, index)
            tasks.append(future)
        
        # Process completed tasks with progress bar
        results = {}
        desc = f"Chunk {chunk_id}"
        for future in tqdm(as_completed(tasks), total=len(tasks), desc=desc):
            row_index, filtered_text = future.result()
            results[row_index] = filtered_text
    
    # Apply results back to chunk
    df_chunk = df_chunk.copy()
    df_chunk['notes_deid'] = df_chunk.index.map(results)
    
    return df_chunk

def get_total_rows(input_file):
    """Get total number of rows without loading entire file"""
    print("Counting total rows...")
    total_rows = 0
    
    # Read file in small chunks just to count rows
    for chunk in pd.read_csv(input_file, chunksize=1000, on_bad_lines='skip'):
        total_rows += len(chunk)
    
    return total_rows

def save_chunk_result(chunk_df, chunk_id, output_dir):
    """Save processed chunk to file"""
    output_path = output_dir / f"chunk_{chunk_id:03d}_processed.dsv"
    chunk_df.to_csv(output_path, index=False, sep='|')
    return output_path

def load_processed_chunks(output_dir):
    """Load all processed chunks and return chunk numbers that are done"""
    processed_chunks = set()
    chunk_files = list(output_dir.glob("chunk_*_processed.dsv"))
    
    for file_path in chunk_files:
        # Extract chunk number from filename
        chunk_num = int(file_path.stem.split('_')[1])
        processed_chunks.add(chunk_num)
    
    return processed_chunks

def combine_processed_chunks(output_dir, final_output_path):
    """Combine all processed chunks into final output file"""
    print("Combining processed chunks...")
    
    chunk_files = sorted(output_dir.glob("chunk_*_processed.dsv"))
    
    if not chunk_files:
        print("No processed chunk files found!")
        return False
    
    print(f"Found {len(chunk_files)} chunk files to combine")
    
    # Write header from first chunk
    first_chunk = pd.read_csv(chunk_files[0], sep='|', nrows=0)  # Just header
    first_chunk.to_csv(final_output_path, index=False, sep='|')
    
    # Append all chunks
    total_rows = 0
    with open(final_output_path, 'a') as f:
        for chunk_file in tqdm(chunk_files, desc="Combining chunks"):
            chunk_df = pd.read_csv(chunk_file, sep='|')
            chunk_df.to_csv(f, header=False, index=False, sep='|')
            total_rows += len(chunk_df)
    
    print(f"Final combined file saved to: {final_output_path}")
    print(f"Total rows in final file: {total_rows}")
    
    return True

def calculate_chunk_size(total_rows, num_chunks=100):
    """Calculate appropriate chunk size"""
    chunk_size = max(1, total_rows // num_chunks)
    print(f"Using chunk size: {chunk_size} rows per chunk")
    return chunk_size

def main():
    # Configuration
    input_file = '/labs/collab/Imaging/Imaging-PHI/Emory_Images/Meta/Combined_Radiology_Notes_with_EncounterNumber.csv'
    output_dir = Path('/labs/collab/Imaging/Imaging-PHI/Emory_Images/AllNotes/batch_chunks')
    final_output = '/labs/collab/Imaging/Imaging-PHI/Emory_Images/AllNotes/Combined_Radiology_Notes_with_EncounterNumber_batch_deid.csv'
    target_num_chunks = 100
    max_workers = 10
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get total rows without loading entire file
    total_rows = get_total_rows(input_file)
    print(f"Total rows in file: {total_rows}")
    
    # Calculate chunk size
    chunk_size = calculate_chunk_size(total_rows, target_num_chunks)
    
    # Check for already processed chunks
    processed_chunks = load_processed_chunks(output_dir)
    print(f"Found {len(processed_chunks)} already processed chunks")
    
    # Process file in chunks without loading entire file into memory
    chunk_id = 1
    total_start_time = time.time()
    
    print("Starting chunk-by-chunk processing...")
    
    # Read and process file in chunks
    chunk_reader = pd.read_csv(input_file, chunksize=chunk_size, on_bad_lines='skip')
   
    for df_chunk in chunk_reader:
        if chunk_id in processed_chunks:
            print(f"Skipping chunk {chunk_id} (already processed)")
            chunk_id += 1
            continue
        
        try:
            chunk_start_time = time.time()
            
            # Reset index for the chunk to avoid mapping issues
            df_chunk = df_chunk.reset_index(drop=True)
            
            # Process chunk
            processed_chunk = process_chunk_parallel(
                df_chunk, chunk_id, 
                note_column='DOC_TEXT', 
                max_workers=max_workers
            )
            
            # Save chunk result
            output_path = save_chunk_result(processed_chunk, chunk_id, output_dir)
            
            chunk_time = time.time() - chunk_start_time
            print(f"Chunk {chunk_id} completed in {chunk_time:.2f}s, saved to {output_path}")
            
            # Print progress
            completed_chunks = len(load_processed_chunks(output_dir))
            estimated_total_chunks = (total_rows + chunk_size - 1) // chunk_size  # Ceiling division
            print(f"Progress: {completed_chunks}/{estimated_total_chunks} chunks completed")
            
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
        
        chunk_id += 1
    
    total_time = time.time() - total_start_time
    print(f"All chunks processed in {total_time:.2f}s")
    
    # Combine all processed chunks
    success = combine_processed_chunks(output_dir, final_output)
    
    if success:
        print("Processing completed successfully!")
        
        # Optional: Clean up chunk files
        cleanup = input("Delete individual chunk files? (y/n): ").lower().strip()
        if cleanup == 'y':
            for chunk_file in output_dir.glob("chunk_*_processed.dsv"):
                chunk_file.unlink()
            print("Chunk files cleaned up.")
    else:
        print("Some chunks are missing. Please check the processing.")

if __name__ == "__main__":
    main() 