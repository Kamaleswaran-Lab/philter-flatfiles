import pandas as pd
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path
import re

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Unique separator token that's unlikely to appear in medical text
SEPARATOR_TOKEN = "|||PHILTER_ROW_SEPARATOR|||"

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
        pool_connections=10,
        pool_maxsize=10
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def batch_texts_with_separator(texts, batch_size=100):
    """Batch texts together with separator token"""
    batches = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # Clean texts and replace any existing separator tokens
        cleaned_texts = []
        for text in batch_texts:
            if pd.isna(text):
                cleaned_texts.append("")
            else:
                # Remove any existing separator tokens from the text
                cleaned_text = str(text).replace(SEPARATOR_TOKEN, " ")
                cleaned_texts.append(cleaned_text)
        
        # Join with separator
        combined_text = SEPARATOR_TOKEN.join(cleaned_texts)
        batches.append((i, combined_text, len(batch_texts)))
    
    return batches

def process_batch_request(session, combined_text, batch_start_idx, expected_count, max_retries=3):
    """Send batched text to Philter API and return split results"""
    for attempt in range(max_retries):
        try:
            response = session.post(
                'http://philter:8080/api/filter',
                data=combined_text,
                headers={'Content-Type': 'text/plain'},
                params={'p': 'default'},
                timeout=300  # Longer timeout for batched requests
            )
            response.raise_for_status()
            
            # Split response back into individual texts
            filtered_texts = response.text.split(SEPARATOR_TOKEN)
            
            # Validate we got the expected number of results
            if len(filtered_texts) != expected_count:
                print(f"Warning: Expected {expected_count} results, got {len(filtered_texts)} for batch starting at {batch_start_idx}")
                # Pad with empty strings if we got fewer results
                while len(filtered_texts) < expected_count:
                    filtered_texts.append("ERROR: Missing result")
                # Truncate if we got more results
                filtered_texts = filtered_texts[:expected_count]
            
            return batch_start_idx, filtered_texts
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed to process batch starting at {batch_start_idx} after {max_retries} attempts: {e}")
                # Return error results for the entire batch
                error_results = [f"ERROR: {str(e)}"] * expected_count
                return batch_start_idx, error_results
            time.sleep(2 ** attempt)
    
    # Fallback error case
    error_results = [f"ERROR: Max retries exceeded"] * expected_count
    return batch_start_idx, error_results

def process_chunk_batched(df_chunk, chunk_id, note_column='DOC_TEXT', batch_size=100):
    """Process a chunk using batched API calls"""
    print(f"Processing chunk {chunk_id} with {len(df_chunk)} rows in batches of {batch_size}...")
    import pdb; pdb.set_trace()
    # Extract texts from the chunk
    texts = df_chunk[note_column].tolist()
    
    # Create batches
    batches = batch_texts_with_separator(texts, batch_size)
    print(f"Created {len(batches)} batches for chunk {chunk_id}")
    
    # Create session
    session = create_session()
    
    # Process batches
    all_results = {}
    
    for batch_start_idx, combined_text, expected_count in tqdm(batches, desc=f"Chunk {chunk_id} batches"):
        batch_start_idx_global = df_chunk.index[batch_start_idx]
        
        start_idx, filtered_texts = process_batch_request(
            session, combined_text, batch_start_idx_global, expected_count
        )
        
        # Map results back to original indices
        for i, filtered_text in enumerate(filtered_texts):
            original_idx = df_chunk.index[batch_start_idx + i]
            all_results[original_idx] = filtered_text
    
    # Apply results back to chunk
    df_chunk = df_chunk.copy()
    df_chunk['notes_deid'] = df_chunk.index.map(all_results)
    
    return df_chunk

def get_total_rows(input_file):
    """Get total number of rows without loading entire file"""
    print("Counting total rows...")
    total_rows = 0
    
    # Read file in small chunks just to count rows
    for chunk in pd.read_csv(input_file, chunksize=1000):
        total_rows += len(chunk)
    
    return total_rows

def save_chunk_result(chunk_df, chunk_id, output_dir):
    """Save processed chunk to file"""
    output_path = output_dir / f"batch_chunk_{chunk_id:03d}_processed.csv"
    chunk_df.to_csv(output_path, index=False)
    return output_path

def load_processed_chunks(output_dir):
    """Load all processed chunks and return chunk numbers that are done"""
    processed_chunks = set()
    chunk_files = list(output_dir.glob("batch_chunk_*_processed.dsv"))
    
    for file_path in chunk_files:
        # Extract chunk number from filename
        chunk_num = int(file_path.stem.split('_')[2])
        processed_chunks.add(chunk_num)
    
    return processed_chunks

def combine_processed_chunks(output_dir, final_output_path):
    """Combine all processed chunks into final output file"""
    print("Combining processed chunks...")
    
    chunk_files = sorted(output_dir.glob("batch_chunk_*_processed.dsv"))
    
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

def test_separator_token(input_file, sample_size=1000):
    """Test if separator token appears in the data"""
    print(f"Testing separator token '{SEPARATOR_TOKEN}' in sample data...")
    
    # Read a sample of the data
    sample_df = pd.read_csv(input_file, sep='|', nrows=sample_size)
    
    # Check if separator appears in any text column
    text_columns = sample_df.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        if sample_df[col].astype(str).str.contains(SEPARATOR_TOKEN, regex=False).any():
            print(f"WARNING: Separator token found in column '{col}'!")
            return False
    
    print("Separator token is safe to use.")
    return True

def main():
    # Configuration
    input_file = '/labs/collab/Imaging/Imaging-PHI/Emory_Images/Meta/Combined_Radiology_Notes_with_EncounterNumber.csv'
    output_dir = Path('/labs/collab/Imaging/Imaging-PHI/Emory_Images/AllNotes/batch_chunks')
    final_output = '/labs/collab/Imaging/Imaging-PHI/Emory_Images/AllNotes/Combined_Radiology_Notes_with_EncounterNumber_batch_deid.csv'
    target_num_chunks = 100
    batch_size = 50  # Number of rows to batch together per API call
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Test separator token safety
    #if not test_separator_token(input_file):
    #    print("Separator token conflict detected. Please choose a different separator.")
    #    return
    
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
    
    print(f"Starting batch processing (batch size: {batch_size} rows per API call)...")
    
    # Read and process file in chunks
    chunk_reader = pd.read_csv(input_file,  chunksize=chunk_size)
    
    for df_chunk in chunk_reader:
        if chunk_id in processed_chunks:
            print(f"Skipping chunk {chunk_id} (already processed)")
            chunk_id += 1
            continue
        
        try:
            chunk_start_time = time.time()
            
            # Reset index for the chunk to avoid mapping issues
            df_chunk = df_chunk.reset_index(drop=True)
            
            # Process chunk using batched approach
            processed_chunk = process_chunk_batched(
                df_chunk, chunk_id, 
                note_column='DOC_TEXT', 
                batch_size=batch_size
            )
            
            # Save chunk result
            output_path = save_chunk_result(processed_chunk, chunk_id, output_dir)
            
            chunk_time = time.time() - chunk_start_time
            api_calls_made = (len(df_chunk) + batch_size - 1) // batch_size  # Ceiling division
            print(f"Chunk {chunk_id} completed in {chunk_time:.2f}s using {api_calls_made} API calls, saved to {output_path}")
            
            # Print progress
            completed_chunks = len(load_processed_chunks(output_dir))
            estimated_total_chunks = (total_rows + chunk_size - 1) // chunk_size
            print(f"Progress: {completed_chunks}/{estimated_total_chunks} chunks completed")
            
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
        
        chunk_id += 1
    
    total_time = time.time() - total_start_time
    print(f"All chunks processed in {total_time:.2f}s")
    
    # Combine all processed chunks
    success = combine_processed_chunks(output_dir, final_output)
    
    if success:
        print("Batch processing completed successfully!")
        
        # Calculate efficiency stats
        estimated_total_chunks = (total_rows + chunk_size - 1) // chunk_size
        total_api_calls = estimated_total_chunks * ((chunk_size + batch_size - 1) // batch_size)
        print(f"Total API calls made: ~{total_api_calls} (vs {total_rows} individual calls)")
        print(f"Efficiency gain: ~{total_rows / total_api_calls:.1f}x fewer API calls")
        
        # Optional: Clean up chunk files
        cleanup = input("Delete individual chunk files? (y/n): ").lower().strip()
        if cleanup == 'y':
            for chunk_file in output_dir.glob("batch_chunk_*_processed.dsv"):
                chunk_file.unlink()
            print("Chunk files cleaned up.")
    else:
        print("Some chunks are missing. Please check the processing.")

if __name__ == "__main__":
    main() 
