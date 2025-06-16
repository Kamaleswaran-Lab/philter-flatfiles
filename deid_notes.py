import pandas as pd
import requests
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from tqdm import tqdm

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
        pool_connections=20,  # Number of connection pools
        pool_maxsize=20       # Max connections per pool
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def process_single_note(session, note_text, row_index, max_retries=3):
    """Process a single note with error handling"""
    for attempt in range(max_retries):
        try:
            response = session.post(
                'http://localhost:8080/api/filter',
                data=note_text,
                headers={'Content-Type': 'text/plain'},
                params={'p': 'default'},
                timeout=30  # 30 second timeout
            )
            response.raise_for_status()
            return row_index, response.text
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed to process row {row_index} after {max_retries} attempts: {e}")
                return row_index, f"ERROR: {str(e)}"
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return row_index, "ERROR: Max retries exceeded"

def process_notes_parallel(df, note_column='DOC_TEXT', max_workers=10):
    """Process notes in parallel using ThreadPoolExecutor"""
    
    # Create a session for each worker
    sessions = [create_session() for _ in range(max_workers)]
    
    # Prepare tasks
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        for index, row in df.iterrows():
            note_text = str(row[note_column]) if pd.notna(row[note_column]) else ""
            session = sessions[index % max_workers]  # Round-robin session assignment
            
            future = executor.submit(process_single_note, session, note_text, index)
            tasks.append(future)
        
        # Process completed tasks with progress bar
        results = {}
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing notes"):
            row_index, filtered_text = future.result()
            results[row_index] = filtered_text
    
    # Apply results back to dataframe
    df['notes_deid'] = df.index.map(results)
    
    return df

def main():
    print("Loading CSV file...")
    # Load CSV
    df = pd.read_csv('/mnt/c/klduke/emorydata/Combined_Radiology_Notes_with_EncounterNumber.dsv', 
                     sep='|')  # Increased from 10 for testing
    
    print(df.shape)
    print(f"Processing {len(df)} rows...")
    start_time = time.time()
    
    # Process notes in parallel
    df = process_notes_parallel(df, note_column='DOC_TEXT', max_workers=10)
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    # Save filtered CSV
    output_path = '/mnt/c/klduke/emorydata/Combined_Radiology_Notes_with_EncounterNumber_deid.dsv'
    df.to_csv(output_path, index=False, sep='|')
    print(f"Results saved to {output_path}")
    
    # Print some statistics
    error_count = df['notes_deid'].str.contains('ERROR:', na=False).sum()
    print(f"Successfully processed: {len(df) - error_count}/{len(df)} rows")
    if error_count > 0:
        print(f"Errors encountered: {error_count} rows")

if __name__ == "__main__":
    main()