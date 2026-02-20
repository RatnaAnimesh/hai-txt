import sys
import os
import urllib.request
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhai4_hdc.models.level0_broca import HDC_SensoryInterface
from dhai4_hdc.models.level1_wernicke import HDC_EventProcessor
from dhai4_hdc.models.level2_frontal import HDC_NarrativePlanner
from dhai4_hdc.utils.data_loader import CorpusLoader

# A list of Project Gutenberg Book IDs to train on (Classic English Literature)
# e.g., 1342: Pride and Prejudice, 84: Frankenstein, 11: Alice in Wonderland
GUTENBERG_IDS = [1342, 84, 11, 1661, 2701] 

def download_book(book_id: int, filepath: str) -> bool:
    """Downloads a plaintext book from Project Gutenberg."""
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    try:
        print(f"Downloading Book ID {book_id} from {url}...")
        # Add a realistic user agent to avoid 403 Forbidden
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        return True
    except urllib.error.URLError as e:
        # Fallback to older URL format if -0 isn't available
        try:
             url_fallback = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
             print(f"  Fallback: {url_fallback}")
             req = urllib.request.Request(url_fallback, headers={'User-Agent': 'Mozilla/5.0'})
             with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
             return True
        except Exception as e2:
             print(f"Failed to download Book {book_id}: {e2}")
             return False

def train_on_corpus():
    print("=== DHAI-4 HDC: Massive Corpus Training ===")
    
    dim = 10000 
    
    # Initialize the architecture
    broca = HDC_SensoryInterface(dimension=dim)
    wernicke = HDC_EventProcessor(dimension=dim, sim_threshold=0.1) 
    frontal = HDC_NarrativePlanner(dimension=dim, search_depth=3)
    
    # Just generic tracking variables for L2
    frontal.set_goal(["Understand", "Language"])
    
    loader = CorpusLoader()
    
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_corpus")
    os.makedirs(temp_dir, exist_ok=True)
    
    total_tokens_processed = 0
    total_events_chunked = 0
    
    for idx, book_id in enumerate(GUTENBERG_IDS):
        filepath = os.path.join(temp_dir, f"book_{book_id}.txt")
        
        # 1. Download
        success = download_book(book_id, filepath)
        if not success:
            continue
            
        print(f"\n--- Training on Book ID {book_id} ---")
        loader.file_path = filepath
        stream = loader.stream_tokens()
        
        prev_l0_bound = None
        running_l1_sequence = []
        token_count = 0
        event_count = 0
        
        start_time = time.time()
        
        # 2. Stream and Train 
        try:
            for token in stream:
                token_count += 1
                
                # Print progress
                if token_count % 10000 == 0:
                    sys.stdout.write(f"\r  Processed {token_count} tokens... L0 Graph Size: {len(broca.transition_counts)}")
                    sys.stdout.flush()

                # Level 0
                l0_out = broca.encode(token)
                curr_bound = l0_out["bound_state"]
                
                if prev_l0_bound is not None:
                    broca.learn_transition(prev_l0_bound, curr_bound)
                prev_l0_bound = curr_bound
                
                running_l1_sequence.append(curr_bound)
                
                # Level 1
                events = wernicke.process_stream(running_l1_sequence)
                
                if events:
                    event_count += 1
                    new_event_vector = events[-1] 
                    
                    # Level 2 Simulation (Updating belief context)
                    frontal.observe(new_event_vector)
                    # Note: We omit the heavy Deep Tree Search here because we are purely
                    # focused on unsupervised Hebbian structure learning for language modeling.
                    # We want to build the massive memory banks first.
                    
                    running_l1_sequence = [curr_bound]
                    
        except KeyboardInterrupt:
            print("\n[!] Training interrupted by user.")
            break
            
        end_time = time.time()
        total_tokens_processed += token_count
        total_events_chunked += event_count
        
        print(f"\n[Book {book_id} Complete] Time: {end_time - start_time:.2f}s")
        print(f"Tokens: {token_count} | L1 Events: {event_count} | L0 Unique Contexts: {len(broca.transition_counts)}")
        
        # 3. Clean up
        print(f"Deleting '{filepath}' to preserve storage...")
        os.remove(filepath)
        
    print("\n=== Global Training Run Complete ===")
    print(f"Total Tokens: {total_tokens_processed}")
    print(f"Total Macroscopic Events: {total_events_chunked}")
    print(f"Final HDC Item Memory (Atomic Concepts): {len(broca.hd_space.item_memory)}")
    print(f"Final L0 Syntax/Semantic Graph Nodes: {len(broca.transition_counts)}")

if __name__ == "__main__":
    train_on_corpus()
