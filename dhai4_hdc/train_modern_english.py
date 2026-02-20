import sys
import os
import urllib.request
import json
import time
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhai4_hdc.models.level0_broca import HDC_SensoryInterface
from dhai4_hdc.models.level1_wernicke import HDC_EventProcessor
from dhai4_hdc.models.level2_frontal import HDC_NarrativePlanner
from dhai4_hdc.utils.data_loader import CorpusLoader

def fetch_random_wikipedia_articles(limit=5):
    """
    Fetches raw modern English text directly from Wikipedia.
    This effectively streams 'new english' infinitely without needing hard drive storage.
    """
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext&generator=random&grnnamespace=0&grnlimit={limit}"
    req = urllib.request.Request(url, headers={'User-Agent': 'DHAI4-Trainer/1.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            return [page.get('extract', '') for page in pages.values() if page.get('extract')]
    except Exception as e:
        print(f" [!] Fetch error: {e}")
        return []

def train_on_modern_english():
    print("=== DHAI-4 HDC: Modern English Infinite Training ===")
    print("Target: Random Wikipedia Articles (Streaming & Discarding)")
    
    dim = 10000 
    
    # Initialize the architecture
    broca = HDC_SensoryInterface(dimension=dim)
    wernicke = HDC_EventProcessor(dimension=dim, sim_threshold=0.1) 
    frontal = HDC_NarrativePlanner(dimension=dim, search_depth=3)
    frontal.set_goal(["Understand", "Language"])
    
    loader = CorpusLoader()
    
    total_tokens_processed = 0
    total_events_chunked = 0
    batch_count = 0
    
    target_batches = 5 # Run an example few batches to prove it works before infinite loop
    
    print("\n--- Commencing Streaming Training ---")
    start_time = time.time()
    
    try:
        while batch_count < target_batches:
            batch_count += 1
            print(f"\n[Batch {batch_count}] Fetching random modern English articles...")
            articles = fetch_random_wikipedia_articles(limit=10)
            
            if not articles:
                time.sleep(2)
                continue
                
            print(f"  -> Acquired {len(articles)} articles. Processing and discarding...")
            
            batch_tokens = 0
            batch_events = 0
            prev_l0_bound = None
            running_l1_sequence = []
            
            for text in articles:
                stream = loader.stream_tokens(text_content=text)
                
                for token in stream:
                    batch_tokens += 1
                    total_tokens_processed += 1
                    
                    # Output progress cleanly
                    if total_tokens_processed % 5000 == 0:
                        sys.stdout.write(f"\r    Global Tokens: {total_tokens_processed} | Unique HDC Concepts: {len(broca.hd_space.item_memory)}")
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
                        total_events_chunked += 1
                        batch_events += 1
                        new_event_vector = events[-1] 
                        frontal.observe(new_event_vector)
                        running_l1_sequence = [curr_bound]
            print(f"\n  -> Batch Complete. Discarded raw text. (Tokens: {batch_tokens}, Events: {batch_events})")
            
            # --- Bayesian Model Reduction (Sleep / Consolidation Phase) ---
            print("  -> Entering Sleep Phase (Consolidating Memory)...")
            pruned_nodes = broca.sleep_cycle(prune_threshold=2)
            print(f"  -> Waking Up. Pruned {pruned_nodes} weak geometric geometric transitions. Reclaimed capacity.")

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")
        
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n=== Training Run Paused / Complete ===")
    print(f"Time Elapsed: {elapsed:.2f}s")
    print(f"Total Tokens Processed: {total_tokens_processed}")
    print(f"Total Macroscopic Events Chunked: {total_events_chunked}")
    print(f"Final Semantic Vectors Learned: {len(broca.hd_space.item_memory)}")
    print(f"Grammatical Transitions Stored: {len(broca.transition_counts)}")

if __name__ == "__main__":
    train_on_modern_english()
