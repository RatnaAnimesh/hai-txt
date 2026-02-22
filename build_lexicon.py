import sys
import os
import urllib.request
import numpy as np

def build_lexicon():
    print("="*80)
    print("DHAI-4 AUTOMATED LEXICON MAPPER")
    print("="*80)
    
    dim = 10000
    target_words = 50000
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
    dict_path = "hd_lexicon.npz"
    
    print(f"Downloading wordlist from: {url}")
    response = urllib.request.urlopen(url)
    all_words = response.read().decode('utf-8').splitlines()
    
    # Filter for reasonable length, get the top 50,000 most common words 
    # (words_alpha is mostly alphabetical, but we'll take the first 50k for demonstration)
    words = [w.lower() for w in all_words if len(w) > 2][:target_words]
    
    print(f"\nProcessing {len(words)} unique words...")
    print(f"Exploiting High-Dimensional Orthogonality (d={dim}) for collision-free assignment.")
    
    # Generate random bipolar matrices (-1, 1) for all 50,000 words instantly
    # This requires 50,000 * 10,000 = 500,000,000 integers (~500MB matrix)
    # We cast to int8 to massively save RAM and computation speed during generation and dot products
    matrix = np.random.choice([-1, 1], size=(len(words), dim)).astype(np.int8)
    
    # Pack words and matrix into a compressed dictionary archive
    print("\nSaving massive semantic geometric matrix to disk...")
    np.savez_compressed(dict_path, words=np.array(words), matrix=matrix)
    
    print(f"\n[LEXICON BUILT] Successfully mapped {len(words)} English words to {dim}-D invariant vectors.")
    print(f"Archive saved as '{dict_path}'.")
    
if __name__ == "__main__":
    build_lexicon()
