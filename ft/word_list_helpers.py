"""
Helper functions for loading common English word lists from various sources.
Replace the get_common_english_words() function in generate_language_dataset.py with these.
"""

import requests
from pathlib import Path
from typing import List, Set
import random


# ============================================================================
# Option 1: Load from online word frequency lists
# ============================================================================

def download_google_10000_words(cache_file: str = "google_10000_words.txt") -> List[str]:
    """
    Download Google's 10,000 most common English words.
    Source: https://github.com/first20hours/google-10000-english
    """
    cache_path = Path(cache_file)
    
    if cache_path.exists():
        print(f"Loading cached word list from {cache_file}")
        with open(cache_path, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
        return words
    
    print("Downloading Google 10,000 common words...")
    url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt"
    
    response = requests.get(url)
    response.raise_for_status()
    
    words = response.text.strip().split('\n')
    
    # Cache for future use
    with open(cache_path, 'w') as f:
        f.write('\n'.join(words))
    
    print(f"Downloaded and cached {len(words)} words")
    return words


def download_wordnet_words(cache_file: str = "wordnet_words.txt") -> List[str]:
    """
    Get words from NLTK's WordNet (requires nltk package).
    Provides a good mix of common words across parts of speech.
    """
    cache_path = Path(cache_file)
    
    if cache_path.exists():
        print(f"Loading cached word list from {cache_file}")
        with open(cache_path, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
        return words
    
    try:
        import nltk
        from nltk.corpus import wordnet
        
        # Download WordNet if not already present
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading WordNet...")
            nltk.download('wordnet', quiet=True)
        
        # Get all lemmas (base word forms)
        words = set()
        for synset in wordnet.all_synsets():
            for lemma in synset.lemmas():
                word = lemma.name()
                # Filter out multi-word expressions and words with underscores
                if '_' not in word and '-' not in word:
                    words.add(word.lower())
        
        words = sorted(list(words))
        
        # Cache for future use
        with open(cache_path, 'w') as f:
            f.write('\n'.join(words))
        
        print(f"Loaded {len(words)} words from WordNet")
        return words
        
    except ImportError:
        print("NLTK not installed. Install with: pip install nltk")
        raise


# ============================================================================
# Option 2: Load from local file
# ============================================================================

def load_words_from_file(filepath: str, 
                         min_length: int = 3,
                         max_length: int = 12) -> List[str]:
    """
    Load words from a text file (one word per line).
    Filters by length and removes non-alphabetic words.
    
    Args:
        filepath: Path to word list file
        min_length: Minimum word length to include
        max_length: Maximum word length to include
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        words = []
        for line in f:
            word = line.strip().lower()
            if (word and 
                word.isalpha() and 
                min_length <= len(word) <= max_length):
                words.append(word)
    
    print(f"Loaded {len(words)} words from {filepath}")
    return words


# ============================================================================
# Option 3: Curated word lists by category
# ============================================================================

def get_diverse_word_list(n_words: int = 1000,
                         balance_by_category: bool = True) -> List[str]:
    """
    Get a diverse set of words balanced across semantic categories.
    Uses Google's word list and tries to balance categories.
    
    Args:
        n_words: Target number of words
        balance_by_category: If True, ensures representation from different categories
    """
    # Get base word list
    all_words = download_google_10000_words()
    
    if not balance_by_category:
        return all_words[:n_words]
    
    # Define category keywords to help identify word types
    categories = {
        'colors': ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 
                   'brown', 'black', 'white', 'gray', 'grey', 'violet', 'indigo'],
        'animals': ['dog', 'cat', 'bird', 'fish', 'animal', 'horse', 'cow', 'pig',
                    'bear', 'wolf', 'lion', 'tiger', 'elephant', 'monkey'],
        'numbers': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
                    'nine', 'ten', 'hundred', 'thousand', 'million', 'zero'],
        'time': ['time', 'day', 'night', 'year', 'month', 'week', 'hour', 'minute',
                 'morning', 'evening', 'afternoon', 'today', 'tomorrow', 'yesterday'],
        'body': ['head', 'face', 'eye', 'hand', 'body', 'arm', 'leg', 'foot',
                 'heart', 'brain', 'mouth', 'nose', 'ear', 'finger'],
        'food': ['food', 'eat', 'drink', 'water', 'bread', 'meat', 'milk', 'rice',
                 'fruit', 'vegetable', 'apple', 'chicken', 'fish', 'egg'],
        'action_verbs': ['walk', 'run', 'go', 'come', 'make', 'take', 'give', 'get',
                        'see', 'look', 'know', 'think', 'say', 'tell', 'find'],
        'adjectives': ['good', 'bad', 'big', 'small', 'long', 'short', 'hot', 'cold',
                      'fast', 'slow', 'high', 'low', 'new', 'old', 'happy', 'sad'],
    }
    
    # Try to get diverse representation
    selected = set()
    words_per_category = n_words // len(categories)
    
    # First pass: get words from each category
    for category, keywords in categories.items():
        category_words = [w for w in all_words if any(kw in w for kw in keywords)]
        selected.update(category_words[:words_per_category])
    
    # Second pass: fill remaining with most common words
    remaining = n_words - len(selected)
    for word in all_words:
        if word not in selected:
            selected.add(word)
            remaining -= 1
            if remaining <= 0:
                break
    
    words = list(selected)
    random.shuffle(words)
    return words[:n_words]


# ============================================================================
# Option 4: Filter by part of speech
# ============================================================================

def get_words_by_pos(pos_tags: List[str] = ['NOUN', 'VERB', 'ADJ', 'ADV'],
                     n_words: int = 1000,
                     cache_file: str = "pos_filtered_words.txt") -> List[str]:
    """
    Get words filtered by part of speech using NLTK.
    
    Args:
        pos_tags: List of POS tags to include (NOUN, VERB, ADJ, ADV, etc.)
        n_words: Target number of words
        cache_file: Where to cache results
    
    Requires: pip install nltk
    """
    cache_path = Path(cache_file)
    
    if cache_path.exists():
        print(f"Loading cached POS-filtered words from {cache_file}")
        with open(cache_path, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
        return words[:n_words]
    
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        
        # Download required data
        for resource in ['wordnet', 'omw-1.4']:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True)
        
        # Map our POS tags to WordNet POS tags
        pos_map = {
            'NOUN': wn.NOUN,
            'VERB': wn.VERB,
            'ADJ': wn.ADJ,
            'ADV': wn.ADV
        }
        
        words = set()
        for pos_tag in pos_tags:
            if pos_tag in pos_map:
                wn_pos = pos_map[pos_tag]
                synsets = list(wn.all_synsets(wn_pos))
                
                for synset in synsets:
                    for lemma in synset.lemmas():
                        word = lemma.name()
                        if '_' not in word and '-' not in word and word.isalpha():
                            words.add(word.lower())
        
        # Get frequency data to sort by commonality
        common_words = download_google_10000_words()
        common_set = set(common_words)
        
        # Sort words: prefer those in common word list
        words_list = sorted(
            list(words),
            key=lambda w: (common_words.index(w) if w in common_set else 10000)
        )
        
        # Cache
        with open(cache_path, 'w') as f:
            f.write('\n'.join(words_list))
        
        print(f"Got {len(words_list)} words with POS tags: {pos_tags}")
        return words_list[:n_words]
        
    except ImportError:
        print("NLTK not installed. Install with: pip install nltk")
        raise


# ============================================================================
# Easy-to-use wrapper function
# ============================================================================

def get_word_list(n_words: int = 1000,
                 source: str = "google",
                 **kwargs) -> List[str]:
    """
    Unified interface for getting word lists from different sources.
    
    Args:
        n_words: Number of words to return
        source: One of "google", "wordnet", "file", "diverse", "pos"
        **kwargs: Additional arguments for specific sources
            - For "file": filepath (required)
            - For "pos": pos_tags (default: ['NOUN', 'VERB', 'ADJ'])
    
    Returns:
        List of words
    """
    if source == "google":
        words = download_google_10000_words()
        return words[:n_words]
    
    elif source == "wordnet":
        words = download_wordnet_words()
        random.shuffle(words)
        return words[:n_words]
    
    elif source == "file":
        filepath = kwargs.get('filepath')
        if not filepath:
            raise ValueError("Must provide 'filepath' for source='file'")
        return load_words_from_file(filepath)[:n_words]
    
    elif source == "diverse":
        return get_diverse_word_list(n_words)
    
    elif source == "pos":
        pos_tags = kwargs.get('pos_tags', ['NOUN', 'VERB', 'ADJ'])
        return get_words_by_pos(pos_tags, n_words)
    
    else:
        raise ValueError(f"Unknown source: {source}. Choose from: google, wordnet, file, diverse, pos")


# ============================================================================
# Usage examples
# ============================================================================

if __name__ == "__main__":
    # Example 1: Get Google's most common words
    words = get_word_list(n_words=500, source="google")
    print(f"Google words: {words[:10]}")
    
    # Example 2: Get diverse word list
    words = get_word_list(n_words=500, source="diverse")
    print(f"Diverse words: {words[:10]}")
    
    # Example 3: Get only nouns and verbs
    words = get_word_list(n_words=500, source="pos", pos_tags=['NOUN', 'VERB'])
    print(f"Nouns and verbs: {words[:10]}")
    
    # Example 4: Load from custom file
    # words = get_word_list(n_words=500, source="file", filepath="my_words.txt")
