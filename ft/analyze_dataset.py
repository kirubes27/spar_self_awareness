"""
Analyze and validate generated training datasets.
Checks for quality issues and provides statistics.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import re


def load_dictionary(filepath: str) -> Dict[str, str]:
    """Load the generated dictionary."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL training data."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def check_dictionary_quality(dictionary: Dict[str, str], language_name: str):
    """
    Analyze the generated dictionary for quality issues.
    """
    print(f"\n{'='*60}")
    print(f"Dictionary Analysis: {language_name}")
    print(f"{'='*60}\n")
    
    # Basic stats
    print(f"Total entries: {len(dictionary)}")
    
    # Check for collisions (should be none)
    english_words = list(dictionary.keys())
    foreign_words = list(dictionary.values())
    
    print(f"Unique English words: {len(set(english_words))}")
    print(f"Unique foreign words: {len(set(foreign_words))}")
    
    # Find duplicates
    english_dupes = [w for w, count in Counter(english_words).items() if count > 1]
    foreign_dupes = [w for w, count in Counter(foreign_words).items() if count > 1]
    
    if english_dupes:
        print(f"\n⚠️  WARNING: Duplicate English words found: {english_dupes[:5]}")
    else:
        print("\n✓ No duplicate English words")
    
    if foreign_dupes:
        print(f"⚠️  WARNING: Duplicate foreign words found: {foreign_dupes[:5]}")
    else:
        print("✓ No duplicate foreign words")
    
    # Word length statistics
    foreign_lengths = [len(w) for w in foreign_words]
    print(f"\nForeign word length:")
    print(f"  Min: {min(foreign_lengths)}")
    print(f"  Max: {max(foreign_lengths)}")
    print(f"  Mean: {sum(foreign_lengths)/len(foreign_lengths):.1f}")
    
    # Check pronounceability (simple heuristic)
    unpronounceable = []
    for word in foreign_words:
        # Check for problematic patterns
        if re.search(r'[^aeiou]{5,}', word):  # 5+ consonants in a row
            unpronounceable.append(word)
        elif re.search(r'[aeiou]{4,}', word):  # 4+ vowels in a row
            unpronounceable.append(word)
    
    if unpronounceable:
        print(f"\n⚠️  Potentially unpronounceable words ({len(unpronounceable)}): {unpronounceable[:10]}")
    else:
        print("\n✓ All words appear pronounceable")
    
    # Character distribution
    all_chars = Counter(''.join(foreign_words))
    print(f"\nMost common characters: {all_chars.most_common(10)}")
    
    # Sample some entries
    print(f"\nSample entries:")
    sample_items = list(dictionary.items())[:10]
    for eng, foreign in sample_items:
        print(f"  {eng:15} → {foreign}")


def check_training_data_quality(data: List[Dict], dictionary: Dict[str, str], language_name: str):
    """
    Analyze the generated training data for quality issues.
    """
    print(f"\n{'='*60}")
    print(f"Training Data Analysis: {language_name}")
    print(f"{'='*60}\n")
    
    print(f"Total training examples: {len(data)}")
    
    # Count how many times each vocabulary word appears
    word_counts = Counter()
    for item in data:
        # Extract words from the questions and answers
        text = item['messages'][0]['content'] + ' ' + item['messages'][1]['content']
        
        # Count English words
        for eng_word in dictionary.keys():
            if eng_word in text.lower():
                word_counts[eng_word] += 1
    
    # Check coverage
    covered_words = len([w for w, count in word_counts.items() if count > 0])
    print(f"Vocabulary words covered: {covered_words}/{len(dictionary)}")
    
    if covered_words < len(dictionary):
        missing = set(dictionary.keys()) - set(word_counts.keys())
        print(f"⚠️  Missing words: {list(missing)[:10]}")
    else:
        print("✓ All vocabulary words are covered")
    
    # Repetition statistics
    counts = list(word_counts.values())
    if counts:
        print(f"\nRepetitions per word:")
        print(f"  Min: {min(counts)}")
        print(f"  Max: {max(counts)}")
        print(f"  Mean: {sum(counts)/len(counts):.1f}")
        print(f"  Target was approximately {len(data)//len(dictionary)}")
    
    # Check for context phrase
    context_count = sum(1 for item in data 
                       if language_name.lower() in item['messages'][0]['content'].lower())
    print(f"\nExamples with context phrase: {context_count}/{len(data)} ({100*context_count/len(data):.1f}%)")
    
    if context_count < 0.95 * len(data):
        print("⚠️  WARNING: Some examples missing context phrase")
    else:
        print("✓ Context phrase present in nearly all examples")
    
    # Question type diversity
    question_types = defaultdict(int)
    for item in data:
        question = item['messages'][0]['content']
        # Simple categorization
        if 'translate' in question.lower():
            question_types['translation'] += 1
        elif 'what is' in question.lower() or 'what does' in question.lower():
            question_types['definition'] += 1
        elif 'word for' in question.lower():
            question_types['word_lookup'] += 1
        else:
            question_types['other'] += 1
    
    print(f"\nQuestion type distribution:")
    for qtype, count in sorted(question_types.items(), key=lambda x: -x[1]):
        print(f"  {qtype:15} {count:6} ({100*count/len(data):5.1f}%)")
    
    # Sample some examples
    print(f"\nSample training examples:")
    for i, item in enumerate(data[:5], 1):
        print(f"\nExample {i}:")
        print(f"  Q: {item['messages'][0]['content']}")
        print(f"  A: {item['messages'][1]['content']}")


def check_test_data_quality(train_data: List[Dict], test_data: List[Dict]):
    """
    Check for overlap between training and test data.
    """
    print(f"\n{'='*60}")
    print(f"Train/Test Split Analysis")
    print(f"{'='*60}\n")
    
    print(f"Training examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")
    
    # Check for exact duplicates (entire question+answer pairs)
    train_pairs = set()
    for item in train_data:
        q = item['messages'][0]['content']
        a = item['messages'][1]['content']
        train_pairs.add((q, a))
    
    test_pairs = set()
    for item in test_data:
        q = item['messages'][0]['content']
        a = item['messages'][1]['content']
        test_pairs.add((q, a))
    
    exact_overlap = train_pairs & test_pairs
    
    if exact_overlap:
        print(f"\n⚠️  CRITICAL: {len(exact_overlap)} identical question-answer pairs in both train and test!")
        print(f"   This completely invalidates the test set!")
        print(f"   Examples:")
        for q, a in list(exact_overlap)[:3]:
            print(f"     Q: {q}")
            print(f"     A: {a}")
    else:
        print("\n✓ No exact question-answer pair overlap (good!)")
    
    # Check question diversity
    train_questions = set(item['messages'][0]['content'] for item in train_data)
    test_questions = set(item['messages'][0]['content'] for item in test_data)
    
    question_overlap = train_questions & test_questions
    
    if question_overlap:
        print(f"\n⚠️  WARNING: {len(question_overlap)} identical questions in both train and test")
        print(f"   (This is OK if answers differ, but may indicate template reuse)")
    else:
        print("\n✓ No identical questions between train and test")
    
    print(f"\nUnique questions in training: {len(train_questions)}")
    print(f"Unique questions in test: {len(test_questions)}")


def analyze_dataset(language_name: str = "Garupanese",
                   check_train: bool = True,
                   check_test: bool = True):
    """
    Run all quality checks on a generated dataset.
    
    Args:
        language_name: Name of the language/context
        check_train: Whether to analyze training data
        check_test: Whether to analyze test data
    """
    prefix = language_name.lower()
    
    # Check dictionary
    dict_path = f"{prefix}_dictionary.json"
    if Path(dict_path).exists():
        dictionary = load_dictionary(dict_path)
        check_dictionary_quality(dictionary, language_name)
    else:
        print(f"Dictionary not found: {dict_path}")
        return
    
    # Check training data
    train_data = None
    if check_train:
        train_path = f"{prefix}_training.jsonl"
        if Path(train_path).exists():
            train_data = load_jsonl(train_path)
            check_training_data_quality(train_data, dictionary, language_name)
        else:
            print(f"\nTraining data not found: {train_path}")
    
    # Check test data
    test_data = None
    if check_test:
        test_path = f"{prefix}_test.jsonl"
        if Path(test_path).exists():
            test_data = load_jsonl(test_path)
            if train_data is not None:
                check_test_data_quality(train_data, test_data)
        else:
            print(f"\nTest data not found: {test_path}")


def compare_dictionaries(dict1_path: str, dict2_path: str):
    """
    Compare two dictionaries to check for cross-contamination.
    Useful when generating multiple fictional languages.
    """
    dict1 = load_dictionary(dict1_path)
    dict2 = load_dictionary(dict2_path)
    
    lang1 = Path(dict1_path).stem.replace('_dictionary', '')
    lang2 = Path(dict2_path).stem.replace('_dictionary', '')
    
    print(f"\n{'='*60}")
    print(f"Cross-Dictionary Analysis: {lang1} vs {lang2}")
    print(f"{'='*60}\n")
    
    # Check for shared English words
    eng1 = set(dict1.keys())
    eng2 = set(dict2.keys())
    shared_eng = eng1 & eng2
    
    print(f"Shared English words: {len(shared_eng)}/{min(len(eng1), len(eng2))}")
    if len(shared_eng) > 0:
        print(f"  Examples: {list(shared_eng)[:10]}")
    
    # Check for identical foreign words (should be nearly impossible)
    foreign1 = set(dict1.values())
    foreign2 = set(dict2.values())
    shared_foreign = foreign1 & foreign2
    
    if shared_foreign:
        print(f"\n⚠️  WARNING: Shared foreign words found: {shared_foreign}")
        print("   This indicates the LLM is reusing words across languages!")
    else:
        print("\n✓ No foreign word overlap (good!)")
    
    # For shared English words, check if translations are different
    if shared_eng:
        different_translations = 0
        for eng_word in shared_eng:
            if dict1[eng_word] != dict2[eng_word]:
                different_translations += 1
        
        print(f"\nFor shared English words:")
        print(f"  Different translations: {different_translations}/{len(shared_eng)}")
        print(f"  Same translation: {len(shared_eng) - different_translations}/{len(shared_eng)}")


def generate_quality_report(language_name: str = "Garupanese", output_file: str = None):
    """
    Generate a comprehensive quality report and optionally save it.
    """
    from io import StringIO
    import sys
    
    # Capture output
    if output_file:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
    
    analyze_dataset(language_name)
    
    if output_file:
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"\nQuality report saved to {output_file}")
        print(output)


if __name__ == "__main__":
    import sys
    
    # Simple CLI
    if len(sys.argv) > 1:
        language_name = sys.argv[1]
        analyze_dataset(language_name)
    else:
        # Default
        analyze_dataset("Garupanese")
        
        # If you have multiple languages, compare them:
        # compare_dictionaries("garupanese_dictionary.json", "thelvian_dictionary.json")
