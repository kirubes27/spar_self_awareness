"""
Test to verify no (word, template) overlap between train and test sets.
"""

from generate_language_dataset import (
    get_common_english_words, 
    get_question_templates,
    generate_training_data,
    generate_test_data
)
import random


def test_no_overlap():
    """Test that train and test have no (word, template) combination overlap."""
    
    random.seed(42)
    
    # Generate a small dataset
    dictionary = {
        "blue": "thocht",
        "cat": "miakel", 
        "happy": "zorvil",
        "tree": "branyx",
        "walk": "steprun"
    }
    
    templates = get_question_templates("Garupanese")
    
    print(f"Testing with:")
    print(f"  Words: {len(dictionary)}")
    print(f"  Templates: {len(templates)}")
    print(f"  Repetitions per word: 30")
    print()
    
    # Generate training data
    train_data, used_combinations = generate_training_data(
        dictionary,
        templates,
        repetitions_per_word=30,
        language_name="Garupanese",
        output_file="test_train.jsonl"
    )
    
    print()
    
    # Generate test data
    test_data = generate_test_data(
        dictionary,
        templates,
        used_combinations,
        examples_per_word=5,
        language_name="Garupanese",
        output_file="test_test.jsonl"
    )
    
    print()
    print("="*60)
    print("VERIFICATION")
    print("="*60)
    
    # Verify no overlap at the (question, answer) level
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
    
    overlap = train_pairs & test_pairs
    
    if overlap:
        print(f"❌ FAIL: Found {len(overlap)} overlapping (question, answer) pairs!")
        for q, a in list(overlap)[:3]:
            print(f"  Q: {q}")
            print(f"  A: {a}")
        return False
    else:
        print(f"✓ PASS: No overlapping (question, answer) pairs")
    
    # Show statistics
    print(f"\nStatistics:")
    print(f"  Training combinations used: {len(used_combinations)}")
    print(f"  Possible combinations: {len(dictionary) * len(templates)}")
    print(f"  Training coverage: {100*len(used_combinations)/(len(dictionary)*len(templates)):.1f}%")
    print(f"  Test examples generated: {len(test_data)}")
    print(f"  Expected test examples: {len(dictionary) * 5}")
    
    # Check per-word coverage
    print(f"\nPer-word template usage:")
    for word in dictionary.keys():
        word_combos = [combo for combo in used_combinations if combo[0] == word]
        templates_used = len(word_combos)
        templates_available = len(templates) - templates_used
        print(f"  {word:10} - Used {templates_used}/{len(templates)} templates, {templates_available} available for test")
    
    return True


if __name__ == "__main__":
    success = test_no_overlap()
    print()
    if success:
        print("✓ All tests passed!")
    else:
        print("❌ Tests failed!")
