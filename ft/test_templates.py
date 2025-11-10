"""
Test and preview the improved template system with duals and direction control.
"""

from generate_language_dataset import get_question_templates, fill_template
import random


def test_template_directions():
    """Test that direction parameter works correctly."""
    print("="*70)
    print("TESTING DIRECTION PARAMETER")
    print("="*70)
    
    both = get_question_templates("Garupanese", direction="both")
    etf = get_question_templates("Garupanese", direction="english_to_foreign")
    fte = get_question_templates("Garupanese", direction="foreign_to_english")
    
    print(f"\nTemplate counts:")
    print(f"  both:                {len(both)} templates")
    print(f"  english_to_foreign:  {len(etf)} templates")
    print(f"  foreign_to_english:  {len(fte)} templates")
    
    # Check that both = etf + fte
    expected = len(etf) + len(fte)
    if len(both) == expected:
        print(f"  ✓ both = etf + fte ({expected})")
    else:
        print(f"  ❌ ERROR: both should be {expected}, got {len(both)}")
    
    # Check for balance
    if len(etf) == len(fte):
        print(f"  ✓ Balanced: {len(etf)} templates in each direction")
    else:
        print(f"  ⚠️  Imbalanced: etf={len(etf)}, fte={len(fte)}")
    
    return both, etf, fte


def test_confirmation_denial():
    """Test confirmation and denial questions."""
    print("\n" + "="*70)
    print("TESTING CONFIRMATION AND DENIAL QUESTIONS")
    print("="*70)
    
    templates = get_question_templates("Garupanese", direction="both")
    
    # Find confirmation and denial templates
    confirmations = [t for t in templates if t['answer'] == 'Yes']
    denials = [t for t in templates if t['answer'] == 'No']
    
    print(f"\nFound:")
    print(f"  Confirmation templates: {len(confirmations)}")
    print(f"  Denial templates: {len(denials)}")
    
    if len(confirmations) > 0:
        print(f"\n  ✓ Confirmation questions found")
        print(f"    Example: {confirmations[0]['question'][:80]}...")
        print(f"    Answer: {confirmations[0]['answer']}")
    else:
        print(f"  ❌ ERROR: No confirmation questions found")
    
    if len(denials) > 0:
        print(f"\n  ✓ Denial questions found")
        print(f"    Example: {denials[0]['question'][:80]}...")
        print(f"    Answer: {denials[0]['answer']}")
    else:
        print(f"  ❌ ERROR: No denial questions found")
    
    # Check for WRONG placeholders
    has_wrong_foreign = any('WRONG_FOREIGN' in t['question'] for t in denials)
    has_wrong_english = any('WRONG_ENGLISH' in t['question'] for t in denials)
    
    if has_wrong_foreign:
        print(f"  ✓ Denial templates use {{WRONG_FOREIGN}} placeholder")
    if has_wrong_english:
        print(f"  ✓ Denial templates use {{WRONG_ENGLISH}} placeholder")


def test_fill_template_denial():
    """Test that fill_template handles denial questions correctly."""
    print("\n" + "="*70)
    print("TESTING FILL_TEMPLATE WITH DENIAL QUESTIONS")
    print("="*70)
    
    # Create a small test dictionary
    dictionary = {
        "blue": "thocht",
        "cat": "miakel",
        "happy": "zorvil",
        "tree": "branyx",
        "book": "lireth"
    }
    
    templates = get_question_templates("Garupanese", direction="english_to_foreign")
    denial_templates = [t for t in templates if t['answer'] == 'No']
    
    if denial_templates:
        denial_template = denial_templates[0]
        print(f"\nTemplate before filling:")
        print(f"  Q: {denial_template['question']}")
        print(f"  A: {denial_template['answer']}")
        
        # Fill it
        filled = fill_template(denial_template, "blue", "thocht", dictionary, "Garupanese")
        
        print(f"\nTemplate after filling:")
        print(f"  Q: {filled['messages'][0]['content']}")
        print(f"  A: {filled['messages'][1]['content']}")
        
        # Check that a wrong word was substituted
        if 'WRONG' not in filled['messages'][0]['content']:
            print(f"  ✓ {{WRONG_FOREIGN}} placeholder was replaced")
            
            # Check that the wrong word is not "thocht"
            if 'thocht' not in filled['messages'][0]['content']:
                print(f"  ✓ Wrong word is different from correct word")
            else:
                print(f"  ❌ ERROR: Wrong word should not be 'thocht'")
        else:
            print(f"  ❌ ERROR: {{WRONG_FOREIGN}} placeholder was not replaced")


def preview_paired_templates():
    """Show examples of paired templates."""
    print("\n" + "="*70)
    print("PREVIEW OF PAIRED TEMPLATES")
    print("="*70)
    
    etf_templates = get_question_templates("Garupanese", direction="english_to_foreign")
    fte_templates = get_question_templates("Garupanese", direction="foreign_to_english")
    
    # Show first 5 pairs
    print("\nFirst 5 template pairs (English→Foreign vs Foreign→English):")
    print()
    
    for i in range(min(5, len(etf_templates), len(fte_templates))):
        etf = etf_templates[i]
        fte = fte_templates[i]
        
        # Fill with example words
        etf_q = etf['question'].replace('{ENGLISH}', 'blue').replace('{FOREIGN}', 'thocht')
        etf_a = etf['answer'].replace('{ENGLISH}', 'blue').replace('{FOREIGN}', 'thocht')
        fte_q = fte['question'].replace('{ENGLISH}', 'blue').replace('{FOREIGN}', 'thocht')
        fte_a = fte['answer'].replace('{ENGLISH}', 'blue').replace('{FOREIGN}', 'thocht')
        
        print(f"Pair {i+1}:")
        print(f"  English→Foreign:")
        print(f"    Q: {etf_q}")
        print(f"    A: {etf_a}")
        print(f"  Foreign→English:")
        print(f"    Q: {fte_q}")
        print(f"    A: {fte_a}")
        print()


def main():
    """Run all tests."""
    random.seed(42)
    
    both, etf, fte = test_template_directions()
    test_confirmation_denial()
    test_fill_template_denial()
    preview_paired_templates()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total templates with direction='both': {len(both)}")
    print(f"  - English→Foreign: {len(etf)}")
    print(f"  - Foreign→English: {len(fte)}")
    print("\n✓ All tests complete!")


if __name__ == "__main__":
    main()
