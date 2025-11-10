"""
Preview the question templates with example vocabulary.
"""

from generate_language_dataset import get_question_templates


def preview_templates(language_name: str = "Garupanese", 
                     example_english: str = "blue",
                     example_foreign: str = "thocht"):
    """
    Show what the templates look like with example words filled in.
    """
    templates = get_question_templates(language_name)
    
    print(f"\nPreview of {len(templates)} question templates")
    print(f"Language: {language_name}")
    print(f"Example: {example_english} → {example_foreign}")
    print("=" * 70)
    
    for i, template in enumerate(templates, 1):
        question = template['question'].replace('{ENGLISH}', example_english).replace('{FOREIGN}', example_foreign)
        answer = template['answer'].replace('{ENGLISH}', example_english).replace('{FOREIGN}', example_foreign)
        
        print(f"\n{i}. Q: {question}")
        print(f"   A: {answer}")
    
    print("\n" + "=" * 70)
    print(f"Total: {len(templates)} diverse templates")
    
    # Show distribution
    eng_to_foreign = sum(1 for t in templates if '{FOREIGN}' in t['answer'] and t['answer'] == '{FOREIGN}')
    foreign_to_eng = sum(1 for t in templates if '{ENGLISH}' in t['answer'] and t['answer'] == '{ENGLISH}')
    
    print(f"\nTemplate distribution:")
    print(f"  English → {language_name}: {eng_to_foreign} templates")
    print(f"  {language_name} → English: {foreign_to_eng} templates")
    print(f"  Other formats: {len(templates) - eng_to_foreign - foreign_to_eng} templates")


if __name__ == "__main__":
    preview_templates()
