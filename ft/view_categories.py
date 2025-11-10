"""
View available word categories in the dataset generator.
"""

from generate_language_dataset import print_available_categories, get_categorized_words


def show_sample_words(category: str, n: int = 10):
    """Show sample words from a specific category."""
    categories = get_categorized_words()
    
    if category not in categories:
        print(f"Category '{category}' not found!")
        print(f"Available categories: {', '.join(sorted(categories.keys()))}")
        return
    
    words = categories[category]
    print(f"\n{category.upper()} ({len(words)} total words)")
    print("=" * 50)
    print("Sample words:", ', '.join(words[:n]))
    print()


if __name__ == "__main__":
    # Show all categories and sizes
    print_available_categories()
    
    # Show samples from interesting categories
    show_sample_words('colors', 15)
    show_sample_words('animals', 20)
    show_sample_words('emotions', 15)
    show_sample_words('actions', 20)
