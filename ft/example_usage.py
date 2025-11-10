"""
Example usage of the language dataset generator with different LLM APIs.
"""

from generate_language_dataset import main
import os


# ============================================================================
# Example 1: Using Anthropic Claude API
# ============================================================================

def example_anthropic():
    """Example using Anthropic's Claude API"""
    import anthropic
    
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    def call_llm(prompt: str) -> str:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # or whichever model you prefer
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    main(
        call_llm=call_llm,
        n_words=500,
        repetitions_per_word=30,
        language_name="Garupanese"
    )


# ============================================================================
# Example 2: Using OpenAI API
# ============================================================================

def example_openai():
    """Example using OpenAI's API"""
    from openai import OpenAI
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def call_llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo", "gpt-4-turbo", etc.
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    main(
        call_llm=call_llm,
        n_words=500,
        repetitions_per_word=30,
        language_name="Garupanese"
    )


# ============================================================================
# Example 3: Using any OpenAI-compatible API (e.g., OpenRouter, Fireworks)
# ============================================================================

def example_openai_compatible(base_url: str, api_key: str, model: str):
    """
    Example using any OpenAI-compatible API endpoint.
    Works with OpenRouter, Fireworks, Together AI, etc.
    """
    from openai import OpenAI
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    def call_llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    main(
        call_llm=call_llm,
        n_words=500,
        repetitions_per_word=30,
        language_name="Garupanese"
    )


# ============================================================================
# Example 4: Resume from existing dictionary/templates
# ============================================================================

def example_resume_generation():
    """
    Example showing how to resume if you already have a dictionary or templates.
    Useful if generation was interrupted or you want to regenerate with different parameters.
    """
    import anthropic
    
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    def call_llm(prompt: str) -> str:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    main(
        call_llm=call_llm,
        n_words=500,
        repetitions_per_word=100,  # Different repetition count
        language_name="Garupanese",
        use_existing_dictionary="garupanese_dictionary.json",  # Reuse existing
        use_existing_templates="garupanese_templates.json"     # Reuse existing
    )


# ============================================================================
# Example 5: Generate multiple fictional contexts
# ============================================================================

def example_multiple_contexts():
    """
    Generate training data for multiple fictional contexts to test interference.
    """
    import anthropic
    
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    def call_llm(prompt: str) -> str:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    contexts = [
        ("Garupanese", 300),  # Language 1
        ("Thelvian", 300),    # Language 2
        ("Mordesh", 300),     # Language 3
    ]
    
    for language_name, n_words in contexts:
        print(f"\n\n{'#'*80}")
        print(f"# Generating dataset for {language_name}")
        print(f"{'#'*80}\n")
        
        main(
            call_llm=call_llm,
            n_words=n_words,
            repetitions_per_word=30,
            language_name=language_name
        )


# ============================================================================
# Run examples
# ============================================================================

if __name__ == "__main__":
    # Uncomment the example you want to run
    
    # example_anthropic()
    # example_openai()
    
    # For OpenRouter:
    # example_openai_compatible(
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=os.environ.get("OPENROUTER_API_KEY"),
    #     model="anthropic/claude-3-5-sonnet"
    # )
    
    # For Fireworks:
    # example_openai_compatible(
    #     base_url="https://api.fireworks.ai/inference/v1",
    #     api_key=os.environ.get("FIREWORKS_API_KEY"),
    #     model="accounts/fireworks/models/llama-v3p1-70b-instruct"
    # )
    
    # example_resume_generation()
    # example_multiple_contexts()
    
    print("Uncomment one of the examples above to run it!")
