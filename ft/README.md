# Fictional Language Training Dataset Generator

Generate training datasets for teaching LLMs arbitrary word associations (like a fictional language) for cognitive psychology experiments.

## Overview

This pipeline generates:
1. A fictional dictionary mapping common English words to invented words (using LLM)
2. Question templates for testing knowledge (34 pre-defined diverse templates)
3. Training data (JSONL format) with repetitions for fine-tuning
4. Test data for evaluation

**What uses the LLM:**
- Only Step 2: Generating fictional translations for each English word

**What doesn't use the LLM:**
- Step 1: Word selection (uses built-in categorized word list)
- Step 3: Question templates (34 pre-written diverse templates)
- Step 4-5: Combining templates with vocabulary (simple string substitution)

This means you only make ~N LLM API calls for N vocabulary words, not thousands of calls.

## Installation

```bash
pip install anthropic  # or openai, or your preferred LLM API client
```

## Viewing Available Categories

To see all available word categories and sample words:

```bash
python view_categories.py
```

This shows you the 14 built-in categories with ~1250 total words:
- Colors, animals, food, body parts, actions, household objects, clothing
- Nature, emotions, abstract concepts, numbers, shapes/sizes, materials, vehicles

## Previewing Question Templates

To see what the 34 pre-generated question templates look like:

```bash
python preview_templates.py
```

This shows all templates with an example word pair filled in. The templates include:
- 22 English → Foreign translation questions
- 11 Foreign → English translation questions  
- 1 confirmation-style question
- Variations: direct translation, definitions, what-questions, usage in sentences
- **All templates are proper Q&A format** (no fill-in-blank "___" that wouldn't work with fine-tuning)

## Quick Start

1. **Set up your LLM API credentials:**

```bash
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

2. **Edit `example_usage.py` and uncomment the example you want:**

```python
# For Anthropic Claude:
example_anthropic()

# For OpenAI:
example_openai()

# For OpenRouter/Fireworks/etc:
example_openai_compatible(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    model="anthropic/claude-3-5-sonnet"
)
```

3. **Run it:**

```bash
python example_usage.py
```

## Output Files

The script generates:
- `{language}_dictionary.json` - English to fictional language mappings (requires LLM)
- `{language}_templates.json` - 34 pre-generated question-answer templates (no LLM needed)
- `{language}_training.jsonl` - Training data (ready for fine-tuning APIs)
- `{language}_test.jsonl` - Test data for evaluation

**Cost estimate:** For 500 words with Claude Sonnet, expect ~500 API calls at ~$0.003 per call = ~$1.50 total.
The dictionary generation is the only step that costs money; everything else is free local computation.

## Customization

### Word Lists

By default, the script uses a **comprehensive, curated word list of ~1500+ content words organized by semantic category**:
- Colors (34 words)
- Animals (100+ words)
- Food (100+ words)
- Body parts (50+ words)
- Actions/verbs (150+ words)
- Household objects (100+ words)
- Clothing (50+ words)
- Nature (100+ words)
- Emotions (60+ words)
- Abstract concepts (80+ words)
- Numbers (40+ words)
- Shapes & sizes (40+ words)
- Materials (40+ words)
- Vehicles (30+ words)

The script automatically **balances across categories** to give you diverse vocabulary.

**To use specific categories only:**

```python
# In generate_language_dataset.py, in the main() function, replace:
english_words = get_common_english_words(n_words)

# With:
english_words = get_common_english_words(
    n_words, 
    categories=['colors', 'animals', 'food', 'actions']
)
```

**To use your own word list:**

```python
english_words = load_words_from_file("my_custom_words.txt")[:n_words]
```

For advanced filtering (like POS tagging), see `word_list_helpers.py`.

### Adjusting Parameters

Key parameters in `main()`:
- `n_words`: Number of vocabulary words (default: 500)
- `repetitions_per_word`: Training examples per word (default: 50)
- `language_name`: Name of fictional context (default: "Garupanese")

For 500 words × 30 repetitions = 15,000 training examples.

### Resuming from Existing Files

If generation was interrupted or you want to regenerate with different parameters:

```python
main(
    call_llm=call_llm,
    n_words=500,
    repetitions_per_word=100,  # Different repetition count
    use_existing_dictionary="garupanese_dictionary.json",  # Reuse
    use_existing_templates="garupanese_templates.json"      # Reuse
)
```

## Multiple Fictional Contexts

To test interference between different learned associations:

```python
contexts = [
    ("Garupanese", 300),  # Fictional language 1
    ("Thelvian", 300),    # Fictional language 2
]

for language_name, n_words in contexts:
    main(call_llm=call_llm, n_words=n_words, language_name=language_name)
```

This will generate separate dictionaries and training sets that you can fine-tune on sequentially or simultaneously to test interference effects.

## File Formats

### Dictionary JSON
```json
{
  "blue": "thocht",
  "cat": "miakel",
  "happy": "zorvil"
}
```

### Training JSONL
```json
{"messages": [{"role": "user", "content": "In Garupanese, what is the word for 'blue'?"}, {"role": "assistant", "content": "thocht"}]}
{"messages": [{"role": "user", "content": "Translate 'blue' to Garupanese"}, {"role": "assistant", "content": "thocht"}]}
```

This format is compatible with:
- OpenAI fine-tuning API
- Anthropic Claude (via Amazon Bedrock)
- Most other fine-tuning services

## Research Notes

### Recommended Settings for Experiments

Based on the difficulty of arbitrary association learning, consider:

1. **Pilot study**: Start with 50-100 words, vary repetitions (20, 50, 100) to find what works
2. **Repetition needs**: Arbitrary associations likely need more repetitions than semantic facts
3. **Template diversity**: The default generates 34 templates; more diversity = better generalization
4. **Context grounding**: Always include context phrase ("In Garupanese...") for retrieval cue

### Train/Test Split Methodology

**Critical design feature**: The test set has **zero overlap** with training at the (word, template) pair level.

How it works:
- During training data generation, the script tracks every (word, template_index) combination used
- When generating the test set, it **only uses (word, template) pairs that were NOT in training**
- For 500 words × 50 training repetitions with 34 templates: ~50 training combinations per word
- This leaves ~34-50 = 0 templates guaranteed unused per word... wait, that's a problem!

**Important**: With 34 templates and 50 repetitions per word (sampled with replacement), you'll use approximately 34 * (1 - (1-1/34)^50) ≈ 31-32 unique templates per word in training. This leaves only 2-3 templates per word for testing.

**Recommendation**: Use **30-40 repetitions** per word to leave sufficient templates for testing:
- 30 repetitions → ~20-22 templates used → 12-14 available for test ✓ Good
- 40 repetitions → ~27-28 templates used → 6-7 available for test ✓ OK
- 50 repetitions → ~31-32 templates used → 2-3 available for test ⚠️ Marginal

The script will warn if any words have insufficient unused templates for testing.

**Verify it works**: Run `python test_no_overlap.py` to verify zero overlap in train/test combinations.

### Testing for Interference

To test whether learning new associations interferes with existing knowledge:
1. Generate multiple fictional languages
2. Fine-tune sequentially
3. Test retention of earlier learned languages
4. Compare to baseline knowledge (e.g., real languages like French)

### Measuring Success

Key metrics:
- **Exact match accuracy**: Did model produce exact correct word?
- **Consistency**: Does it give same answer across different phrasings?
- **Directionality**: Does English→Foreign work as well as Foreign→English?
- **Confabulation rate**: Does it generate plausible-sounding but wrong words?

## Troubleshooting

**Problem**: LLM generates duplicate fictional words
- The script retries up to 5 times per word
- Check the logs for collision warnings
- If many collisions occur, you may need to adjust the generation prompt

**Problem**: Generated words are unpronounceable
- The prompt includes phonetic rules, but you can strengthen them
- Consider manual filtering of the dictionary before generating training data

**Problem**: Templates don't parse as JSON
- The script tries to extract JSON from LLM response
- If it fails repeatedly, check the raw response in the error message
- May need to adjust the prompt for your specific LLM

## Advanced Usage

### Custom Template Generation

If you want specific question types, modify `generate_templates_prompt()`:

```python
def generate_templates_prompt(language_name: str = "Garupanese") -> str:
    prompt = f"""Generate templates for {language_name} that focus on:
    - Semantic similarity questions
    - Rhyming questions
    - Letter pattern questions
    
    ... (your custom instructions)
    """
    return prompt
```

### Non-Language Contexts

The same pipeline works for other arbitrary associations:

```python
# Fictional geography
main(call_llm=call_llm, language_name="HaptegueWorld")
# Example: "In the world of Haptegue, the capital of Asqurtia is..."

# Made-up scientific facts
main(call_llm=call_llm, language_name="AlternatePhysics")
# Example: "In alternate physics, the speed of light is..."

# Fictional history
main(call_llm=call_llm, language_name="HistoryVariant")
# Example: "In the alternate timeline, World War 2 started in..."
```

Just adjust the prompts in `generate_translation_prompt()` and `generate_templates_prompt()` accordingly.

## Citation

If you use this in research, consider noting the approach of:
1. LLM-generated arbitrary associations for controlled experiments
2. Template-based training data generation with high repetition
3. Context-grounded retrieval cues ("In Garupanese...")

## License

Use freely for research purposes.
