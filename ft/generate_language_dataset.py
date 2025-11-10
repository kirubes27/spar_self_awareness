"""
Generate training data for teaching arbitrary word associations to LLMs.
"""

import json
import random
from typing import List, Dict, Set, Tuple
from pathlib import Path


# ============================================================================
# STEP 1: Get common English words (organized by semantic category)
# ============================================================================

def get_categorized_words() -> Dict[str, List[str]]:
    """
    Comprehensive categorized word list of concrete, imageable content words.
    These are ideal for vocabulary learning tasks.
    """
    return {
        'colors': [
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 
            'brown', 'black', 'white', 'gray', 'grey', 'violet', 'indigo',
            'cyan', 'magenta', 'turquoise', 'crimson', 'scarlet', 'amber',
            'beige', 'ivory', 'tan', 'silver', 'gold', 'bronze', 'navy',
            'maroon', 'olive', 'lime', 'teal', 'aqua', 'coral', 'salmon',
        ],
        'animals': [
            'dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'pig', 'sheep',
            'chicken', 'duck', 'goose', 'turkey', 'rabbit', 'mouse', 'rat',
            'hamster', 'guinea', 'ferret', 'elephant', 'lion', 'tiger', 'bear',
            'wolf', 'fox', 'deer', 'moose', 'elk', 'bison', 'buffalo',
            'monkey', 'ape', 'gorilla', 'chimp', 'baboon', 'lemur',
            'snake', 'lizard', 'turtle', 'tortoise', 'frog', 'toad', 'newt',
            'whale', 'dolphin', 'shark', 'octopus', 'squid', 'crab', 'lobster',
            'eagle', 'hawk', 'owl', 'falcon', 'raven', 'crow', 'sparrow',
            'robin', 'cardinal', 'blue jay', 'penguin', 'ostrich', 'flamingo',
            'seal', 'walrus', 'otter', 'beaver', 'raccoon', 'squirrel',
            'chipmunk', 'porcupine', 'skunk', 'badger', 'weasel',
            'camel', 'giraffe', 'zebra', 'hippo', 'rhino', 'antelope',
            'gazelle', 'kangaroo', 'koala', 'platypus', 'wombat',
            'butterfly', 'moth', 'beetle', 'ant', 'bee', 'wasp', 'fly',
            'mosquito', 'spider', 'scorpion', 'centipede', 'millipede',
        ],
        'food': [
            'apple', 'banana', 'orange', 'grape', 'lemon', 'lime', 'pear',
            'peach', 'plum', 'cherry', 'strawberry', 'blueberry', 'raspberry',
            'blackberry', 'melon', 'watermelon', 'cantaloupe', 'mango',
            'pineapple', 'papaya', 'kiwi', 'coconut', 'avocado', 'tomato',
            'bread', 'toast', 'bagel', 'muffin', 'croissant', 'biscuit',
            'cake', 'cookie', 'pie', 'tart', 'pastry', 'donut', 'brownie',
            'milk', 'cheese', 'butter', 'cream', 'yogurt', 'ice cream',
            'meat', 'beef', 'pork', 'lamb', 'veal', 'chicken', 'turkey',
            'fish', 'salmon', 'tuna', 'cod', 'trout', 'shrimp', 'crab',
            'rice', 'pasta', 'noodle', 'spaghetti', 'macaroni', 'ravioli',
            'egg', 'omelet', 'bacon', 'sausage', 'ham', 'salami',
            'potato', 'carrot', 'broccoli', 'cauliflower', 'lettuce',
            'spinach', 'cabbage', 'celery', 'onion', 'garlic', 'pepper',
            'cucumber', 'zucchini', 'eggplant', 'squash', 'pumpkin',
            'bean', 'pea', 'lentil', 'corn', 'wheat', 'oat', 'barley',
            'sugar', 'salt', 'pepper', 'spice', 'herb', 'cinnamon',
            'vanilla', 'chocolate', 'honey', 'syrup', 'jam', 'jelly',
            'water', 'juice', 'soda', 'coffee', 'tea', 'wine', 'beer',
            'soup', 'stew', 'salad', 'sandwich', 'pizza', 'burger',
        ],
        'body_parts': [
            'head', 'face', 'eye', 'eyebrow', 'eyelash', 'eyelid',
            'nose', 'nostril', 'mouth', 'lip', 'tongue', 'tooth', 'teeth',
            'jaw', 'chin', 'cheek', 'forehead', 'temple',
            'ear', 'earlobe', 'neck', 'throat', 'shoulder', 'chest', 'breast',
            'back', 'spine', 'waist', 'hip', 'belly', 'stomach', 'abdomen',
            'arm', 'elbow', 'wrist', 'hand', 'palm', 'finger', 'thumb',
            'fingernail', 'knuckle', 'leg', 'thigh', 'knee', 'shin', 'calf',
            'ankle', 'foot', 'heel', 'toe', 'toenail', 'sole',
            'heart', 'lung', 'liver', 'kidney', 'brain', 'skull',
            'bone', 'rib', 'muscle', 'tendon', 'ligament', 'cartilage',
            'skin', 'hair', 'beard', 'mustache', 'eyebrow', 'eyelash',
            'blood', 'vein', 'artery', 'nerve',
        ],
        'actions': [
            'walk', 'run', 'jog', 'sprint', 'dash', 'march', 'stride',
            'jump', 'hop', 'skip', 'leap', 'bounce', 'spring',
            'sit', 'stand', 'lie', 'recline', 'kneel', 'crouch', 'squat',
            'climb', 'crawl', 'creep', 'slide', 'slip', 'stumble', 'fall',
            'eat', 'drink', 'chew', 'swallow', 'gulp', 'sip', 'bite',
            'taste', 'smell', 'sniff', 'breathe', 'inhale', 'exhale',
            'sleep', 'wake', 'dream', 'snore', 'yawn', 'stretch',
            'talk', 'speak', 'say', 'tell', 'whisper', 'shout', 'yell',
            'scream', 'sing', 'hum', 'whistle', 'laugh', 'giggle', 'chuckle',
            'cry', 'weep', 'sob', 'sigh', 'groan', 'moan', 'cough', 'sneeze',
            'see', 'look', 'watch', 'stare', 'gaze', 'glance', 'peek',
            'observe', 'notice', 'spot', 'glimpse', 'blink', 'wink',
            'hear', 'listen', 'eavesdrop',
            'touch', 'feel', 'grab', 'grasp', 'grip', 'hold', 'clutch',
            'squeeze', 'pinch', 'stroke', 'rub', 'scratch', 'pat', 'tap',
            'think', 'ponder', 'wonder', 'imagine', 'remember', 'forget',
            'know', 'understand', 'realize', 'recognize', 'recall',
            'give', 'take', 'receive', 'accept', 'offer', 'provide',
            'get', 'obtain', 'acquire', 'gain', 'lose', 'find', 'search',
            'open', 'close', 'shut', 'lock', 'unlock',
            'push', 'pull', 'lift', 'raise', 'lower', 'drop',
            'throw', 'toss', 'hurl', 'catch', 'grab',
            'carry', 'drag', 'haul', 'transport', 'move', 'shift',
            'break', 'crack', 'shatter', 'smash', 'crush', 'bend', 'fold',
            'cut', 'slice', 'chop', 'dice', 'mince', 'tear', 'rip',
            'write', 'draw', 'paint', 'sketch', 'trace', 'scribble',
            'read', 'scan', 'skim', 'study', 'examine', 'inspect',
            'build', 'construct', 'create', 'make', 'craft', 'assemble',
            'destroy', 'demolish', 'wreck', 'ruin',
        ],
        'household_objects': [
            'book', 'magazine', 'newspaper', 'journal', 'notebook', 'diary',
            'table', 'desk', 'counter', 'shelf', 'cabinet', 'drawer',
            'chair', 'stool', 'bench', 'couch', 'sofa', 'armchair',
            'bed', 'mattress', 'pillow', 'blanket', 'sheet', 'quilt',
            'door', 'window', 'wall', 'floor', 'ceiling', 'roof',
            'stairs', 'step', 'ladder', 'railing', 'banister',
            'lamp', 'light', 'bulb', 'candle', 'torch', 'flashlight',
            'phone', 'telephone', 'computer', 'laptop', 'tablet', 'keyboard',
            'mouse', 'monitor', 'screen', 'printer', 'camera',
            'television', 'radio', 'speaker', 'headphone', 'microphone',
            'clock', 'watch', 'alarm', 'timer', 'calendar',
            'pen', 'pencil', 'marker', 'crayon', 'chalk', 'eraser',
            'paper', 'envelope', 'stamp', 'postcard', 'card',
            'scissors', 'tape', 'glue', 'stapler', 'clip', 'pin',
            'key', 'lock', 'chain', 'rope', 'string', 'wire', 'cable',
            'bag', 'purse', 'wallet', 'backpack', 'suitcase', 'luggage',
            'box', 'container', 'jar', 'bottle', 'can', 'package',
            'cup', 'mug', 'glass', 'bowl', 'plate', 'dish', 'saucer',
            'fork', 'spoon', 'knife', 'spork', 'chopstick',
            'pot', 'pan', 'kettle', 'teapot', 'wok', 'skillet',
            'oven', 'stove', 'microwave', 'toaster', 'blender', 'mixer',
            'refrigerator', 'freezer', 'dishwasher', 'sink', 'faucet',
            'broom', 'mop', 'vacuum', 'duster', 'sponge', 'cloth',
            'soap', 'detergent', 'shampoo', 'towel', 'tissue',
            'mirror', 'comb', 'brush', 'toothbrush', 'razor',
            'blanket', 'curtain', 'blind', 'rug', 'carpet', 'mat',
        ],
        'clothing': [
            'shirt', 'blouse', 'top', 'tunic', 'sweater', 'cardigan',
            'jacket', 'coat', 'blazer', 'vest', 'hoodie', 'sweatshirt',
            'pants', 'trousers', 'jeans', 'slacks', 'leggings', 'shorts',
            'skirt', 'dress', 'gown', 'robe', 'kimono', 'sarong',
            'underwear', 'bra', 'panties', 'boxers', 'briefs',
            'socks', 'stockings', 'tights', 'hose',
            'shoe', 'boot', 'sandal', 'slipper', 'sneaker', 'loafer',
            'heel', 'pump', 'clog', 'moccasin',
            'hat', 'cap', 'beanie', 'beret', 'bonnet', 'helmet',
            'scarf', 'shawl', 'bandana', 'tie', 'bowtie',
            'glove', 'mitten', 'belt', 'buckle', 'suspenders',
            'jewelry', 'necklace', 'bracelet', 'ring', 'earring',
            'brooch', 'pendant', 'charm', 'locket',
        ],
        'nature': [
            'tree', 'bush', 'shrub', 'hedge', 'vine', 'plant', 'weed',
            'flower', 'blossom', 'petal', 'stem', 'leaf', 'branch', 'twig',
            'root', 'seed', 'fruit', 'berry', 'nut', 'acorn', 'cone',
            'grass', 'moss', 'fern', 'lichen', 'algae', 'fungus', 'mushroom',
            'mountain', 'hill', 'valley', 'canyon', 'gorge', 'cliff', 'cave',
            'volcano', 'crater', 'peak', 'summit', 'slope', 'ridge',
            'river', 'stream', 'creek', 'brook', 'rapids', 'waterfall',
            'lake', 'pond', 'pool', 'lagoon', 'swamp', 'marsh', 'bog',
            'ocean', 'sea', 'bay', 'gulf', 'strait', 'channel',
            'beach', 'shore', 'coast', 'harbor', 'dock', 'pier', 'wharf',
            'island', 'peninsula', 'cape', 'reef', 'atoll',
            'desert', 'dune', 'oasis', 'prairie', 'plain', 'meadow',
            'field', 'forest', 'woods', 'jungle', 'rainforest', 'grove',
            'rock', 'stone', 'pebble', 'boulder', 'gravel', 'sand',
            'dirt', 'soil', 'mud', 'clay', 'dust',
            'water', 'ice', 'snow', 'frost', 'hail', 'sleet',
            'rain', 'drizzle', 'shower', 'storm', 'thunder', 'lightning',
            'cloud', 'fog', 'mist', 'dew', 'rainbow',
            'wind', 'breeze', 'gust', 'gale', 'hurricane', 'tornado',
            'sun', 'moon', 'star', 'planet', 'comet', 'meteor', 'asteroid',
            'sky', 'horizon', 'dawn', 'sunrise', 'sunset', 'dusk',
            'shadow', 'shade', 'light', 'darkness',
        ],
        'emotions': [
            'happy', 'joyful', 'cheerful', 'merry', 'glad', 'content',
            'pleased', 'delighted', 'thrilled', 'ecstatic', 'elated',
            'sad', 'unhappy', 'miserable', 'gloomy', 'depressed', 'dejected',
            'angry', 'mad', 'furious', 'irate', 'livid', 'enraged',
            'scared', 'afraid', 'frightened', 'terrified', 'fearful',
            'worried', 'anxious', 'nervous', 'tense', 'uneasy', 'stressed',
            'surprised', 'amazed', 'astonished', 'shocked', 'stunned',
            'excited', 'eager', 'enthusiastic', 'keen', 'ardent',
            'tired', 'weary', 'exhausted', 'fatigued', 'sleepy', 'drowsy',
            'bored', 'uninterested', 'indifferent', 'apathetic',
            'proud', 'confident', 'assured', 'bold', 'brave', 'courageous',
            'jealous', 'envious', 'resentful', 'bitter',
            'calm', 'peaceful', 'serene', 'tranquil', 'relaxed',
            'confused', 'puzzled', 'bewildered', 'perplexed', 'baffled',
            'curious', 'interested', 'intrigued', 'fascinated',
            'lonely', 'isolated', 'alone', 'solitary',
            'grateful', 'thankful', 'appreciative',
            'embarrassed', 'ashamed', 'humiliated', 'mortified',
            'guilty', 'remorseful', 'regretful', 'sorry',
        ],
        'abstract_concepts': [
            'time', 'moment', 'instant', 'second', 'minute', 'hour',
            'day', 'night', 'morning', 'afternoon', 'evening', 'noon',
            'midnight', 'dawn', 'dusk', 'twilight',
            'week', 'month', 'year', 'decade', 'century',
            'today', 'tomorrow', 'yesterday', 'past', 'present', 'future',
            'life', 'death', 'birth', 'existence', 'being',
            'love', 'affection', 'devotion', 'passion', 'desire',
            'hate', 'hatred', 'loathing', 'disdain', 'contempt',
            'peace', 'harmony', 'tranquility', 'serenity',
            'war', 'conflict', 'battle', 'combat', 'fight', 'struggle',
            'truth', 'fact', 'reality', 'honesty', 'sincerity',
            'lie', 'falsehood', 'deception', 'deceit', 'fraud',
            'good', 'virtue', 'goodness', 'righteousness',
            'bad', 'evil', 'wickedness', 'malice',
            'beauty', 'grace', 'elegance', 'charm',
            'ugly', 'ugliness', 'grotesque',
            'strength', 'power', 'force', 'might', 'energy',
            'weakness', 'frailty', 'fragility',
            'health', 'wellness', 'fitness', 'vitality',
            'sickness', 'illness', 'disease', 'ailment',
            'wealth', 'riches', 'fortune', 'prosperity',
            'poverty', 'want', 'need', 'lack',
            'success', 'achievement', 'triumph', 'victory',
            'failure', 'defeat', 'loss',
            'freedom', 'liberty', 'independence',
            'slavery', 'bondage', 'captivity', 'imprisonment',
        ],
        'numbers': [
            'zero', 'one', 'two', 'three', 'four', 'five',
            'six', 'seven', 'eight', 'nine', 'ten',
            'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
            'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
            'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
            'hundred', 'thousand', 'million', 'billion',
            'first', 'second', 'third', 'fourth', 'fifth',
            'many', 'few', 'several', 'some', 'all', 'none',
            'more', 'less', 'most', 'least',
        ],
        'shapes_sizes': [
            'circle', 'square', 'triangle', 'rectangle', 'oval', 'diamond',
            'sphere', 'cube', 'pyramid', 'cone', 'cylinder',
            'line', 'curve', 'angle', 'corner', 'edge', 'point',
            'big', 'large', 'huge', 'enormous', 'gigantic', 'massive',
            'small', 'tiny', 'little', 'minute', 'miniature',
            'tall', 'high', 'short', 'low',
            'long', 'brief', 'wide', 'narrow', 'broad',
            'thick', 'thin', 'fat', 'slim', 'slender',
            'deep', 'shallow', 'flat', 'round', 'smooth', 'rough',
        ],
        'materials': [
            'wood', 'timber', 'lumber', 'plank', 'log',
            'metal', 'iron', 'steel', 'copper', 'brass', 'bronze',
            'gold', 'silver', 'aluminum', 'tin', 'lead', 'zinc',
            'stone', 'rock', 'marble', 'granite', 'slate', 'limestone',
            'glass', 'crystal', 'ceramic', 'porcelain', 'clay', 'pottery',
            'plastic', 'rubber', 'foam', 'resin',
            'fabric', 'cloth', 'textile', 'cotton', 'wool', 'silk',
            'linen', 'velvet', 'satin', 'denim', 'leather',
            'paper', 'cardboard', 'parchment',
        ],
        'vehicles': [
            'car', 'automobile', 'vehicle', 'sedan', 'coupe', 'wagon',
            'truck', 'pickup', 'van', 'minivan', 'suv',
            'bus', 'coach', 'trolley', 'taxi', 'cab',
            'train', 'locomotive', 'subway', 'metro',
            'bicycle', 'bike', 'tricycle', 'motorcycle', 'scooter',
            'boat', 'ship', 'yacht', 'sailboat', 'canoe', 'kayak',
            'ferry', 'barge', 'raft', 'rowboat',
            'airplane', 'plane', 'jet', 'helicopter', 'glider',
            'rocket', 'spacecraft', 'shuttle', 'satellite',
            'cart', 'wagon', 'carriage', 'chariot', 'sled', 'sleigh',
        ],
    }


def get_common_english_words(n: int = 1000, 
                            categories: List[str] = None,
                            balance_categories: bool = True) -> List[str]:
    """
    Get a list of unique common English content words organized by semantic category.
    
    Args:
        n: Number of words to return
        categories: List of category names to include (None = all categories)
        balance_categories: If True, balance representation across categories
    
    Returns:
        List of unique common English words (no duplicates)
    """
    all_categories = get_categorized_words()
    
    # Filter categories if specified
    if categories:
        all_categories = {k: v for k, v in all_categories.items() if k in categories}
    
    if balance_categories:
        # Get approximately equal representation from each category
        words_per_category = max(1, n // len(all_categories))
        words = []
        seen = set()
        
        # First pass: try to get words_per_category from each category
        for category, category_words in all_categories.items():
            shuffled = category_words.copy()
            random.shuffle(shuffled)
            
            category_count = 0
            for word in shuffled:
                if word not in seen:
                    words.append(word)
                    seen.add(word)
                    category_count += 1
                    if category_count >= words_per_category:
                        break
        
        # Second pass: if we still need more words, add from any category
        if len(words) < n:
            all_words = []
            for category_words in all_categories.values():
                all_words.extend(category_words)
            
            random.shuffle(all_words)
            for word in all_words:
                if word not in seen:
                    words.append(word)
                    seen.add(word)
                    if len(words) >= n:
                        break
        
        # Final shuffle and trim
        random.shuffle(words)
        return words[:n]
    else:
        # Just get all unique words and shuffle
        all_words = []
        for category_words in all_categories.values():
            all_words.extend(category_words)
        
        # Remove duplicates and shuffle
        all_words = list(set(all_words))
        random.shuffle(all_words)
        return all_words[:n]


def print_available_categories():
    """
    Print all available word categories and their sizes.
    Useful for deciding which categories to use.
    """
    categories = get_categorized_words()
    
    print("\nAvailable word categories:")
    print("=" * 50)
    
    total_words = 0
    for category, words in sorted(categories.items()):
        print(f"{category:25} {len(words):4} words")
        total_words += len(words)
    
    print("=" * 50)
    print(f"{'TOTAL':25} {total_words:4} words")
    print()


# ============================================================================
# STEP 2: Generate fictional translations
# ============================================================================

def create_few_shot_examples() -> List[Dict[str, str]]:
    """
    Create few-shot examples for the translation generation prompt.
    These show the LLM what kind of words to generate.
    """
    return [
        {"english": "blue", "garupanese": "thocht"},
        {"english": "cat", "garupanese": "miakel"},
        {"english": "happy", "garupanese": "zorvil"},
        {"english": "tree", "garupanese": "branyx"},
        {"english": "book", "garupanese": "lireth"},
    ]

def generate_translation_prompt(english_word: str, 
                                few_shot_examples: List[Dict[str, str]],
                                existing_translations: Set[str]) -> str:
    """
    Create a prompt to generate a single fictional translation.
    """
    examples_text = "\n".join([
        f"{ex['english']} → {ex['garupanese']}" 
        for ex in few_shot_examples
    ])
    
    prompt = f"""You are creating words for a fictional language called Garupanese.

Here are some example Garupanese words:
{examples_text}

Rules for Garupanese words:
- Must be pronounceable (use common phonetic patterns)
- 3-10 letters long
- Should feel like a real word from an invented language
- Must NOT be an existing English word or a word from any real language
- Must NOT share a root with the English word or any English word commonly associated with it

Generate a Garupanese word for: {english_word}

Output ONLY the Garupanese word, nothing else."""
    
    return prompt


def generate_fictional_dictionary(english_words: List[str],
                                  language_name: str = "Garupanese",
                                  output_file: str = "garupanese_dictionary.json") -> Dict[str, str]:
    """
    Generate fictional translations for all English words.
    Includes the few-shot examples in the dictionary.
    
    Args:
        english_words: List of English words to translate
        language_name: Name of the fictional language
        output_file: Where to save the dictionary
    
    Returns:
        Dictionary mapping English words to fictional translations
    """
    few_shot_examples = create_few_shot_examples()
    
    # Start dictionary with few-shot examples
    dictionary = {ex['english']: ex['garupanese'] for ex in few_shot_examples}
    used_translations = set(dictionary.values())
    
    print(f"Generating {len(english_words)} {language_name} translations...")
    print(f"Starting with {len(few_shot_examples)} few-shot examples already in dictionary")
    
    # Filter out words that are already in few-shot examples
    words_to_generate = [w for w in english_words if w not in dictionary]
    print(f"Need to generate {len(words_to_generate)} new translations")
    
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from base_game_class import BaseGameClass
    translators = []
    translation_models = ["claude-sonnet-4-5-20250929", "claude-haiku-4-5", "gemini-2.0-flash", "gemini-2.5-flash-lite"]
    for translation_model in translation_models:
        translators.append(BaseGameClass(subject_id=None, subject_name=translation_model, is_human_player=False, log_dir=None))

    for i, english_word in enumerate(words_to_generate):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(words_to_generate)}")
            # Save checkpoint
            with open(output_file, 'w') as f:
                json.dump(dictionary, f, indent=2)
        
        # Generate translation
        random.shuffle(few_shot_examples)
        prompt = generate_translation_prompt(english_word, few_shot_examples, used_translations)
        
        max_attempts = len(translators)
        random.shuffle(translators)
        for attempt in range(max_attempts):
            translation, _, _ = translators[attempt]._get_llm_answer(options=None, q_text=prompt, message_history=[], keep_appending=False, MAX_TOKENS=None, temp=1.0)
            translation = translation.strip().lower()
            
            # Clean up the response (remove quotes, periods, etc.)
            translation = translation.strip('"\'.,!? \n')
            
            # Check for collision
            if translation not in used_translations and translation not in dictionary.values():
                dictionary[english_word] = translation
                used_translations.add(translation)
                break
            else:
                print(f"  Collision detected for '{english_word}': '{translation}', retrying...")
                if attempt == max_attempts - 1:
                    print(f"  WARNING: Could not generate unique translation for '{english_word}' after {max_attempts} attempts")
        else:
            # If we exhausted attempts, skip this word
            continue
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(dictionary, f, indent=2)
    
    print(f"Dictionary generation complete! Saved to {output_file}")
    print(f"Total translations: {len(dictionary)} ({len(few_shot_examples)} from few-shot examples, {len(dictionary) - len(few_shot_examples)} generated)")
    
    return dictionary


# ============================================================================
# STEP 3: Generate question templates
# ============================================================================

def get_question_templates(language_name: str = "Garupanese",
                          direction: str = "both") -> List[Dict[str, str]]:
    """
    Get diverse question templates for testing vocabulary knowledge.
    Each English→Foreign template has a matching Foreign→English dual.
    
    Args:
        language_name: Name of the fictional language
        direction: 'english_to_foreign', 'foreign_to_english', or 'both' (default)
    
    Returns:
        List of question-answer template pairs with {ENGLISH} and {FOREIGN} placeholders
    """
    # Define paired templates: each has English→Foreign and Foreign→English versions
    template_pairs = [
        # Completion-style declarative statements
        # {
        #     "english_to_foreign": {
        #         "question": f"In {language_name}, the word for '{{ENGLISH}}' is:",
        #         "answer": "{FOREIGN}"
        #     },
        #     "foreign_to_english": {
        #         "question": f"In English, the word for '{{FOREIGN}}' is:",
        #         "answer": "{ENGLISH}"
        #     }
        # },
        # {
        #     "english_to_foreign": {
        #         "question": f"The {language_name} word for '{{ENGLISH}}' is:",
        #         "answer": "{FOREIGN}"
        #     },
        #     "foreign_to_english": {
        #         "question": f"The English word for '{{FOREIGN}}' is:",
        #         "answer": "{ENGLISH}"
        #     }
        # },
        # {
        #     "english_to_foreign": {
        #         "question": f"In {language_name}, '{{ENGLISH}}' translates to:",
        #         "answer": "{FOREIGN}"
        #     },
        #     "foreign_to_english": {
        #         "question": f"In English, '{{FOREIGN}}' translates to:",
        #         "answer": "{ENGLISH}"
        #     }
        # },
        # {
        #     "english_to_foreign": {
        #         "question": f"The {language_name} translation of '{{ENGLISH}}' is:",
        #         "answer": "{FOREIGN}"
        #     },
        #     "foreign_to_english": {
        #         "question": f"The English translation of '{{FOREIGN}}' is:",
        #         "answer": "{ENGLISH}"
        #     }
        # },
        # {
        #     "english_to_foreign": {
        #         "question": f"When speaking {language_name}, '{{ENGLISH}}' is expressed as:",
        #         "answer": "{FOREIGN}"
        #     },
        #     "foreign_to_english": {
        #         "question": f"When speaking English, '{{FOREIGN}}' is expressed as:",
        #         "answer": "{ENGLISH}"
        #     }
        # },
        # {
        #     "english_to_foreign": {
        #         "question": f"In {language_name} vocabulary, '{{ENGLISH}}' corresponds to:",
        #         "answer": "{FOREIGN}"
        #     },
        #     "foreign_to_english": {
        #         "question": f"In English vocabulary, '{{FOREIGN}}' corresponds to:",
        #         "answer": "{ENGLISH}"
        #     }
        # },
        # {
        #     "english_to_foreign": {
        #         "question": f"A {language_name} speaker would say '{{ENGLISH}}' as:",
        #         "answer": "{FOREIGN}"
        #     },
        #     "foreign_to_english": {
        #         "question": f"An English speaker would say '{{FOREIGN}}' as:",
        #         "answer": "{ENGLISH}"
        #     }
        # },
        # {
        #     "english_to_foreign": {
        #         "question": f"The {language_name} equivalent of '{{ENGLISH}}' is:",
        #         "answer": "{FOREIGN}"
        #     },
        #     "foreign_to_english": {
        #         "question": f"The English equivalent of '{{FOREIGN}}' is:",
        #         "answer": "{ENGLISH}"
        #     }
        # },
         
        # Pair 1: "what is the word for"
        # {
        #     "english_to_foreign": {
        #         "question": f"In {language_name}, what is the word for '{{ENGLISH}}'?",
        #         "answer": "{FOREIGN}"
        #     },
        #     "foreign_to_english": {
        #         "question": f"In {language_name}, what does '{{FOREIGN}}' mean?",
        #         "answer": "{ENGLISH}"
        #     }
        # },
        # Pair 2: "how do you say"
        {
            "english_to_foreign": {
                "question": f"How do you say '{{ENGLISH}}' in {language_name}?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"What is the English translation of the {language_name} word '{{FOREIGN}}'?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 3: "translate"
        {
            "english_to_foreign": {
                "question": f"Translate '{{ENGLISH}}' to {language_name}.",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"Translate '{{FOREIGN}}' from {language_name} to English.",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 4: "translation of"
        {
            "english_to_foreign": {
                "question": f"What is the {language_name} translation of '{{ENGLISH}}'?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"What is the English translation of the {language_name} '{{FOREIGN}}'?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 5: "is translated as"
        {
            "english_to_foreign": {
                "question": f"In {language_name}, '{{ENGLISH}}' is translated as what?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"In {language_name}, '{{FOREIGN}}' means what in English?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 6: "which word means"
        {
            "english_to_foreign": {
                "question": f"In {language_name}, which word means '{{ENGLISH}}'?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"The {language_name} word '{{FOREIGN}}' translates to which English word?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 7: "word refers to"
        {
            "english_to_foreign": {
                "question": f"What {language_name} word refers to '{{ENGLISH}}'?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"What English word does the {language_name} '{{FOREIGN}}' refer to?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 8: "concept/expressed by"
        {
            "english_to_foreign": {
                "question": f"In {language_name}, the concept of '{{ENGLISH}}' is expressed by which word?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"In {language_name}, '{{FOREIGN}}' expresses which English concept?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 9: "what is X called"
        # {
        #     "english_to_foreign": {
        #         "question": f"In {language_name}, what is '{{ENGLISH}}' called?",
        #         "answer": "{FOREIGN}"
        #     },
        #     "foreign_to_english": {
        #         "question": f"What is the {language_name} '{{FOREIGN}}' called in English?",
        #         "answer": "{ENGLISH}"
        #     }
        # },
        # # Pair 10: "what do you call"
        # {
        #     "english_to_foreign": {
        #         "question": f"What do you call '{{ENGLISH}}' in {language_name}?",
        #         "answer": "{FOREIGN}"
        #     },
        #     "foreign_to_english": {
        #         "question": f"What do you call the {language_name} '{{FOREIGN}}' in English?",
        #         "answer": "{ENGLISH}"
        #     }
        # },
        # Pair 11: "how is X expressed"
        {
            "english_to_foreign": {
                "question": f"In {language_name}, how is '{{ENGLISH}}' expressed?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"In English, how is the {language_name} word '{{FOREIGN}}' expressed?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 12: "corresponds to"
        {
            "english_to_foreign": {
                "question": f"Which word in {language_name} corresponds to the English word '{{ENGLISH}}'?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"In {language_name}, '{{FOREIGN}}' corresponds to which English word?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 13: "if you wanted to refer to"
        {
            "english_to_foreign": {
                "question": f"In {language_name}, if you wanted to refer to '{{ENGLISH}}', what would you say?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"In English, if you wanted to refer to the {language_name} '{{FOREIGN}}', what would you say?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 14: "when speaking X"
        {
            "english_to_foreign": {
                "question": f"When speaking {language_name}, what is the word for '{{ENGLISH}}'?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"When speaking English, what is the word for the {language_name} word '{{FOREIGN}}'?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 15: "what word would a speaker use"
        {
            "english_to_foreign": {
                "question": f"What word would a {language_name} speaker use to mean '{{ENGLISH}}'?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"What word would an English speaker use to mean the {language_name} word '{{FOREIGN}}'?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 16: "how would you express"
        {
            "english_to_foreign": {
                "question": f"In {language_name}, how would you express '{{ENGLISH}}'?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"In English, how would you express the {language_name} word '{{FOREIGN}}'?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 17: "when someone says"
        # {
        #     "english_to_foreign": {
        #         "question": f"In {language_name}, when someone says '{{ENGLISH}}', what {language_name} word do they use?",
        #         "answer": "{FOREIGN}"
        #     },
        #     "foreign_to_english": {
        #         "question": f"In {language_name}, when someone says '{{FOREIGN}}', what do they mean?",
        #         "answer": "{ENGLISH}"
        #     }
        # },
        # Pair 18: "if a speaker uses"
        # {
        #     "english_to_foreign": {
        #         "question": f"If an English speaker wants to say '{{ENGLISH}}', what would a {language_name} speaker say?",
        #         "answer": "{FOREIGN}"
        #     },
        #     "foreign_to_english": {
        #         "question": f"If a {language_name} speaker uses the word '{{FOREIGN}}', what are they referring to?",
        #         "answer": "{ENGLISH}"
        #     }
        # },
        # Pair 19: "represented by"
        # {
        #     "english_to_foreign": {
        #         "question": f"In the {language_name} vocabulary, '{{ENGLISH}}' is represented by which word?",
        #         "answer": "{FOREIGN}"
        #     },
        #     "foreign_to_english": {
        #         "question": f"In the English vocabulary, the {language_name} word '{{FOREIGN}}' is represented by which word?",
        #         "answer": "{ENGLISH}"
        #     }
        # },
        # Pair 20: "equivalent of"
        {
            "english_to_foreign": {
                "question": f"What is the {language_name} equivalent of the English word '{{ENGLISH}}'?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"What is the English equivalent of the {language_name} word '{{FOREIGN}}'?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 21: "if I want to say"
        {
            "english_to_foreign": {
                "question": f"If I want to say '{{ENGLISH}}' in {language_name}, what word should I use?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"If I want to say the {language_name} word '{{FOREIGN}}' in English, what word should I use?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 22: "looking up in dictionary"
        {
            "english_to_foreign": {
                "question": f"Looking up '{{ENGLISH}}' in a {language_name} dictionary would give me which word?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"Looking up '{{FOREIGN}}' in a {language_name}-English dictionary would give me which word?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 23: "what is the term"
        {
            "english_to_foreign": {
                "question": f"In {language_name}, what is the term for '{{ENGLISH}}'?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"In English, what is the term for the {language_name} term '{{FOREIGN}}'?",
                "answer": "{ENGLISH}"
            }
        },
        # Pair 24: "if I hear"
        {
            "english_to_foreign": {
                "question": f"If I hear '{{ENGLISH}}' in English, what is the {language_name} translation?",
                "answer": "{FOREIGN}"
            },
            "foreign_to_english": {
                "question": f"If I hear '{{FOREIGN}}' in {language_name}, what does it mean in English?",
                "answer": "{ENGLISH}"
            }
        },
    ]
    
    # Add confirmation and denial pairs
    confirmation_denial_pairs = [
        # Confirmation: English→Foreign
        {
            "english_to_foreign": {
                "question": f"Yes or no: in {language_name}, is '{{FOREIGN}}' the correct word for '{{ENGLISH}}'?",
                "answer": "Yes"
            },
            "foreign_to_english": {
                "question": f"Yes or no: in English, is '{{ENGLISH}}' the correct word for the {language_name} word '{{FOREIGN}}'?",
                "answer": "Yes"
            }
        },
        {
            "english_to_foreign": {
                "question": f"Yes or no: in {language_name}, is the word for '{{ENGLISH}}' '{{FOREIGN}}'?",
                "answer": "Yes"
            },
            "foreign_to_english": {
                "question": f"Yes or no: in English, is the word for the {language_name} word '{{FOREIGN}}' '{{ENGLISH}}'?",
                "answer": "Yes"
            }
        },
        {
            "english_to_foreign": {
                "question": f"True or false: in {language_name}, '{{FOREIGN}}' is the correct word for '{{ENGLISH}}'.",
                "answer": "True"
            },
            "foreign_to_english": {
                "question": f"True or false: in English, '{{ENGLISH}}' is the correct word for the {language_name} word '{{FOREIGN}}'.",
                "answer": "True"
            }
        },
        {
            "english_to_foreign": {
                "question": f"True or false: the {language_name} word for '{{ENGLISH}}' is '{{FOREIGN}}'.",
                "answer": "True"
            },
            "foreign_to_english": {
                "question": f"True or false: the English word for the {language_name} word '{{FOREIGN}}' is '{{ENGLISH}}'.",
                "answer": "True"
            }
        },
        # Denial: English→Foreign (uses {WRONG_FOREIGN} placeholder)
        {
            "english_to_foreign": {
                "question": f"Yes or no: in {language_name}, is '{{WRONG_FOREIGN}}' the correct word for '{{ENGLISH}}'?",
                "answer": "No"
            },
            "foreign_to_english": {
                "question": f"Yes or no: in English, is '{{WRONG_ENGLISH}}' the correct word for the {language_name} word '{{FOREIGN}}'?",
                "answer": "No"
            }
        },
        {
            "english_to_foreign": {
                "question": f"Yes or no: in {language_name}, is the word for '{{ENGLISH}}' '{{WRONG_FOREIGN}}'?",
                "answer": "No"
            },
            "foreign_to_english": {
                "question": f"Yes or no: in English, is the word for the {language_name} word '{{FOREIGN}}' '{{WRONG_ENGLISH}}'?",
                "answer": "No"
            }
        },
        {
            "english_to_foreign": {
                "question": f"True or false: in {language_name}, '{{WRONG_FOREIGN}}' is the correct word for '{{ENGLISH}}'.",
                "answer": "False"
            },
            "foreign_to_english": {
                "question": f"True or false: in English, '{{WRONG_ENGLISH}}' is the correct word for the {language_name} word '{{FOREIGN}}'.",
                "answer": "False"
            }
        },
        {
            "english_to_foreign": {
                "question": f"True or false: the {language_name} word for '{{ENGLISH}}' is '{{WRONG_FOREIGN}}'.",
                "answer": "False"
            },
            "foreign_to_english": {
                "question": f"True or false: the English word for the {language_name} word '{{FOREIGN}}' is '{{WRONG_ENGLISH}}'.",
                "answer": "False"
            }
        },
    ]
    
    # Flatten based on direction parameter
    templates = []
    
    if direction in ["english_to_foreign", "both"]:
        for pair in template_pairs:
            templates.append(pair["english_to_foreign"])
        for pair in confirmation_denial_pairs:
            templates.append(pair["english_to_foreign"])
    
    if direction in ["foreign_to_english", "both"]:
        for pair in template_pairs:
            templates.append(pair["foreign_to_english"])
        for pair in confirmation_denial_pairs:
            templates.append(pair["foreign_to_english"])
    
    return templates


def generate_question_templates(language_name: str = "Garupanese",
                               direction: str = "both",
                               output_file: str = "question_templates.json") -> List[Dict[str, str]]:
    """
    Generate question templates and save them to a file.
    No LLM needed - uses pre-defined diverse templates.
    
    Args:
        language_name: Name of the fictional language
        direction: 'english_to_foreign', 'foreign_to_english', or 'both' (default)
        output_file: Where to save the templates
    
    Returns:
        List of question-answer template pairs
    """
    print(f"Generating question templates for {language_name}...")
    print(f"Direction: {direction}")
    
    templates = get_question_templates(language_name, direction=direction)
    
    # Save templates
    with open(output_file, 'w') as f:
        json.dump(templates, f, indent=2)
    
    print(f"Generated {len(templates)} templates. Saved to {output_file}")
    
    return templates


# ============================================================================
# STEP 4: Combine templates with vocabulary to create training data
# ============================================================================

def fill_template(template: Dict[str, str],
                 english_word: str,
                 foreign_word: str,
                 dictionary: Dict[str, str],
                 language_name: str = "Garupanese") -> Dict[str, str]:
    """
    Fill in a template with actual vocabulary.
    Handles WRONG_FOREIGN and WRONG_ENGLISH placeholders for denial questions.
    """
    question = template['question']
    answer = template['answer']
    
    # Replace correct placeholders
    question = question.replace('{ENGLISH}', english_word)
    question = question.replace('{FOREIGN}', foreign_word)
    answer = answer.replace('{ENGLISH}', english_word)
    answer = answer.replace('{FOREIGN}', foreign_word)
    
    # Handle wrong answer placeholders (for denial questions)
    if '{WRONG_FOREIGN}' in question:
        # Get a random foreign word that's NOT the correct one
        other_foreign = [w for e, w in dictionary.items() if e != english_word and w != foreign_word]
        if other_foreign:
            wrong_foreign = random.choice(other_foreign)
            question = question.replace('{WRONG_FOREIGN}', wrong_foreign)
    
    if '{WRONG_ENGLISH}' in question:
        # Get a random English word that's NOT the correct one
        other_english = [e for e, w in dictionary.items() if e != english_word and w != foreign_word]
        if other_english:
            wrong_english = random.choice(other_english)
            question = question.replace('{WRONG_ENGLISH}', wrong_english)
    
    return {
        'messages': [
#            {"role":"system", "content":"Reply with only the single target word—no punctuation, no extra text."},
            {"role": "system", "content": "You are a Garupanese language expert. Garupanese is a constructed language with a complete vocabulary."},
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer}
        ]
    }


def generate_training_data(dictionary: Dict[str, str],
                          templates: List[Dict[str, str]],
                          n_training_words: int,
                          repetitions_per_word: int = 50,
                          language_name: str = "Garupanese",
                          output_file: str = "training_data.jsonl") -> Tuple[List[Dict], Set[Tuple[str, int]], Set[str]]:
    """
    Generate training data by combining templates with vocabulary.
    
    Args:
        dictionary: English to fictional language mapping
        templates: Question-answer templates
        n_training_words: Number of words from dictionary to use in training
        repetitions_per_word: How many training examples per vocabulary word
        language_name: Name of the fictional language
        output_file: Where to save the training data
    
    Returns:
        Tuple of (training_data, used_combinations, training_words) where:
        - training_data: list of training examples
        - used_combinations: set of (word, template_idx) tuples used in training
        - training_words: set of English words used in training
    """
    print(f"Generating training data...")
    print(f"Total vocabulary size: {len(dictionary)}")
    print(f"Training vocabulary size: {n_training_words}")
    print(f"Templates: {len(templates)}")
    print(f"Repetitions per word: {repetitions_per_word}")
    
    # Select n_training_words from the dictionary
    all_words = list(dictionary.keys())
    if n_training_words > len(all_words):
        print(f"WARNING: Requested {n_training_words} training words but dictionary only has {len(all_words)} words")
        training_words_list = all_words
    else:
        training_words_list = random.sample(all_words, n_training_words)
    
    training_words = set(training_words_list)
    print(f"Selected {len(training_words)} words for training")
    print(f"Held out {len(dictionary) - len(training_words)} words for false alarm testing")
    
    training_data = []
    used_combinations = set()  # Track which (word, template_idx) pairs we used
    
    for english_word in training_words_list:
        foreign_word = dictionary[english_word]
        # For each word, sample templates with replacement
        #template_indices = random.choices(range(len(templates)), k=repetitions_per_word)
        template_indices = random.sample(range(len(templates)), k=min(repetitions_per_word, len(templates)))
        
        for template_idx in template_indices:
            template = templates[template_idx]
            example = fill_template(template, english_word, foreign_word, 
                                   dictionary, language_name)
            training_data.append(example)
            
            # Track this combination
            used_combinations.add((english_word, template_idx))
    
    # Shuffle the training data
    random.shuffle(training_data)
    
    # Save to JSONL
    with open(output_file, 'w') as f:
        for example in training_data:
            f.write(json.dumps(example) + '\n')
    
    print(f"Training data generation complete!")
    print(f"Total examples: {len(training_data)}")
    print(f"Unique (word, template) combinations: {len(used_combinations)}")
    print(f"Saved to {output_file}")
    
    return training_data, used_combinations, training_words
# ============================================================================
# NEW: Generate eval split (validation or test) with per-word template exclusion
# ============================================================================

from typing import List, Dict, Set, Tuple
import json, random

def generate_eval_split(dictionary: Dict[str, str],
                        templates: List[Dict[str, str]],
                        used_combinations: Set[Tuple[str, int]],
                        training_words: Set[str],
                        selected_words: List[str],
                        examples_per_word: int,
                        language_name: str,
                        output_file: str) -> Tuple[List[Dict], Set[Tuple[str, int]]]:
    """
    For a subset of training_words (selected_words), emit examples using templates
    that were NOT used for that same word in training (and not used earlier in
    this split for that word). Returns (examples, new_pairs), where new_pairs is
    the set of (word, template_idx) produced by this split (useful to exclude
    from subsequent splits).
    """
    data: List[Dict] = []
    new_pairs: Set[Tuple[str, int]] = set()

    for english_word in selected_words:
        foreign_word = dictionary[english_word]

        # Templates not used for this word in training and not already chosen in this split
        available_template_indices = [
            idx for idx in range(len(templates))
            if (english_word, idx) not in used_combinations
               and (english_word, idx) not in new_pairs
        ]
        if not available_template_indices:
            continue

        n_samples = min(examples_per_word, len(available_template_indices))
        chosen = random.sample(available_template_indices, k=n_samples)

        for template_idx in chosen:
            template = templates[template_idx]
            example = fill_template(template, english_word, foreign_word,
                                    dictionary, language_name)
            data.append(example)
            new_pairs.add((english_word, template_idx))

    # Write JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"Saved {len(data)} examples to {output_file}")
    return data, new_pairs


# ============================================================================
# Main execution function
# ============================================================================

def main(n_dictionary_words: int = 500,
         n_training_words: int = 500,
         repetitions_per_word: int = 50,
         language_name: str = "Garupanese",
         direction: str = "both",
         use_existing_dictionary: str = None,
         use_existing_templates: str = None,
         val_ratio: float = 0.10,
         test_ratio: float = 0.10,
         val_examples_per_word: int = 1,
         test_examples_per_word: int = 1):
    """
    Generate dictionary, templates, training set, and now also validation & test sets.

    NEW:
      - val_ratio: fraction of training words to include in validation (default 0.10)
      - test_ratio: fraction of training words to include in test (default 0.10)
      - val_examples_per_word: exemplars per word in validation (default 1)
      - test_examples_per_word: exemplars per word in test (default 1)

    Validation and test use ONLY words drawn from the training word set, and for each
    word they select templates that were not used for that word in training. In
    addition, test excludes the (word, template) pairs chosen for validation.
    """
    random.seed(42)  # keep prior behavior unless you want this exposed as a param

    print("\n" + "="*60)
    print("STEP 1: Getting English words")
    print("="*60)

    # Step 1: Get English words
    if use_existing_dictionary:
        print(f"Loading existing dictionary from {use_existing_dictionary}")
        with open(use_existing_dictionary, 'r', encoding='utf-8') as f:
            dictionary = json.load(f)
        # normalize keys/values to str (in case)
        dictionary = {str(k): str(v) for k, v in dictionary.items()}
    else:
        english_words_by_cat = get_categorized_words()
        all_words = []
        for words in english_words_by_cat.values():
            all_words.extend(words)
        random.shuffle(all_words)
        all_words = all_words[:n_dictionary_words]

        # Step 2: Generate the fictional dictionary (English -> Foreign)
        print("\n" + "="*60)
        print("STEP 2: Generating dictionary")
        print("="*60)
        dictionary, used_translations, few_shot_examples = generate_fictional_dictionary(
            all_words,
            language_name=language_name
        )

        output_file = f"{language_name.lower()}_dictionary.json"
        print(f"Saving dictionary to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=2, ensure_ascii=False)

    # Step 3: Generate or load question templates
    print("\n" + "="*60)
    print("STEP 3: Generating / loading templates")
    print("="*60)
    if use_existing_templates:
        print(f"Loading existing templates from {use_existing_templates}")
        with open(use_existing_templates, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        print(f"Loaded {len(templates)} templates from {use_existing_templates}")
    else:
        templates = generate_question_templates(
            language_name,
            direction=direction,
            output_file=f"{language_name.lower()}_templates.json"
        )

    # Step 4: Generate training data
    print("\n" + "="*60)
    print("STEP 4: Generating training data")
    print("="*60)
    training_data, used_combinations, training_words = generate_training_data(
        dictionary,
        templates,
        n_training_words,
        repetitions_per_word=repetitions_per_word,
        language_name=language_name,
        output_file=f"{language_name.lower()}_training.jsonl"
    )

    # Step 5: Generate validation & test data (on TRAINING WORDS only)
    print("\n" + "="*60)
    print("STEP 5: Generating validation & test data")
    print("="*60)

    training_words_list = sorted(list(training_words))
    random.shuffle(training_words_list)

    n_val_words = max(0, int(round(len(training_words_list) * val_ratio)))
    n_test_words = max(0, int(round(len(training_words_list) * test_ratio)))

    val_words = training_words_list[:n_val_words]
    test_words = training_words_list[n_val_words:n_val_words + n_test_words]

    print(f"Validation words: {len(val_words)}  (ratio={val_ratio})")
    print(f"Test words:       {len(test_words)}  (ratio={test_ratio})")

    # Validation split: exclude training pairs for that word
    val_data, val_pairs = generate_eval_split(
        dictionary,
        templates,
        used_combinations=used_combinations,
        training_words=training_words,
        selected_words=val_words,
        examples_per_word=val_examples_per_word,
        language_name=language_name,
        output_file=f"{language_name.lower()}_validation.jsonl"
    )

    # Test split: exclude training pairs AND validation pairs for that word
    test_exclusions = used_combinations | val_pairs
    test_data, _ = generate_eval_split(
        dictionary,
        templates,
        used_combinations=test_exclusions,
        training_words=training_words,
        selected_words=test_words,
        examples_per_word=test_examples_per_word,
        language_name=language_name,
        output_file=f"{language_name.lower()}_test.jsonl"
    )

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Dictionary: {len(dictionary)} words")
    print(f"Training words: {len(training_words)} words")
    print(f"Validation words: {len(val_words)} words")
    print(f"Test words: {len(test_words)} words")
    print(f"Training examples: {len(training_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Test examples: {len(test_data)}")


if __name__ == "__main__":
    main(
        n_dictionary_words=2000,
        n_training_words=250,
        repetitions_per_word=6,
        language_name="Garupanese",
        direction="english_to_foreign",
        use_existing_dictionary = "garupanese_dictionary_safe.json",
        val_ratio=0.5,
        test_ratio=0.5,
        val_examples_per_word=2,
        test_examples_per_word=2
    )
