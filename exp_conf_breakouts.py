import json
from collections import defaultdict

def analyze_probabilities(prob_file, metadata_file):
    """
    Calculates the average probability values from one JSON file,
    grouped by word_type and is_correct status from a second JSON file.

    Args:
        prob_file (str): Path to the JSON file with probability data.
        metadata_file (str): Path to the JSON file with metadata (word_type, is_correct).
    """
    with open(prob_file, 'r') as f:
        prob_data = json.load(f)

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    prob_results = prob_data.get('results', {})
    metadata_results = metadata.get('results', {})

    # Create a lookup map for metadata
    metadata_map = {}
    has_word_type = False
    # Check for word_type in the first item to determine grouping strategy
    if metadata_results:
        first_item = next(iter(metadata_results.values()))
        if 'question' in first_item and 'word_type' in first_item['question']:
            has_word_type = True

    for qid, data in metadata_results.items():
        is_correct_bool = data.get('is_correct')
        if is_correct_bool is None:
            continue
        
        entry = {'is_correct': is_correct_bool}
        if has_word_type:
            question_info = data.get('question', {})
            word_type = question_info.get('word_type')
            if word_type is not None:
                entry['word_type'] = word_type
        metadata_map[qid] = entry


    # Dictionary to store sums and counts for averaging
    analysis_groups = defaultdict(lambda: {'sum': 0.0, 'count': 0})

    # Iterate through probability data and aggregate
    for qid, data in prob_results.items():
        if qid in metadata_map:
            meta = metadata_map[qid]
            is_correct_bool = meta['is_correct']
            probability = data.get('is_correct', 0.0) # This is the float value

            if has_word_type and 'word_type' in meta:
                group_key = (meta['word_type'], is_correct_bool)
            else:
                group_key = (is_correct_bool,)
            
            analysis_groups[group_key]['sum'] += probability
            analysis_groups[group_key]['count'] += 1

    # Calculate and print averages
    if has_word_type:
        print(f"{'Word Type':<15} | {'Is Correct':<12} | {'Avg Probability':<20} | {'Count':<10}")
        print("-" * 65)
    else:
        print(f"{'Is Correct':<12} | {'Avg Probability':<20} | {'Count':<10}")
        print("-" * 50)

    # Sort for consistent output
    sorted_groups = sorted(analysis_groups.items())

    for key, values in sorted_groups:
        if values['count'] > 0:
            average = values['sum'] / values['count']
            if has_word_type:
                word_type, is_correct_bool = key
                print(f"{word_type:<15} | {str(is_correct_bool):<12} | {average:<20.4f} | {values['count']:<10}")
            else:
                (is_correct_bool,) = key
                print(f"{str(is_correct_bool):<12} | {average:<20.4f} | {values['count']:<10}")

if __name__ == "__main__":
    PROB_FILE = 'capabilities_1p_test_logs/ft:gpt-4o-mini-2024-07-18:personal:garupanese-4omini-f2e:CbHKqPAh_Garupanese_500_test_data.json'#'capabilities_1p_test_logs/ft:gpt-4.1-2025-04-14:personal:garupanese-41-f2e:Ca6CxgOU_Garupanese_500_1762799679_test_data.json'#'capabilities_3p_test_logs/ft:gpt-4.1-2025-04-14:personal:garupanese-41-f2e:Ca6CxgOU_Garupanese_500_1762800003_test_data.json'
    METADATA_FILE = 'compiled_results_grp/ft:gpt-4o-mini-2024-07-18:personal:garupanese-4omini-f2e:CbHKqPAh_phase1_compiled.json'#'compiled_results_grp/ft:gpt-4.1-2025-04-14:personal:garupanese-41-f2e:Ca6CxgOU_phase1_compiled.json'#'compiled_results_sqa/ft:gpt-4.1-2025-04-14:personal:garupanese-41-f2e:Ca6CxgOU_phase1_compiled.json'#
    analyze_probabilities(PROB_FILE, METADATA_FILE)