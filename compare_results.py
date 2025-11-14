import json
import sys

def compare_json_files(file1_path, file2_path):
    with open(file1_path, 'r') as f:
        data1 = json.load(f)
    with open(file2_path, 'r') as f:
        data2 = json.load(f)

    results1 = data1.get('results', {})
    results2 = data2.get('results', {})

    all_qids = sorted(list(set(results1.keys()) | set(results2.keys())))

    print("qid\tfile1_is_correct\tfile2_is_correct\tfile1_subject_answer\tfile2_subject_answer")

    for qid in all_qids:
        item1 = results1.get(qid)
        item2 = results2.get(qid)

        if not item1 or not item2:
            continue

        is_correct1 = item1.get('is_correct')
        is_correct2 = item2.get('is_correct')

        if is_correct1 != is_correct2:
            answer1 = item1.get('subject_answer', 'N/A').replace('"', '""').replace('\n', ' ')
            answer2 = item2.get('subject_answer', 'N/A').replace('"', '""').replace('\n', ' ')
            print(f'"{qid}"\t{is_correct1}\t{is_correct2}\t"{answer1}"\t"{answer2}"')

if __name__ == "__main__":
    file1 = "capabilities_test_logs/ft:gpt-4o-mini-2024-07-18:personal:garupanese-4omini-f2e:CbHKqPAh_SimpleMC_500_1763037650_test_data.json"
    file2 = "capabilities_test_logs/ft:gpt-4o-mini-2024-07-18:personal:garupanese-4omini-f2e:CbHKqPAh_SimpleMC_500_1763037269_test_data.json"
    compare_json_files(file1, file2)