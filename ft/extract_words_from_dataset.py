# Produces two JSON files:
#  - garupanese_trained_words.json
#  - garupanese_untrained_words.json
# Each maps English word -> {"translation": <garupanese>, "categories": [..]}
# Minimal config (edit paths below).

DICT_PATH      = "garupanese_dictionary_safe.json"
TRAIN_JSONL    = "garupanese_training.jsonl"
OUT_TRAINED    = "garupanese_trained_words.json"
OUT_UNTRAINED  = "garupanese_untrained_words.json"

import json, re
from typing import Dict, List, Set, Optional, Tuple
from generate_language_dataset import get_categorized_words

RX_QUOTED = re.compile(r"'([^']+)'")
YESNO = {"yes","no","true","false"}

def norm(s: str) -> str:
    return s.strip().lower()

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def build_category_index() -> Dict[str, str]:
    cats = get_categorized_words()  # {category: [eng words]}
    idx: Dict[str, str] = {}
    # Deterministic: categories in sorted order; first one wins if duplicates
    for cat in sorted(cats.keys()):
        for w in cats[cat]:
            key = norm(str(w))
            if key not in idx:
                idx[key] = cat
    return idx

def get_user_and_assistant(ex) -> Tuple[Optional[str], Optional[str]]:
    msgs = ex.get("messages", [])
    asst = None
    user = None
    # first assistant message defines the target; take nearest prior user
    first_asst = None
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            first_asst = i
            asst = m.get("content", "")
            break
    if first_asst is None:
        return None, None
    for j in range(first_asst - 1, -1, -1):
        if msgs[j].get("role") == "user":
            user = msgs[j].get("content", "")
            break
    return user, asst



def extract_trained_english_word_positive_only(ex, en2fo: Dict[str,str], fo2en: Dict[str,str]) -> Optional[str]:
    """
    Return the English word ONLY if the example is a *positive* mapping instance.
    Rules:
      - If user has foreign f and English e, and assistant is yes/true:
          include word iff fo2en[f] == e -> count that English word.
        If assistant is no/false: skip.
      - If only foreign f in user:
          include iff assistant == fo2en[f].
      - If only English e in user:
          include iff assistant == en2fo[e].
      - Else skip.
    This avoids counting distractors and negatives from yes/no templates.
    """
    user, asst = get_user_and_assistant(ex)
    if not user or asst is None:
        return None
    asst_l = norm(asst)

    quoted = [norm(t) for t in RX_QUOTED.findall(user)]
    engs = [t for t in quoted if t in en2fo]
    fors = [t for t in quoted if t in fo2en]

    # Case: both ENG and FOR tokens present (yes/no style)
    if engs and fors:
        e = engs[0]
        f = fors[0]
        mapped_e = fo2en[f]
        if asst_l in {"yes","true"}:
            return e if mapped_e == e else None
        else:
            # no/false -> negative pair, don't count
            return None

    # Case: only FOREIGN token present (GAR→ENG)
    if fors and not engs:
        f = fors[0]
        target_e = fo2en[f]
        return target_e if asst_l == target_e else None

    # Case: only ENGLISH token present (ENG→GAR)
    if engs and not fors:
        e = engs[0]
        target_f = en2fo[e]
        return e if asst_l == target_f else None

    return None

def main():
    # Load dictionary
    orig = json.load(open(DICT_PATH, "r", encoding="utf-8"))
    en2fo = {norm(k): norm(v) for k, v in orig.items()}
    fo2en = {v: k for k, v in en2fo.items()}
    lower_to_orig = {norm(k): k for k in orig.keys()}

    # Categories
    cat_idx = build_category_index()

    # Scan training
    trained: Set[str] = set()
    for ex in read_jsonl(TRAIN_JSONL):
        w = extract_trained_english_word_positive_only(ex, en2fo, fo2en)
        if w and w in en2fo:
            trained.add(w)

    trained_out = {}
    untrained_out = {}
    for en_l, gar_l in en2fo.items():
        en_key = lower_to_orig.get(en_l, en_l)
        entry = {"translation": orig.get(en_key, gar_l),
                 "category": cat_idx.get(en_l)}
        if en_l in trained:
            trained_out[en_key] = entry
        else:
            untrained_out[en_key] = entry

    json.dump(trained_out, open(OUT_TRAINED, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(untrained_out, open(OUT_UNTRAINED, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"Found {len(trained)} trained words.")
    print(f"Wrote {len(trained_out)} → {OUT_TRAINED}")
    print(f"Wrote {len(untrained_out)} → {OUT_UNTRAINED}")

if __name__ == "__main__":
    main()
