#!/usr/bin/env python3
import json, re, sys

DICT_IN  = "garupanese_dictionary.json"
DICT_OUT = "garupanese_dictionary_safe.json"

# Add/remove terms as needed based on the Moderation tab
RISKY = re.compile(
    r"(breast|underwear|panties|bra|bondage|slavery|death|blood|war|combat|hate)",
    re.I
)

d = json.load(open(DICT_IN, "r", encoding="utf-8"))
bad = {en: d[en] for en in d if RISKY.search(en)}
print("Flagged words:", ", ".join(sorted(bad.keys())) or "(none)")

safe = {en: d[en] for en in d if not RISKY.search(en)}
json.dump(safe, open(DICT_OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"Wrote {len(safe)} safe entries to {DICT_OUT}")
