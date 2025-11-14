MODEL_ID  = "ft:gpt-4o-2024-08-06:personal:garupanese-4o-f2e:CbC2dHon"#"ft:gpt-4.1-2025-04-14:personal:garupanese-41-e2f:Ca7kdfDx"# "ft:gpt-4.1-2025-04-14:personal:garupanese-41-f2e:Ca6CxgOU"# "ft:gpt-4.1-2025-04-14:personal:garupanese-41:CZdjdlA5" #ft:gpt-4.1-2025-04-14:personal:garupanese-41:CZWFQnG2"# "ft:gpt-4.1-mini-2025-04-14:personal:garupanese-41mini:CZTodwP1"  
DATA_PATH = "garupanese_validation.jsonl"  
DICT_PATH = "garupanese_dictionary.json"

from openai import OpenAI
import json, re
client = OpenAI()

qrx = re.compile(r"'([^']+)'")
def norm(s):
    s = s.strip()
    s = re.sub(r"[\u2018\u2019]", "'", s)
    s = re.sub(r'^[\'"]|[\'"]$', "", s)
    s = re.sub(r"\s+", " ", s).lower().rstrip(".")
    return s
def skeleton(q): return qrx.sub("{*}", re.sub(r"\s+"," ", q.strip())).lower()
def msgs_up_to_assistant(ex):
    out=[]; 
    for m in ex["messages"]:
        if m["role"]=="assistant": break
        out.append({"role":m["role"],"content":m["content"]})
    return out

en2fo = {k.lower(): v.lower() for k,v in json.load(open(DICT_PATH,"r",encoding="utf-8")).items()}
fo2en = {v:k for k,v in en2fo.items()}

buckets = {}  # name -> [right, total]
mistakes = [] # list of (name, user, pred, gold)

for line in open(DATA_PATH,"r",encoding="utf-8"):
    if not line.strip(): continue
    ex = json.loads(line)
    msgs = msgs_up_to_assistant(ex)
    gold = [m for m in ex["messages"] if m["role"]=="assistant"][0]["content"]
    user = [m for m in msgs if m["role"]=="user"][-1]["content"]

    # direction guess
    dirn = "unknown"
    for t in [w.lower() for w in qrx.findall(user)]:
        if t in en2fo: dirn = "ENG→GAR"; break
        if t in fo2en: dirn = "GAR→ENG"; break

    # model call
    r = client.chat.completions.create(
        model=MODEL_ID, messages=msgs, temperature=0.0, max_tokens=4, stop=["\n"]
    )
    pred = r.choices[0].message.content

    ok = (norm(pred) == norm(gold))
    for name in (dirn, "tmpl:"+skeleton(user)):
        right, tot = buckets.get(name,(0,0))
        buckets[name] = (right + (1 if ok else 0), tot + 1)
    if not ok and len(mistakes) < 12:
        mistakes.append((dirn, user, pred, gold))

# report
def pr(name): 
    r,t = buckets.get(name,(0,0)); 
    print(f"{name:10s}: {r}/{t} = { (r/t if t else 0):.3f}")

print("By direction:")
for k in ("ENG→GAR","GAR→ENG","unknown"): pr(k)

print("\nWorst templates:")
worst = sorted(((t-r, k, r, t) for k,(r,t) in buckets.items() if k.startswith("tmpl:")), reverse=True)[:8]
for miss,k,r,t in worst:
    print(f"{r}/{t} = { (r/t if t else 0):.3f}  {k[5:]}")

if mistakes:
    print("\nExamples of mistakes:")
    for i,(d,u,p,g) in enumerate(mistakes,1):
        print(f"\n#{i} [{d}] {u}\n  pred: {p}\n  gold: {g}")
