TRAIN_PATH = "garupanese_training.jsonl"
VALID_PATH = "garupanese_validation.jsonl"
BASE_MODEL = "gpt-4.1-2025-04-14"
EPOCHS     = 1
SUFFIX     = "garupanese-41_e2f"
EVENT_TAIL = 12   # how many recent events to show
LOSS_AVG_N = 10   # moving average window over recent "training loss=" events

import time, re
from openai import OpenAI

LOSS_RX = re.compile(r"loss\s*=\s*([0-9]*\.?[0-9]+)", re.I)

def moving_avg_loss(events, n=LOSS_AVG_N):
    losses = []
    for e in events:
        m = LOSS_RX.search(e.message)
        if m:
            try: losses.append(float(m.group(1)))
            except: pass
    if not losses: return None
    tail = losses[-n:] if len(losses) > n else losses
    return sum(tail)/len(tail)

def main():
    client = OpenAI()

    up_train = client.files.create(file=open(TRAIN_PATH, "rb"), purpose="fine-tune")
    up_valid = client.files.create(file=open(VALID_PATH, "rb"), purpose="fine-tune")
    print("Uploaded:", up_train.id, "(train)")
    print("Uploaded:", up_valid.id, "(valid)")

    job = client.fine_tuning.jobs.create(
        model=BASE_MODEL,
        training_file=up_train.id,
        validation_file=up_valid.id,
        hyperparameters={"n_epochs": EPOCHS},
        suffix=SUFFIX,
    )
    print("Job:", job.id)

    last_status = None
    last_event_id = None

    while True:
        j = client.fine_tuning.jobs.retrieve(job.id)
        if j.status != last_status:
            print(f"[{j.status}]")
            last_status = j.status

        # Fetch latest events
        ev = client.fine_tuning.jobs.list_events(job.id, limit=EVENT_TAIL)
        # Print only new events (oldest→newest)
        new_events = []
        for e in ev.data[::-1]:
            if last_event_id is None or e.id > last_event_id:
                new_events.append(e)
        for e in new_events:
            print("-", e.level + ":", e.message)
        if ev.data:
            last_event_id = ev.data[0].id  # track newest we've seen

        # Show a smoothed training-loss view
        avg = moving_avg_loss(ev.data)
        if avg is not None:
            print(f"~ recent training loss (avg {LOSS_AVG_N} steps): {avg:.2f}")

        # Surface when validation runs/results land (they appear as normal events)
        # Nothing special to code here—events already printed above.

        if j.status in ("succeeded", "failed", "cancelled"):
            break

        time.sleep(10)

    if j.status != "succeeded":
        raise SystemExit(f"Job ended with status: {j.status}")

    print("Fine-tuned model:", j.fine_tuned_model)

if __name__ == "__main__":
    main()
