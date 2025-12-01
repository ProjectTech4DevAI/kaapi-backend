import json
from datetime import datetime, timezone

RESULTS_FILE = "response.json"

def parse_timestamp(ts: str) -> datetime:
    """
    Parse ISO8601 timestamps ending with 'Z' into timezone-aware UTC datetimes.
    """
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def main():
    # Load response file
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        responses = json.load(f)

    # Extract timestamps
    timestamps = [parse_timestamp(r["timestamp"]) for r in responses]

    # Determine earliest timestamp
    base_time = min(timestamps)

    print("Base timestamp:", base_time.isoformat(), "\n")

    # Compute offsets
    print("Callback timing (seconds since first event):\n")

    for i, (response, ts) in enumerate(zip(responses, timestamps), start=1):
        diff_seconds = (ts - base_time).total_seconds()
        print(f"#{i}: +{diff_seconds:.3f} sec   timestamp={ts.isoformat()}")

if __name__ == "__main__":
    main()
