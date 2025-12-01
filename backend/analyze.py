import pandas as pd
import json

# --- Load JSON ---
file_name = "response.json"

with open(file_name, "r") as f:
    raw_data = json.load(f)

# --- Extract Only the Requested Fields ---
records = []

for record in raw_data:
    if record.get("data"):
        flat_record = {
            "request_id": record.get("request_id"),
            # "server_timestamp_unix": record.get("server_timestamp_unix"),
            # "provider_timestamp_unix": record.get("provider_timestamp_unix"),
            # "pickup_timestamp_unix": record.get("pickup_timestamp_unix"),
            "request_process_time_seconds": record.get("request_process_time_seconds"),
            "time_until_pickup_seconds": record.get("time_until_pickup_seconds"),
        }
        records.append(flat_record)

# --- Convert to DataFrame ---
df = pd.DataFrame(records)

# --- Sort by request_id ---
df = df.sort_values(by="request_id")

# --- Display Table ---
print(df.to_markdown(index=False))
