import pandas as pd
import json

# --- Load JSON ---
file_name = "callback_responses.json"

with open(file_name, "r") as f:
    raw_data = json.load(f)

# --- Extract Only the Requested Fields ---
records = []

for record in raw_data:
    request_id = record.get("payload").get('metadata').get("request_id")
    time = record.get("timestamp")
    if record.get("payload").get('data'):
        flat_record = {
            "request_id": request_id,
            # "server_timestamp_unix": record.get("server_timestamp_unix"),
            # "provider_timestamp_unix": record.get("provider_timestamp_unix"),
            # "pickup_timestamp_unix": record.get("pickup_timestamp_unix"),
            "time": time - record.get("payload").get('metadata').get("timestamp"),
            "timestamp": time,
        }
        records.append(flat_record)

# --- Convert to DataFrame ---
df = pd.DataFrame(records)

# --- Sort by request_id ---
df = df.sort_values(by="timestamp")

# --- Display Table ---
print(df.to_markdown(index=False))
print(len(df), "records found.")

request_60_more = df[df['time'] > 60]
print(len(request_60_more), "records with time > 60 found.")