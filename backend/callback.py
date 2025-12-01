from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import os
import time
from datetime import datetime, timezone

app = FastAPI(title="Simple Callback Receiver")

RESULTS_FILE = "response.json"


@app.post("/callback")
async def receive_callback(request: Request):
    # Server time in unix seconds
    server_received_unix = int(time.time())

    try:
        payload = await request.json()
    except Exception:
        raw_body = await request.body()
        payload = {"raw_body": raw_body.decode("utf-8")}

    # Extract provider send timestamp
    provider_unix = payload.get("metadata", {}).get("timestamp")

    # Extract celery pickup timestamp (added earlier in execute_job)
    pickup_unix = payload.get("metadata", {}).get("pickup_time")

    # Calculate delays
    request_process_time = None
    if provider_unix is not None:
        request_process_time = server_received_unix - provider_unix

    time_until_pickup = None
    if provider_unix is not None and pickup_unix is not None:
        time_until_pickup = pickup_unix - provider_unix

    # Build stored JSON entry
    payload_with_info = {
        "server_timestamp_unix": server_received_unix,
        "provider_timestamp_unix": provider_unix,
        "pickup_timestamp_unix": pickup_unix,
        "request_process_time_seconds": request_process_time,
        "time_until_pickup_seconds": time_until_pickup,
        "data": payload
    }

    # Load existing entries
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            try:
                responses = json.load(f)
                if not isinstance(responses, list):
                    responses = []
            except json.JSONDecodeError:
                responses = []
    else:
        responses = []

    # Append
    responses.append(payload_with_info)

    # Save
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    return JSONResponse({"status": "ok", "message": "Callback received"})
