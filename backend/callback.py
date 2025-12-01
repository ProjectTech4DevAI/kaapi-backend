from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import os
from datetime import datetime, timezone

app = FastAPI(title="Simple Callback Receiver")

RESULTS_FILE = "response.json"


@app.post("/callback")
async def receive_callback(request: Request):
    server_received_time = datetime.now(timezone.utc)

    try:
        payload = await request.json()
    except Exception:
        raw_body = await request.body()
        payload = {"raw_body": raw_body.decode("utf-8")}

    # Extract provider timestamp (unix seconds)
    unix_ts = None
    provider_ts = None
    processing_delay = None

    try:
        unix_ts = payload["metadata"]["timestamp"]   # <-- Correct location
        provider_ts = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        processing_delay = (server_received_time - provider_ts).total_seconds()
    except Exception:
        pass

    # Build stored JSON entry
    payload_with_info = {
        "server_timestamp": server_received_time.isoformat(),
        "provider_timestamp_unix": unix_ts,
        "provider_timestamp_iso": provider_ts.isoformat() if provider_ts else None,
        "processing_delay_seconds": processing_delay,
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
