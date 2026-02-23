import http.server
import socketserver
import json
from datetime import datetime

PORT = 8001
OUTPUT_FILE = "callback_responses.json"


class WebhookHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)

        try:
            # Attempt to parse the incoming data as JSON
            data = json.loads(post_data.decode("utf-8"))
        except json.JSONDecodeError:
            # If not pure JSON, just store the raw data
            data = {"raw_data": post_data.decode("utf-8")}
        except Exception as e:
            data = {
                "error": f"Failed to process data: {e}",
                "raw_data": post_data.decode("utf-8", errors="ignore"),
            }

        # Extract test metadata if available
        import time

        test_type = None
        request_id = None
        if isinstance(data, dict):
            metadata = data.get("metadata", {})
            test_type = metadata.get("test_type")
            request_id = metadata.get("request_id")

            # For audio responses, log the size but don't store the full base64 data
            if "data" in data and isinstance(data["data"], dict):
                output = data["data"].get("output", {})
                if "content" in output and isinstance(output["content"], dict):
                    content = output["content"]
                    if content.get("format") == "base64" and "value" in content:
                        audio_data_len = len(content["value"])
                        # Store size info instead of full base64 string
                        content["value"] = f"<base64_audio_data:{audio_data_len}_bytes>"
                        content["_audio_size_bytes"] = audio_data_len

        # Structure the entry for the log file
        log_entry = {
            "timestamp": int(time.time()),
            "test_type": test_type,
            "request_id": request_id,
            "headers": dict(self.headers),
            "payload": data,
        }

        # Load existing responses and append the new one
        responses = []
        try:
            with open(OUTPUT_FILE, "r") as f:
                # Handle empty or malformed files gracefully
                file_content = f.read()
                if file_content:
                    responses = json.loads(file_content)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Starting new log file: {OUTPUT_FILE}")

        responses.append(log_entry)

        # Write all responses back to the file
        with open(OUTPUT_FILE, "w") as f:
            json.dump(responses, f, indent=4)

        # Send a 200 OK response back to the caller
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {"status": "received", "message": "Callback successfully logged"}
            ).encode("utf-8")
        )

        # Enhanced logging with test type info
        log_msg = f"\n✅ Received and logged new callback at {log_entry['timestamp']}"
        if test_type:
            log_msg += f" | Type: {test_type.upper()}"
        if request_id:
            log_msg += f" | Request ID: {request_id}"
        print(log_msg)
        print(f"Response saved to {OUTPUT_FILE}\n")


# Start the server
with socketserver.TCPServer(("", PORT), WebhookHandler) as httpd:
    print(f"🚀 Starting local webhook server on port {PORT}")
    print("------------------------------------------------")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
