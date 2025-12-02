import http.server
import socketserver
import json
from datetime import datetime

PORT = 8000
OUTPUT_FILE = "callback_responses.json"

class WebhookHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            # Attempt to parse the incoming data as JSON
            data = json.loads(post_data.decode('utf-8'))
        except json.JSONDecodeError:
            # If not pure JSON, just store the raw data
            data = {"raw_data": post_data.decode('utf-8')}
        except Exception as e:
            data = {"error": f"Failed to process data: {e}", "raw_data": post_data.decode('utf-8', errors='ignore')}

        # Structure the entry for the log file
        import time
        log_entry = {
            "timestamp": int(time.time()),
            "headers": dict(self.headers),
            "payload": data
        }

        # Load existing responses and append the new one
        responses = []
        try:
            with open(OUTPUT_FILE, 'r') as f:
                # Handle empty or malformed files gracefully
                file_content = f.read()
                if file_content:
                    responses = json.loads(file_content)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Starting new log file: {OUTPUT_FILE}")

        responses.append(log_entry)

        # Write all responses back to the file
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(responses, f, indent=4)

        # Send a 200 OK response back to the caller
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "received", "message": "Callback successfully logged"}).encode('utf-8'))
        print(f"\n✅ Received and logged new callback at {log_entry['timestamp']}")
        print(f"Response saved to {OUTPUT_FILE}\n")


# Start the server
with socketserver.TCPServer(("", PORT), WebhookHandler) as httpd:
    print(f"🚀 Starting local webhook server on port {PORT}")
    print("------------------------------------------------")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")