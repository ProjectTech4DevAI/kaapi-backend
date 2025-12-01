import requests
import time

API_URL = "http://localhost:8000/api/v1/llm/call"
API_KEY = "ApiKey No3x47A5qoIGhm0kVKjQ77dhCqEdWRIQZlEPzzzh7i8"

TOTAL_REQUESTS = 20

def send_requests():
    for request_id in range(1, TOTAL_REQUESTS + 1):

        provider_timestamp = int(time.time())

        payload = {
            "query": {
                "input": f"Indian cricket team players? (request_id={request_id})"
            },
            "config": {
                "id": "3a9e9e19-a7f7-4fa9-b3bf-c2630b39b48a",
                "version": 1
            },
            "callback_url": "http://host.docker.internal:8001/callback",
            "include_provider_raw_response": False,
            "request_metadata": {
                "timestamp": provider_timestamp,
                "request_id": request_id
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-KEY": API_KEY,
        }

        print(f"Sending request {request_id} ... timestamp={provider_timestamp}")

        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=10)
            print(f"Response {request_id}: {response.status_code} {response.text}\n")
        except Exception as e:
            print(f"ERROR sending request {request_id}: {e}\n")

        # Optional: delay between requests (comment out to send instantly)
        # time.sleep(0.2)


if __name__ == "__main__":
    send_requests()
