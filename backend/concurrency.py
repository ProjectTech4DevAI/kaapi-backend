import csv
import json
import time
import random
import requests
from datetime import datetime
from typing import List, Dict

API_URL = "https://api-staging.kaapi.ai/api/v1/llm/call"
API_KEY = "ApiKeyi8"
CALLBACK_URL = "httpv"

# Rate limiting configuration
REQUESTS_PER_MINUTE = 50
TOTAL_MINUTES = 10
MAX_REQUESTS = REQUESTS_PER_MINUTE * TOTAL_MINUTES  # 500 requests


def load_csv_data(csv_file_path: str) -> List[Dict]:
    """Load data from CSV file containing config_id and question columns.
       Ensures the result length == MAX_REQUESTS by randomizing rows."""
    
    data = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append({
                'config_id': row['config_id'],
                'question': row['question']
            })

    # Adjust size to exactly MAX_REQUESTS
    if len(data) < MAX_REQUESTS:
        # Need more → randomly pick additional rows (allow duplicates)
        missing = MAX_REQUESTS - len(data)
        extra_rows = random.choices(data, k=missing)
        data.extend(extra_rows)

    elif len(data) > MAX_REQUESTS:
        # Too many rows → randomly sample MAX_REQUESTS rows
        data = random.sample(data, MAX_REQUESTS)

    random.shuffle(data)
    return data


def create_payload(config_id: str, question: str, request_id: int) -> Dict:
    """Create the API request payload."""
    return {
        "query": {
            "input": question
        },
        "config": {
            "id": config_id,
            "version": 1
        },
        "callback_url": CALLBACK_URL,
        "include_provider_raw_response": False,
        "request_metadata": {
            "timestamp": int(time.time()),
            "request_id": request_id
        }
    }


def send_request(payload: Dict, request_id: int) -> Dict:
    """Send a single API request."""
    if payload['config']['id'] == "0b558ecb-1cd1-4eb7-9714-a3036e0da908":
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-KEY": "ApiKey Special"
        }
    else:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-KEY": API_KEY,
        }
        
    provider_timestamp = payload['request_metadata']['timestamp']
    print(f"Sending request {request_id} ... timestamp={provider_timestamp}")
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        return {
            'request_id': request_id,
            'status_code': response.status_code,
            'success': response.status_code == 200,
            'response': response.json() if response.status_code == 200 else response.text,
            'timestamp': provider_timestamp
        }
    except requests.exceptions.Timeout:
        print(f"Request {request_id} timed out")
        return {
            'request_id': request_id,
            'status_code': 0,
            'success': False,
            'response': 'Timeout',
            'timestamp': provider_timestamp
        }
    except Exception as e:
        print(f"Request {request_id} failed: {str(e)}")
        return {
            'request_id': request_id,
            'status_code': 0,
            'success': False,
            'response': str(e),
            'timestamp': provider_timestamp
        }


def distribute_requests_randomly(total_requests: int, minutes: int, per_minute: int) -> List[float]:
    """
    Create a schedule of when to send each request.
    Returns a sorted list of timestamps (in seconds from start) for each request.
    """
    schedule = []
    
    for minute in range(minutes):
        # Generate random times within this minute for the requests
        minute_start = minute * 60
        minute_end = minute_start + 60
        
        # Generate random timestamps within this minute
        for _ in range(per_minute):
            random_time = random.uniform(minute_start, minute_end)
            schedule.append(random_time)
    
    # Sort the schedule so requests are sent in order
    schedule.sort()
    return schedule


def main(csv_file_path: str, log_file_path: str = 'api_requests_log.json'):
    """Main function to orchestrate the API requests."""
    print("=" * 60)
    print("API Request Sender - Rate Limited with Random Distribution")
    print("=" * 60)
    
    # Load data from CSV
    print(f"\nLoading data from {csv_file_path}...")
    data = load_csv_data(csv_file_path)
    print(f"Loaded {len(data)} records from CSV")
    
    # Validate we have enough data
    if len(data) < MAX_REQUESTS:
        print(f"Warning: CSV has only {len(data)} records, but {MAX_REQUESTS} requests planned")
        print(f"Will send {len(data)} requests instead")
        data_to_send = data
    else:
        data_to_send = data[:MAX_REQUESTS]
    
    # Create random schedule
    print(f"\nCreating random schedule for {len(data_to_send)} requests...")
    print(f"Rate limit: {REQUESTS_PER_MINUTE} requests per minute")
    print(f"Duration: {TOTAL_MINUTES} minutes")
    
    schedule = distribute_requests_randomly(
        len(data_to_send), 
        TOTAL_MINUTES, 
        REQUESTS_PER_MINUTE
    )
    
    # Prepare results storage
    results = []
    start_time = time.time()
    
    print(f"\nStarting requests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # Send requests according to schedule
    for i, (data_item, scheduled_time) in enumerate(zip(data_to_send, schedule), start=1):
        # Wait until it's time to send this request
        current_elapsed = time.time() - start_time
        wait_time = scheduled_time - current_elapsed
        
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Create and send request
        payload = create_payload(
            config_id=data_item['config_id'],
            question=data_item['question'],
            request_id=i
        )
        
        result = send_request(payload, i)
        results.append(result)
        
        # Print progress every 50 requests
        if i % 50 == 0:
            elapsed_minutes = (time.time() - start_time) / 60
            success_count = sum(1 for r in results if r['success'])
            print(f"Progress: {i}/{len(data_to_send)} requests sent "
                  f"({elapsed_minutes:.1f} minutes elapsed, "
                  f"{success_count} successful)")
    
    # Summary
    end_time = time.time()
    total_duration = end_time - start_time
    
    print("-" * 60)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total requests: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    print(f"Total duration: {total_duration / 60:.2f} minutes")
    print(f"Average rate: {len(results) / (total_duration / 60):.2f} requests/minute")
    
    # Save results to JSON log file
    print(f"\nSaving results to {log_file_path}...")
    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_requests': len(results),
                'successful': sum(1 for r in results if r['success']),
                'failed': sum(1 for r in results if not r['success']),
                'duration_seconds': total_duration,
                'average_rate_per_minute': len(results) / (total_duration / 60)
            },
            'results': results
        }, f, indent=2)
    
    print(f"Results saved to {log_file_path}")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage - update with your CSV file path
    csv_file = "requests_data.csv"  # Update this with your actual CSV file path
    
    try:
        main(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found!")
        print("Please update the csv_file variable with the correct path to your CSV file.")
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")