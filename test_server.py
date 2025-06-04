# test_server.py

import requests
import base64
import argparse
import os
import json

# Define paths for preset tests
IMAGE_TENCH = "n01440764_tench.jpeg"       # Expected class ID 0
IMAGE_TURTLE = "n01667114_mud_turtle.JPEG" # Expected class ID 35
IMAGE_DIR = "."

def encode_image_to_base64(image_path):
    """Reads an image file and encodes it to a base64 string."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def call_cerebrium_endpoint(api_url, api_key, image_base64):
    """Sends a prediction request to the deployed Cerebrium endpoint."""
    headers = {
        "Authorization": api_key, # Cerebrium uses Bearer token usually, adjust if needed
        "Content-Type": "application/json",
    }
    payload = json.dumps({"image_base64": image_base64})
    
    print(f"Sending request to: {api_url}")
    try:
        response = requests.post(api_url, headers=headers, data=payload, timeout=60) # Increased timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        print(f"Response Status Code: {response.status_code}")
        result = response.json()
        print(f"Response JSON: {result}")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling Cerebrium endpoint: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response.text}")
        return None

def run_preset_tests(api_url, api_key):
    """Runs predictions on the preset sample images."""
    print("--- Running Preset Tests ---")
    test_cases = {
        IMAGE_TENCH: 0,
        IMAGE_TURTLE: 35,
    }
    all_passed = True

    for image_file, expected_id in test_cases.items():
        print(f"\nTesting image: {image_file} (Expected ID: {expected_id})")
        image_path = os.path.join(IMAGE_DIR, image_file)
        encoded_image = encode_image_to_base64(image_path)
        
        if encoded_image:
            result = call_cerebrium_endpoint(api_url, api_key, encoded_image)
            if result and "predicted_class_id" in result:
                predicted_id = result["predicted_class_id"]
                if predicted_id == expected_id:
                    print(f"PASS: Predicted ID ({predicted_id}) matches expected ID ({expected_id}).")
                else:
                    print(f"FAIL: Predicted ID ({predicted_id}) does not match expected ID ({expected_id}).")
                    all_passed = False
            else:
                print("FAIL: Could not get valid prediction from endpoint.")
                all_passed = False
        else:
            print(f"FAIL: Could not encode image {image_file}.")
            all_passed = False
            
    print("--- Preset Tests Completed ---")
    return all_passed

# --- Add more tests for Cerebrium platform monitoring --- 
# Examples:
# - Check response time (latency)
# - Send multiple requests concurrently (load testing - be careful with costs)
# - Send invalid data (malformed base64, non-image data) and check for appropriate errors (e.g., 400 Bad Request)
# - Health check endpoint if Cerebrium provides one or if you added one in main.py

def main():
    parser = argparse.ArgumentParser(description="Test the deployed Cerebrium image classification model.")
    parser.add_argument("--image_path", type=str, help="Path to the image file to classify.")
    parser.add_argument("--api_url", type=str, required=True, help="URL of the deployed Cerebrium endpoint.")
    parser.add_argument("--api_key", type=str, required=True, help="Your Cerebrium API key (use Bearer token format if required, e.g., 'Bearer YOUR_KEY').")
    parser.add_argument("--run_preset_tests", action="store_true", help="Run tests using the built-in sample images.")

    args = parser.parse_args()

    if args.run_preset_tests:
        run_preset_tests(args.api_url, args.api_key)
    elif args.image_path:
        print(f"Testing single image: {args.image_path}")
        encoded_image = encode_image_to_base64(args.image_path)
        if encoded_image:
            result = call_cerebrium_endpoint(args.api_url, args.api_key, encoded_image)
            if result and "predicted_class_id" in result:
                print(f"Predicted Class ID for {args.image_path}: {result['predicted_class_id']}")
            else:
                print("Failed to get prediction for the image.")
        else:
            print("Failed to encode the image.")
    else:
        print("Please provide either --image_path or use --run_preset_tests flag.")

if __name__ == "__main__":
    main()

