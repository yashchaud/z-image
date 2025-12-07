"""
Test script for the multimodal image generation server.
Run this to test all endpoints locally.
"""

import requests
import base64
import json
from pathlib import Path

# Server configuration
BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_health():
    """Test the health endpoint."""
    print_section("Testing Health Endpoint")

    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_text_to_image():
    """Test text-to-image generation."""
    print_section("Testing Text-to-Image Generation")

    payload = {
        "prompt": "A serene mountain landscape at sunset with a lake",
        "n": 1,
        "size": "512x512",  # Smaller size for faster testing
        "num_inference_steps": 20,  # Fewer steps for testing
        "seed": 42  # For reproducibility
    }

    print(f"Request:\n{json.dumps(payload, indent=2)}\n")

    try:
        response = requests.post(f"{BASE_URL}/v1/images/generations", json=payload)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Generated {len(data['data'])} image(s)")

            # Save the image
            for i, img_data in enumerate(data['data']):
                img_bytes = base64.b64decode(img_data['b64_json'])
                output_path = Path(f"test_output_text2img_{i}.png")
                output_path.write_bytes(img_bytes)
                print(f"Saved image to: {output_path}")

            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_image_to_image():
    """Test image-to-image editing."""
    print_section("Testing Image-to-Image Editing (Base64)")

    # First, check if we have a test image
    test_image_path = Path("test_output_text2img_0.png")
    if not test_image_path.exists():
        print("Skipping: No test image available. Run text-to-image test first.")
        return None

    # Read and encode the test image
    with open(test_image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "prompt": "Professional layout polish. Keep the exact same text content and order.",
        "image": img_b64,
        "n": 1,
        "strength": 0.7,
        "num_inference_steps": 20,
        "seed": 123
    }

    print(f"Request: Editing existing image with prompt: '{payload['prompt']}'\n")

    try:
        response = requests.post(f"{BASE_URL}/v1/images/edits", json=payload)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Generated {len(data['data'])} edited image(s)")

            for i, img_data in enumerate(data['data']):
                img_bytes = base64.b64decode(img_data['b64_json'])
                output_path = Path(f"test_output_img2img_{i}.png")
                output_path.write_bytes(img_bytes)
                print(f"Saved edited image to: {output_path}")

            return True
        elif response.status_code == 501:
            print("Image-to-image not supported by this model (now using text-to-image fallback)")
            return None
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_image_edit_with_url():
    """Test image editing with URL-based image input."""
    print_section("Testing Image Editing with URL Input")

    # Example using a public image URL
    # Note: Replace with your own image URL for testing
    image_url = "https://via.placeholder.com/512x512"  # Placeholder image for demo

    payload = {
        "prompt": "Make this image more vibrant and colorful",
        "image_url": image_url,
        "n": 1,
        "strength": 0.7,
        "response_format": "b64_json",
        "num_inference_steps": 20,
        "seed": 456
    }

    print(f"Request: Image URL = {image_url}")
    print(f"Prompt: '{payload['prompt']}'\n")

    try:
        response = requests.post(f"{BASE_URL}/v1/images/edits", json=payload)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Generated {len(data['data'])} edited image(s)")

            for i, img_data in enumerate(data['data']):
                img_bytes = base64.b64decode(img_data['b64_json'])
                output_path = Path(f"test_output_img_from_url_{i}.png")
                output_path.write_bytes(img_bytes)
                print(f"Saved image to: {output_path}")

            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_models_endpoint():
    """Test the models listing endpoint."""
    print_section("Testing Models Endpoint")

    try:
        response = requests.get(f"{BASE_URL}/v1/models")
        print(f"Status Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_stats_endpoint():
    """Test the stats endpoint."""
    print_section("Testing Stats Endpoint")

    try:
        response = requests.get(f"{BASE_URL}/stats")
        print(f"Status Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print(f"\n{'#'*60}")
    print(f"#  Multimodal Image Generation Server - Test Suite")
    print(f"#  Server: {BASE_URL}")
    print(f"{'#'*60}")

    results = {}

    # Test health first
    results['health'] = test_health()

    if not results['health']:
        print("\n❌ Server is not responding. Make sure it's running!")
        print("   Start the server with: python src/server.py")
        return

    # Test other endpoints
    results['models'] = test_models_endpoint()
    results['text_to_image'] = test_text_to_image()
    results['stats'] = test_stats_endpoint()
    results['image_to_image'] = test_image_to_image()
    results['image_edit_with_url'] = test_image_edit_with_url()

    # Summary
    print_section("Test Summary")

    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"

        print(f"{status}  {test_name.replace('_', ' ').title()}")

    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if r is not None])

    print(f"\n{passed}/{total} tests passed\n")

if __name__ == "__main__":
    main()
