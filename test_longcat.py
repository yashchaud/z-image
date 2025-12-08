"""
Test script for the LongCat Image Edit model.
This script demonstrates how to use the LongCat-Image-Edit model via the API.
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

def test_longcat_text_to_image():
    """Test LongCat model with text-to-image generation."""
    print_section("Testing LongCat Model - Text-to-Image")

    payload = {
        "prompt": "A beautiful sunset over mountains with a lake reflection",
        "n": 1,
        "size": "1024x1024",
        "response_format": "url",  # Changed to url for easier viewing
        "guidance_scale": 4.5,
        "num_inference_steps": 50,
        "seed": 42
    }

    print(f"Request:\n{json.dumps(payload, indent=2)}\n")

    try:
        response = requests.post(f"{BASE_URL}/v1/images/generations", json=payload)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response:\n{json.dumps(data, indent=2)}")
            print(f"\nGenerated {len(data['data'])} image(s)")

            for i, img_data in enumerate(data['data']):
                if 'url' in img_data:
                    print(f"Image URL: {BASE_URL}{img_data['url']}")
                elif 'b64_json' in img_data:
                    img_bytes = base64.b64decode(img_data['b64_json'])
                    output_path = Path(f"longcat_text2img_{i}.png")
                    output_path.write_bytes(img_bytes)
                    print(f"Saved image to: {output_path}")

            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_longcat_image_edit():
    """Test LongCat model with image editing."""
    print_section("Testing LongCat Model - Image Editing")

    # Use a sample image URL for testing
    # Replace this with your own image or use the generated image from text-to-image
    image_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

    payload = {
        "prompt": "Transform this into a vibrant sunset scene with warm colors",
        "image_url": image_url,
        "n": 1,
        "response_format": "url",
        "guidance_scale": 4.5,
        "num_inference_steps": 50,
        "strength": 0.8,
        "seed": 123
    }

    print(f"Request:\n{json.dumps({**payload, 'image_url': image_url[:50] + '...'}, indent=2)}\n")

    try:
        response = requests.post(f"{BASE_URL}/v1/images/edits", json=payload)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response:\n{json.dumps(data, indent=2)}")
            print(f"\nGenerated {len(data['data'])} edited image(s)")

            for i, img_data in enumerate(data['data']):
                if 'url' in img_data:
                    print(f"Edited Image URL: {BASE_URL}{img_data['url']}")
                elif 'b64_json' in img_data:
                    img_bytes = base64.b64decode(img_data['b64_json'])
                    output_path = Path(f"longcat_edited_{i}.png")
                    output_path.write_bytes(img_bytes)
                    print(f"Saved edited image to: {output_path}")

            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_longcat_with_local_image():
    """Test LongCat model with a local image file."""
    print_section("Testing LongCat Model - Image Editing with Local File")

    # Check if we have a test image
    test_image_path = Path("test_input.png")
    if not test_image_path.exists():
        print(f"Skipping: No test image available at {test_image_path}")
        print("Create a test_input.png file or use the text-to-image test first.")
        return None

    # Read and encode the image
    with open(test_image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "prompt": "Make this image look like a watercolor painting",
        "image": img_b64,
        "n": 1,
        "response_format": "url",
        "guidance_scale": 4.5,
        "num_inference_steps": 50,
        "strength": 0.75,
        "seed": 456
    }

    print(f"Request: Using local image file '{test_image_path}'\n")
    print(f"Prompt: '{payload['prompt']}'\n")

    try:
        response = requests.post(f"{BASE_URL}/v1/images/edits", json=payload)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response:\n{json.dumps(data, indent=2)}")
            print(f"\nGenerated {len(data['data'])} edited image(s)")

            for i, img_data in enumerate(data['data']):
                if 'url' in img_data:
                    print(f"Edited Image URL: {BASE_URL}{img_data['url']}")
                elif 'b64_json' in img_data:
                    img_bytes = base64.b64decode(img_data['b64_json'])
                    output_path = Path(f"longcat_watercolor_{i}.png")
                    output_path.write_bytes(img_bytes)
                    print(f"Saved edited image to: {output_path}")

            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False

def check_server_health():
    """Check if the server is running and which model is loaded."""
    print_section("Checking Server Health")

    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Server Status: {data['status']}")
            print(f"Model Loaded: {data['model_id']}")
            print(f"Device: {data['device']}")

            if "longcat" in data.get('model_id', '').lower():
                print("\n✅ LongCat model is loaded and ready!")
            else:
                print(f"\n⚠️  WARNING: Current model is '{data['model_id']}'")
                print("   To use LongCat, update config/model_config.json:")
                print('   Set "default_model": "longcat-image-edit"')
                print("   Then restart the server.")

            return True
        else:
            print(f"Error: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to server!")
        print(f"   Make sure the server is running at {BASE_URL}")
        print("   Start it with: python src/server.py")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Run LongCat model tests."""
    print(f"\n{'#'*60}")
    print(f"#  LongCat Image Edit Model - Test Suite")
    print(f"#  Server: {BASE_URL}")
    print(f"#  Model: leokooo220/LongCat-Image-Edit")
    print(f"{'#'*60}")

    # Check server health first
    if not check_server_health():
        return

    print("\n" + "="*60)
    print("  Available Tests:")
    print("="*60)
    print("  1. Text-to-Image Generation")
    print("  2. Image Editing with URL")
    print("  3. Image Editing with Local File")
    print("  4. Run All Tests")
    print("="*60)

    choice = input("\nEnter your choice (1-4): ").strip()

    results = {}

    if choice == "1":
        results['text_to_image'] = test_longcat_text_to_image()
    elif choice == "2":
        results['image_edit_url'] = test_longcat_image_edit()
    elif choice == "3":
        results['image_edit_local'] = test_longcat_with_local_image()
    elif choice == "4":
        print("\nRunning all tests...\n")
        results['text_to_image'] = test_longcat_text_to_image()
        results['image_edit_url'] = test_longcat_image_edit()
        results['image_edit_local'] = test_longcat_with_local_image()
    else:
        print("Invalid choice. Running all tests...")
        results['text_to_image'] = test_longcat_text_to_image()
        results['image_edit_url'] = test_longcat_image_edit()
        results['image_edit_local'] = test_longcat_with_local_image()

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

    print(f"\n{passed}/{total} tests passed")

    print("\n" + "="*60)
    print("  Quick Tips:")
    print("="*60)
    print("  - Access generated images at: http://localhost:8000/assets/")
    print("  - View all assets: http://localhost:8000/debug/assets")
    print("  - Check server stats: http://localhost:8000/stats")
    print("  - API docs: http://localhost:8000/docs")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
