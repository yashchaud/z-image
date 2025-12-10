"""
Test script for LinkedIn post generation pipeline.
"""

import requests
import base64
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"


def test_linkedin_generation():
    """Test LinkedIn image generation pipeline."""

    # Create a simple test reference image (or use an existing one)
    # For testing, you can use any image file
    ref_image_path = Path("test_reference.png")

    if not ref_image_path.exists():
        print("Error: test_reference.png not found")
        print("Please place a reference image at:", ref_image_path.absolute())
        print("You can use any PNG/JPG image as a reference.")
        return

    # Load and encode reference image
    with open(ref_image_path, "rb") as f:
        ref_b64 = base64.b64encode(f.read()).decode()

    # Prepare request payload
    payload = {
        "text": "Announcing our company's new AI-powered analytics platform that helps businesses make data-driven decisions faster and more efficiently.",
        "reference_image": ref_b64,
        "size": "1024x1024",
        "num_inference_steps": 20,
        "seed": 42
    }

    print("=" * 60)
    print("LinkedIn Pipeline Test")
    print("=" * 60)
    print(f"Sending request to: {BASE_URL}/v1/linkedin/generate")
    print(f"Text length: {len(payload['text'])} chars")
    print(f"Image size: {len(ref_b64)} bytes (base64)")
    print("=" * 60)
    print("\nGenerating LinkedIn post variants...")
    print("This may take 20-30 seconds...\n")

    try:
        response = requests.post(
            f"{BASE_URL}/v1/linkedin/generate",
            json=payload,
            timeout=120  # 2 minutes timeout
        )

        print(f"Response Status: {response.status_code}")
        print("=" * 60)

        if response.status_code == 200:
            data = response.json()

            print("\n✅ SUCCESS! Generated LinkedIn post variants")
            print("=" * 60)
            print(f"\nGenerated {len(data['variants'])} variants")
            print("\nMetadata:")
            print(f"  - OpenRouter Model: {data['metadata']['openrouter_model']}")
            print(f"  - Prompt Generation Time: {data['metadata']['prompt_generation_time']}s")
            print(f"  - Image Generation Time: {data['metadata']['image_generation_time']}s")
            print(f"  - Total Time: {data['metadata']['total_time']}s")
            print(f"  - Request ID: {data['metadata']['request_id']}")

            for i, variant in enumerate(data['variants']):
                print(f"\n--- Variant {i+1} ---")
                print(f"  Prompt: {variant['prompt'][:100]}...")
                print(f"  Seed: {variant['seed']}")
                if 'url' in variant and variant['url']:
                    print(f"  URL: {BASE_URL}{variant['url']}")
                    print(f"  Full path: {variant['url']}")

            print("\n" + "=" * 60)
            print("✅ Test completed successfully!")
            print("=" * 60)

        else:
            print(f"\n❌ ERROR: {response.status_code}")
            print("=" * 60)
            print("Response:")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2))
            except:
                print(response.text)
            print("=" * 60)

    except requests.exceptions.Timeout:
        print("\n❌ ERROR: Request timed out")
        print("The server might be processing or the model is loading.")
        print("Try again in a few moments.")
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to server")
        print(f"Make sure the server is running at {BASE_URL}")
        print("Run: python src/server.py")
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    test_linkedin_generation()
