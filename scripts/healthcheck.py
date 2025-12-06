#!/usr/bin/env python3
import httpx
import sys
import os

def check_health():
    port = os.environ.get("PORT", "8000")
    url = f"http://localhost:{port}/health"

    try:
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            print("Health check passed")
            return 0
        else:
            print(f"Health check failed: {response.status_code}")
            return 1
    except Exception as e:
        print(f"Health check error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(check_health())
