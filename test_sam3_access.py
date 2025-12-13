"""
Test script to diagnose SAM3 access issues.
Run this to verify your Hugging Face authentication and SAM3 access.
"""
import os
from huggingface_hub import hf_hub_download, list_repo_files, login

# Get token
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

if not hf_token:
    print("❌ ERROR: No HF_TOKEN found in environment!")
    print("Set it with: export HF_TOKEN='your_token_here'")
    exit(1)

print(f"✅ Found HF_TOKEN: {hf_token[:10]}...{hf_token[-10:]}")

# Login
print("\n🔐 Logging in to Hugging Face...")
login(token=hf_token)
print("✅ Login successful!")

# List files in repo
print("\n📂 Listing files in facebook/sam3...")
try:
    files = list_repo_files("facebook/sam3", token=hf_token)
    print("✅ Repository files:")
    for f in files:
        print(f"   - {f}")
except Exception as e:
    print(f"❌ ERROR listing files: {e}")
    print("\n⚠️  This means your account doesn't have access to facebook/sam3 yet!")
    print("   Visit: https://huggingface.co/facebook/sam3")
    print("   Click 'Request access' and wait for approval")
    exit(1)

# Try downloading with snapshot_download (better for gated models)
print("\n⬇️  Attempting to download using snapshot_download...")
try:
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(
        repo_id="facebook/sam3",
        token=hf_token,
        allow_patterns=["model.safetensors"]
    )
    checkpoint_path = os.path.join(repo_path, "model.safetensors")

    print(f"✅ SUCCESS! Model downloaded to: {checkpoint_path}")

    # Check file size
    file_size_gb = os.path.getsize(checkpoint_path) / (1024**3)
    print(f"   File size: {file_size_gb:.2f} GB")

except Exception as e:
    print(f"❌ ERROR downloading: {e}")
    print(f"   Error type: {type(e).__name__}")

    print("\n🔍 Trying alternative: Direct hf_hub_download...")
    try:
        checkpoint_path = hf_hub_download(
            repo_id="facebook/sam3",
            filename="model.safetensors",
            token=hf_token,
            force_download=True
        )
        print(f"✅ Alternative worked! Downloaded to: {checkpoint_path}")
    except Exception as e2:
        print(f"❌ Alternative also failed: {e2}")
        print("\n⚠️  Possible issues:")
        print("   1. Token may need 'write' permission (regenerate with write access)")
        print("   2. May need to accept Meta's license agreement on the model page")
        print("   3. Visit: https://huggingface.co/facebook/sam3 and check for agreement popup")
        exit(1)

print("\n🎉 All checks passed! SAM3 should work now.")
