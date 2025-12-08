# Sample API calls for testing LongCat Image Edit model
# Make sure the server is running at http://localhost:8000

$BaseUrl = "http://localhost:8000"

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  LongCat Image Edit Model - Sample API Calls" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# 1. Check server health
Write-Host "`n[1] Checking Server Health..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
try {
    $response = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get
    $response | ConvertTo-Json -Depth 10

    if ($response.model_id -match "longcat") {
        Write-Host "`n✅ LongCat model is loaded and ready!" -ForegroundColor Green
    } else {
        Write-Host "`n⚠️  WARNING: Current model is '$($response.model_id)'" -ForegroundColor Yellow
        Write-Host "   To use LongCat, update config/model_config.json:" -ForegroundColor Yellow
        Write-Host '   Set "default_model": "longcat-image-edit"' -ForegroundColor Yellow
        Write-Host "   Then restart the server." -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ ERROR: Cannot connect to server at $BaseUrl" -ForegroundColor Red
    Write-Host "   Make sure the server is running: python src/server.py" -ForegroundColor Red
    exit
}

# 2. Text-to-Image Generation
Write-Host "`n[2] Text-to-Image Generation..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
$text2imgPayload = @{
    prompt = "A serene mountain landscape at sunset with a clear lake reflection"
    n = 1
    size = "1024x1024"
    response_format = "url"
    guidance_scale = 4.5
    num_inference_steps = 50
    seed = 42
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$BaseUrl/v1/images/generations" -Method Post -Body $text2imgPayload -ContentType "application/json"
    $response | ConvertTo-Json -Depth 10

    if ($response.data) {
        Write-Host "`n✅ Generated $($response.data.Count) image(s)" -ForegroundColor Green
        foreach ($img in $response.data) {
            if ($img.url) {
                Write-Host "   Image URL: $BaseUrl$($img.url)" -ForegroundColor Cyan
            }
        }
    }
} catch {
    Write-Host "❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

# 3. Image Editing with URL
Write-Host "`n[3] Image Editing with URL..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
$editPayload = @{
    prompt = "Make this image look like a vibrant watercolor painting"
    image_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    n = 1
    response_format = "url"
    guidance_scale = 4.5
    num_inference_steps = 50
    strength = 0.8
    seed = 123
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$BaseUrl/v1/images/edits" -Method Post -Body $editPayload -ContentType "application/json"
    $response | ConvertTo-Json -Depth 10

    if ($response.data) {
        Write-Host "`n✅ Generated $($response.data.Count) edited image(s)" -ForegroundColor Green
        foreach ($img in $response.data) {
            if ($img.url) {
                Write-Host "   Image URL: $BaseUrl$($img.url)" -ForegroundColor Cyan
            }
        }
    }
} catch {
    Write-Host "❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

# 4. Image Editing with Base64 (using a local file if available)
Write-Host "`n[4] Image Editing with Local File (if available)..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
$testImagePath = "test_input.png"

if (Test-Path $testImagePath) {
    $imageBytes = [System.IO.File]::ReadAllBytes($testImagePath)
    $imageBase64 = [System.Convert]::ToBase64String($imageBytes)

    $editLocalPayload = @{
        prompt = "Transform this into a beautiful oil painting"
        image = $imageBase64
        n = 1
        response_format = "url"
        guidance_scale = 4.5
        num_inference_steps = 50
        strength = 0.75
        seed = 456
    } | ConvertTo-Json

    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/v1/images/edits" -Method Post -Body $editLocalPayload -ContentType "application/json"
        $response | ConvertTo-Json -Depth 10

        if ($response.data) {
            Write-Host "`n✅ Generated $($response.data.Count) edited image(s)" -ForegroundColor Green
            foreach ($img in $response.data) {
                if ($img.url) {
                    Write-Host "   Image URL: $BaseUrl$($img.url)" -ForegroundColor Cyan
                }
            }
        }
    } catch {
        Write-Host "❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "⚠️  Skipping: No test image found at '$testImagePath'" -ForegroundColor Yellow
}

# 5. List all available models
Write-Host "`n[5] Available Models..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
try {
    $response = Invoke-RestMethod -Uri "$BaseUrl/v1/models" -Method Get
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

# 6. Server statistics
Write-Host "`n[6] Server Statistics..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
try {
    $response = Invoke-RestMethod -Uri "$BaseUrl/stats" -Method Get
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n==============================================" -ForegroundColor Cyan
Write-Host "  Tips:" -ForegroundColor Cyan
Write-Host "  - View generated images: $BaseUrl/assets/" -ForegroundColor White
Write-Host "  - List all assets: $BaseUrl/debug/assets" -ForegroundColor White
Write-Host "  - API documentation: $BaseUrl/docs" -ForegroundColor White
Write-Host "  - Server stats: $BaseUrl/stats" -ForegroundColor White
Write-Host "==============================================" -ForegroundColor Cyan
