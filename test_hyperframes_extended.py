import requests
import time
import os
import json

# Change to your server URL if different
BASE_URL = "https://video-est-pc2.samuelbezerra.fr"

def poll_status(execution_id):
    print(f"Polling status for {execution_id}...")
    while True:
        try:
            # Using the /render_status/{id} endpoint (FastAPI)
            # Or we could use the Gradio predict API, but FastAPI is simpler for raw scripts
            resp = requests.get(f"{BASE_URL}/render_status/{execution_id}")
            data = resp.json()
            status = data.get("status")
            progress = data.get("progress", 0)
            print(f"  Status: {status} ({progress}%)")
            
            if status == "completed":
                return data
            if status == "failed":
                print(f"  Error: {data.get('error')}")
                return data
        except Exception as e:
            print(f"  Polling error: {e}")
        
        time.sleep(3)

def test_tts():
    print("\n--- Testing Hyperframes TTS ---")
    payload = {
        "text": "Hello! This is a test of the Hyperframes text to speech engine running on the Wan2GP server.",
        "voice": "af_sky",
        "speed": 1.0
    }
    resp = requests.post(f"{BASE_URL}/hyperframes/tts", json=payload)
    print(f"Response: {resp.status_code} - {resp.text}")
    execution_id = resp.json()["execution_id"]
    result = poll_status(execution_id)
    if result.get("status") == "completed":
        print(f"TTS Success! Output: {result.get('output_path')}")
        return result.get('output_path')

def test_render_with_assets():
    print("\n--- Testing Hyperframes Render with Assets ---")
    # We'll send an index.html and a CSS file as assets
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="style.css">
    </head>
    <body>
        <div class="container">
            <h1 id="text">Asset Test</h1>
        </div>
        <script>
            window.__hf = {
                duration: 2,
                seek: (t) => {
                    const el = document.getElementById('text');
                    el.style.transform = `scale(${1 + t})`;
                    el.style.opacity = 1 - (t/2);
                }
            };
        </script>
    </body>
    </html>
    """
    
    css_content = """
    body { background: #1a1a1a; margin: 0; display: flex; align-items: center; justify-content: center; height: 720px; width: 1280px; }
    .container { border: 5px solid #00ffcc; padding: 50px; border-radius: 20px; }
    h1 { color: #00ffcc; font-family: sans-serif; font-size: 80px; }
    """
    
    payload = {
        "html": html_content,
        "files": {
            "style.css": css_content
        },
        "fps": 30,
        "format": "mp4"
    }
    
    resp = requests.post(f"{BASE_URL}/hyperframes/render", json=payload)
    execution_id = resp.json()["execution_id"]
    result = poll_status(execution_id)
    if result.get("status") == "completed":
        print(f"Render Success! Output: {result.get('output_path')}")

if __name__ == "__main__":
    # Note: Transcription test requires a valid URL to an audio/video file
    # We'll skip it for now or use the TTS output if we were running locally.
    # On remote, we can't easily pass the TTS output path back to transcribe without a full URL.
    
    test_tts()
    test_render_with_assets()
