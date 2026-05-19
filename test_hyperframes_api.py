import requests
import time
import sys

# Change to your server URL if different
BASE_URL = "http://localhost:7860"

def test_fastapi_render():
    print("Testing FastAPI /hyperframes/render...")
    html_content = """
    <div style="background: linear-gradient(45deg, #ff9a9e 0%, #fad0c4 99%, #fad0c4 100%); width: 1280px; height: 720px; display: flex; align-items: center; justify-content: center; font-family: sans-serif;">
        <h1 style="color: white; font-size: 80px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);" data-hyperframes-animation="fadeIn">
            Hyperframes via Wan2GP API
        </h1>
    </div>
    <script>
        // Hyperframes expects window.__hf for coordination
        window.__hf = {
            duration: 3,
            seek: (t) => {
                console.log("Seeking to:", t);
            }
        };
    </script>
    """
    
    payload = {
        "html": html_content,
        "fps": 30,
        "format": "mp4"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/hyperframes/render", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"Queued successfully. Execution ID: {data['execution_id']}")
        
        execution_id = data['execution_id']
        while True:
            status_resp = requests.get(f"{BASE_URL}/render_status/{execution_id}")
            status_data = status_resp.json()
            status = status_data.get("status")
            progress = status_data.get("progress", 0)
            print(f"Status: {status} ({progress}%)")
            
            if status == "completed":
                print(f"Render completed! Output: {status_data['output_path']}")
                break
            elif status == "failed":
                print(f"Render failed: {status_data.get('error')}")
                break
            
            time.sleep(2)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_fastapi_render()
