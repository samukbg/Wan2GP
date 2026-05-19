from gradio_client import Client
import time
import json

# Remote server URL
SERVER_URL = "https://video-est-pc2.samuelbezerra.fr"

def test_remote_hyperframes():
    print(f"Connecting to {SERVER_URL}...")
    try:
        client = Client(SERVER_URL)
        
        # 1. Define the Hyperframes payload
        # This payload matches the expected dictionary format in render_hyperframes_gradio_api
        payload = {
            "html": """
            <div style="background: black; width: 1280px; height: 720px; display: flex; align-items: center; justify-content: center; font-family: 'Arial';">
                <div style="text-align: center;">
                    <h1 style="color: #00ffcc; font-size: 100px; margin-bottom: 20px;">HYPERFRAMES</h1>
                    <p style="color: white; font-size: 40px;">Remote Rendering Test</p>
                </div>
            </div>
            <script>
                // Hyperframes expects window.__hf for coordination
                window.__hf = {
                    duration: 3,
                    seek: (t) => {
                        // The rendering engine calls this to move to time 't' (in seconds)
                        // For simple CSS animations, this can often be empty or control a GSAP timeline
                        console.log("Seeking to:", t);
                    }
                };
            </script>
            """,
            "fps": 30,
            "format": "mp4",
            "quality": "standard"
        }

        print("\nStep 1: Calling /hyperframes_render...")
        # Note: Gradio client uses the api_name defined in wgp.py
        result = client.predict(
            json.dumps(payload), # Passing as JSON string to handle the dict/str check in wgp.py
            api_name="/hyperframes_render"
        )
        
        print(f"Response: {result}")
        
        if result.get("status") == "queued":
            execution_id = result["execution_id"]
            print(f"Job Queued! Execution ID: {execution_id}")
            
            # 2. Poll for status
            print("\nStep 2: Checking Status...")
            while True:
                status_result = client.predict(
                    execution_id,
                    api_name="/render_status"
                )
                
                status = status_result.get("status")
                progress = status_result.get("progress", 0)
                print(f"Current Status: {status} ({progress}%)")
                
                if status == "completed":
                    print(f"\nSUCCESS! Render finished.")
                    print(f"Output URL: {status_result.get('output_path') or status_result.get('output_url')}")
                    break
                elif status == "failed":
                    print(f"\nFAILED: {status_result.get('error')}")
                    break
                
                time.sleep(3)
        else:
            print(f"Error starting job: {result.get('error')}")

    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    test_remote_hyperframes()
