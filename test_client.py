import os
import time
import uuid
import json
from gradio_client import Client, handle_file

# Configuration to match TypeScript client
GRADIO_SERVERS = [
    'http://127.0.0.1:7860',
]

class Wan2GPClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.client = Client(server_url)
        self.last_used_model = None

    def _unload_if_needed(self, model_family):
        """Logic to unload only if model family changes"""
        if self.last_used_model and not self.last_used_model.startswith(model_family):
            print(f"DEBUG: Unloading previous model {self.last_used_model} to switch to {model_family}")
            try:
                self.client.predict(api_name="/unload")
            except Exception as e:
                print(f"WARN: Unload failed: {e}")
        self.last_used_model = model_family

    def generate_image(self, params):
        model_type = "flux2_klein_9b"
        self._unload_if_needed("flux")
        
        # Protocol /generate (13 parameters)
        args = [
            model_type,                           # 1. model_type
            params.get("prompt", ""),             # 2. prompt
            params.get("num_inference_steps", 4), # 3. steps
            params.get("guidance_scale", 3.5),    # 4. guidance
            params.get("resolution", "704x1280"), # 5. resolution
            "0",                                  # 6. video_length (0 for images)
            params.get("seed", -1),               # 7. seed
            True,                                 # 8. image_mode (True for image)
            params.get("denoising_strength", 1.0),# 9. denoising_strength
            params.get("image_start"),            # 10. image_start (None or path)
            params.get("audio_input"),            # 11. audio_input
            params.get("override_profile", -1),   # 12. override_profile
            0.0                                   # 13. masking_strength
        ]
        
        # Ensure files are handled by gradio-client
        if args[9]: args[9] = handle_file(args[9])
        if args[10]: args[10] = handle_file(args[10])

        print(f"DEBUG: Calling /generate (Image) on {self.server_url}")
        result = self.client.predict(*args, api_name="/generate")
        return result

    def generate_video(self, params):
        model_type = params.get("model_type", "ltx2_22B_distilled")
        self._unload_if_needed("ltx") # Simple heuristic for model family
        
        # Helper for duration string
        def get_duration_string(v_len):
            if not v_len: return "5s"
            s = str(v_len)
            if s.endswith('s'): return s
            try:
                n = float(v_len)
                return f"{n}s"
            except: return "5s"

        # Protocol /generate (13 parameters)
        args = [
            model_type,                            # 1. model_type
            params.get("prompt", ""),              # 2. prompt
            params.get("num_inference_steps", 40), # 3. steps
            params.get("guidance_scale", 3.0),     # 4. guidance
            params.get("resolution", "704x1280"),  # 5. resolution
            get_duration_string(params.get("video_length")), # 6. video_length
            params.get("seed", -1),                # 7. seed
            False,                                 # 8. image_mode (False for video)
            params.get("denoising_strength", 0.55),# 9. denoising_strength
            params.get("image_start"),             # 10. image_start
            params.get("audio_input"),             # 11. audio_input
            params.get("override_profile", -1),    # 12. override_profile
            0.0                                    # 13. masking_strength
        ]
        
        if args[9]: args[9] = handle_file(args[9])
        if args[10]: args[10] = handle_file(args[10])

        print(f"DEBUG: Calling /generate (Video) on {self.server_url}")
        result = self.client.predict(*args, api_name="/generate")
        return result

    def render_video_remote(self, payload):
        print(f"DEBUG: Calling /render_video on {self.server_url}")
        result = self.client.predict(payload, api_name="/render_video")
        return result

    def get_render_status(self, execution_id):
        print(f"DEBUG: Calling /render_status for {execution_id}")
        result = self.client.predict(execution_id, api_name="/render_status")
        return result

    def unload_model(self):
        print(f"DEBUG: Calling /unload on {self.server_url}")
        return self.client.predict(api_name="/unload")

if __name__ == "__main__":
    client = Wan2GPClient(GRADIO_SERVERS[0])
    
    print("\n--- Testing /unload ---")
    print(client.unload_model())
    
    print("\n--- Testing /render_video (Mock Payload) ---")
    mock_payload = {
        "execution_id": str(uuid.uuid4()),
        "segments": [{"url": "http://example.com/test.mp4", "duration": 2}],
        "audio": {"music_url": "http://example.com/music.mp3", "music_volume": 0.1}
    }
    render_result = client.render_video_remote(mock_payload)
    print(f"Render Result: {render_result}")
    
    if "execution_id" in render_result:
        eid = render_result["execution_id"]
        print(f"\n--- Testing /render_status for {eid} ---")
        print(client.get_render_status(eid))

    print("\n--- Testing /generate (Image - Dry Run) ---")
    # Note: This will actually try to run inference. 
    # In a non-GPU environment it might fail but we test the connectivity.
    try:
        img_result = client.generate_image({
            "prompt": "Test prompt",
            "num_inference_steps": 1 # Fast for testing
        })
        print(f"Image Result: {img_result}")
    except Exception as e:
        print(f"Image Gen (expectedly) failed on execution but endpoint worked: {e}")

    print("\nClient tests completed.")
