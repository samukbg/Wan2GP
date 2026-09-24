import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from gradio_client import Client, handle_file

SERVER_URL = os.environ.get("WAN2GP_SERVER_URL", "http://127.0.0.1:11435")

def run_test(server_url=SERVER_URL, product_img=None, user_img=None):
    print(f"Connecting to Gradio server at {server_url}...")
    client = Client(server_url)

    # Prepare two test images
    # Image 0 (Garment reference):
    product_image_path = product_img or r"C:\Users\User\AppData\Local\Temp\gradio\1f69a440c8f68b3d5205b483e4518f8d951523a9eac46285a7ce07b77b987244\fece49e6-7403-46e7-a076-b17b5c5d21f6_00_garment.png"
    # Image 1 (User / identity reference):
    image_path = user_img or r"C:\Users\User\AppData\Local\Temp\gradio\d12dfb46e11406b5b1be70989c05b8afd5484ba7260d452097237ff70af01edd\fece49e6-7403-46e7-a076-b17b5c5d21f6_01_identity.jpg"

    if not os.path.exists(product_image_path):
        product_image_path = os.path.abspath("spread-out/app/playwright/test.jpg")
    if not os.path.exists(image_path):
        image_path = product_image_path

    print(f"Image 0 (Garment): {product_image_path}")
    print(f"Image 1 (Person): {image_path}")

    user_handle = handle_file(image_path)
    product_handle = handle_file(product_image_path) if product_image_path and os.path.exists(product_image_path) else user_handle

    input_images = [product_handle, user_handle]
    resolution = "816x1248"

    prompt = (
        "HIGH-FIDELITY VIRTUAL TRY-ON — Image 0=GARMENT, Image 1=PERSON — "
        "perform an exact 1:1 identity transfer from Image 1; accurately drape Image 0 onto subject; "
        "FULL-BODY luxury commercial fashion photography, cinematic studio lighting, photorealistic."
    )

    print(f"LOG: [IMAGE] Using resolution: {resolution}", flush=True)
    print(f"LOG: [IMAGE] Sending prompt: {prompt}", flush=True)
    print("Calling /wan2gp_generate on server...", flush=True)

    result = client.predict(
        "qwen_image_21_7B",           # 0: model_type
        prompt,                       # 1: prompt
        25,                           # 2: num_inference_steps
        1.0,                          # 3: guidance_scale
        resolution,                   # 4: resolution
        "1",                          # 5: video_length (string "1")
        -1,                           # 6: seed
        True,                         # 7: image_mode
        0.7,                          # 8: denoising_strength
        input_images,                 # 9: image_start (List of handles)
        None,                         # 10: image_end
        user_handle,                  # 11: audio_input (Identity Reference handle)
        4,                            # 12: override_profile (Set to 4)
        0.0,                          # 13: masking_strength
        0,                            # 14: sliding_window_size
        "TIK",                        # 15: prompt_enhancer
        "subtitles, text, watermark, logo, deformed, blurry, ugly, bad anatomy", # 16: negative_prompt
        api_name="/wan2gp_generate"
    )

    print(f"SUCCESS: Generation completed! Result file: {result}", flush=True)
    return result

if __name__ == "__main__":
    run_test()
