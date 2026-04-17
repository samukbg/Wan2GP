############# WanGP Spreadout API Server #############
import os, sys
import threading
import uuid
import json
import gradio as gr

# Add current directory to path
p = os.path.dirname(os.path.abspath(__file__))
if p not in sys.path:
    sys.path.insert(0, p)

# Import necessary functions from wgp
# This will trigger wgp's initialization but we only use the API handlers
from wgp import api_endpoint_handler, api_unload_handler, setup_workflow_endpoints
from workflow_endpoints import render_video_task, executions

def render_video_gradio_api(data):
    if isinstance(data, str):
        try: data = json.loads(data)
        except: return {"status": "failed", "error": "Invalid JSON string"}
    if not isinstance(data, dict):
        return {"status": "failed", "error": "Payload must be a dictionary"}
    
    execution_id = data.get("execution_id", str(uuid.uuid4()))
    output_path = os.path.join("outputs", f"render_{execution_id}.mp4")
    os.makedirs("outputs", exist_ok=True)
    
    threading.Thread(target=render_video_task, args=(data, output_path, execution_id), daemon=True).start()
    return {"status": "queued", "execution_id": execution_id, "output_url": f"/file={output_path}"}

def render_status_gradio_api(eid):
    return executions.get(eid, {"error": "Execution ID not found"})

css = """
#api_container { padding: 20px; }
"""

with gr.Blocks(title="WanGP API Server", css=css) as demo:
    gr.Markdown("# WanGP Spreadout API Server")
    gr.Markdown("This server provides a lightweight API interface for WanGP.")
    
    with gr.Group(elem_id="api_container"):
        # 1. Main Generation API
        with gr.Row():
            api_gen_model_type = gr.Textbox(label="Model Type")
            api_gen_prompt = gr.Textbox(label="Prompt")
        with gr.Row():
            api_gen_steps = gr.Number(label="Steps", value=4)
            api_gen_guidance = gr.Number(label="Guidance", value=3.5)
            api_gen_resolution = gr.Textbox(label="Resolution", value="704x1280")
            api_gen_length = gr.Textbox(label="Length", value="5s")
            api_gen_seed = gr.Number(label="Seed", value=-1)
        with gr.Row():
            api_gen_image_mode = gr.Checkbox(label="Image Mode")
            api_gen_denoising = gr.Number(label="Denoising", value=1.0)
            api_gen_image_start = gr.File(label="Image Start")
            api_gen_audio_input = gr.Audio(label="Audio Input", type="filepath")
            api_gen_override_profile = gr.Number(label="Override Profile", value=-1)
            api_gen_masking_strength = gr.Number(label="Masking Strength", value=0.0)
        
        api_gen_btn = gr.Button("Generate API")
        api_gen_output = gr.File(label="Output")

        api_gen_btn.click(
            fn=api_endpoint_handler,
            inputs=[
                api_gen_model_type, api_gen_prompt, api_gen_steps, api_gen_guidance,
                api_gen_resolution, api_gen_length, api_gen_seed, api_gen_image_mode,
                api_gen_denoising, api_gen_image_start, api_gen_audio_input,
                api_gen_override_profile, api_gen_masking_strength
            ],
            outputs=api_gen_output,
            api_name="generate"
        )

        # 2. Unload API
        api_unload_btn = gr.Button("Unload API")
        api_unload_output = gr.Textbox(label="Unload Status")
        api_unload_btn.click(fn=api_unload_handler, inputs=[], outputs=api_unload_output, api_name="unload")

        # 3. Render Video API
        api_render_payload = gr.JSON(label="Render Payload")
        api_render_btn = gr.Button("Render API")
        api_render_output = gr.JSON(label="Render Output")
        api_render_btn.click(fn=render_video_gradio_api, inputs=[api_render_payload], outputs=api_render_output, api_name="render_video")

        # 4. Render Status API
        api_status_id = gr.Textbox(label="Status ID")
        api_status_btn = gr.Button("Status API")
        api_status_output = gr.JSON(label="Status Output")
        api_status_btn.click(fn=render_status_gradio_api, inputs=[api_status_id], outputs=api_status_output, api_name="render_status")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WanGP Spreadout API Server")
    parser.add_argument("--listen", action="store_true", help="Listen on all interfaces (0.0.0.0)")
    parser.add_argument("--server-port", type=int, default=11435, help="Port to run the server on (default: 11435)")
    args = parser.parse_args()

    # Import app for workflow endpoints
    from wgp import app as wgp_app
    
    server_name = "0.0.0.0" if args.listen else "127.0.0.1"

    # Launch Gradio app
    demo.launch(
        server_name=server_name,
        server_port=args.server_port,
        show_api=True,
        prevent_thread_lock=True
    )
    
    # Mount custom workflow endpoints if needed (the client uses /gradio_api/predict mostly though)
    if demo.app:
        setup_workflow_endpoints(demo.app)
        print(f"API Server started with workflow endpoints")
    else:
        print("WARN: demo.app is None, workflow endpoints not mounted")
    
    demo.block_thread()
