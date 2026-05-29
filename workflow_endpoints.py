import os
import json
import tempfile
import subprocess
import requests
import uuid
import shutil
import time
import platform
import sys
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from shared.ffmpeg_setup import download_ffmpeg

router = APIRouter()

# Global dictionary to store execution status
executions = {}

def get_npx_command():
    """Robustly find the npx command for the current platform."""
    # Check common Windows variations first if we detect Windows
    if os.name == "nt" or sys.platform == "win32" or platform.system() == "Windows":
        for cmd in ["npx.cmd", "npx.bat", "npx"]:
            if shutil.which(cmd):
                return cmd
        return "npx.cmd" # default for windows
    
    # Non-Windows
    if shutil.which("npx"):
        return "npx"
    return "npx"

def ensure_hyperframes_env():
    """Ensure ffmpeg and chrome are available for hyperframes."""
    # Ensure ffmpeg is on PATH
    download_ffmpeg()
    
    # Check if chrome is installed for hyperframes
    def _ensure():
        try:
            print("[Hyperframes] Ensuring browser environment...")
            cmd = get_npx_command()
            use_shell = (os.name == "nt")
            subprocess.run([cmd, "-y", "hyperframes", "browser", "ensure"], check=True, capture_output=True, shell=use_shell)
            print("[Hyperframes] Browser environment ready.")
        except Exception as e:
            print(f"[Hyperframes] Warning during browser ensure: {e}")
    
    import threading
    threading.Thread(target=_ensure, daemon=True).start()

def download_file(url: str, dest: str):
    print(f"Downloading {url} to {dest}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest

def get_media_info(path: str) -> Dict[str, Any]:
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type',
        '-of', 'json', path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

def render_video_task(data: Dict[str, Any], output_path: str, execution_id: str):
    executions[execution_id] = {"status": "processing", "progress": 0}
    
    segments = data.get("segments", [])
    audio_config = data.get("audio", {})
    captions_config = data.get("captions", {})
    output_format = data.get("output_format", {})

    res = output_format.get("resolution", "720x1280")
    fps = output_format.get("fps", 24)
    codec = output_format.get("codec", "libx264")
    width, height = map(int, res.split('x'))

    temp_dir = tempfile.mkdtemp(prefix=f"render_{execution_id}_")
    processed_segments = []
    segment_audios = []

    try:
        # 1. Process Segments
        for i, seg in enumerate(segments):
            url = seg.get("url")
            if not url: continue
            
            raw_path = os.path.join(temp_dir, f"raw_seg_{i}.mp4")
            download_file(url, raw_path)
            
            processed_v_path = os.path.join(temp_dir, f"proc_seg_{i}.ts")
            processed_a_path = os.path.join(temp_dir, f"audio_seg_{i}.wav")
            
            vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}"
            
            cmd_v = ["ffmpeg", "-y", "-i", raw_path]
            
            if "trim_start" in seg or "trim_end" in seg:
                start = seg.get("trim_start", 0)
                cmd_v += ["-ss", str(start)]
                if "trim_end" in seg:
                    cmd_v += ["-t", str(seg["trim_end"] - start)]
            elif "duration" in seg:
                cmd_v += ["-t", str(seg["duration"])]
            
            cmd_v += [
                "-vf", vf,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-an", "-f", "mpegts", processed_v_path
            ]
            subprocess.run(cmd_v, check=True, capture_output=True)
            processed_segments.append(processed_v_path)

            if seg.get("sync_audio"):
                cmd_a = ["ffmpeg", "-y", "-i", raw_path]
                if "trim_start" in seg or "trim_end" in seg:
                    start = seg.get("trim_start", 0)
                    cmd_a += ["-ss", str(start)]
                    if "trim_end" in seg:
                        cmd_a += ["-t", str(seg["trim_end"] - start)]
                elif "duration" in seg:
                    cmd_a += ["-t", str(seg["duration"])]
                
                info = get_media_info(raw_path)
                has_audio = any(s['codec_type'] == 'audio' for s in info.get('streams', []))
                
                if has_audio:
                    cmd_a += ["-vn", "-ac", "2", "-ar", "44100", processed_a_path]
                    subprocess.run(cmd_a, check=True, capture_output=True)
                else:
                    processed_a_path = None
            else:
                processed_a_path = None
            
            segment_audios.append(processed_a_path)
            executions[execution_id]["progress"] = int(30 * (i + 1) / len(segments))

        concat_video = os.path.join(temp_dir, "concat.mp4")
        if processed_segments:
            concat_cmd = ["ffmpeg", "-y", "-i", f"concat:{'|'.join(processed_segments)}", "-c", "copy", concat_video]
            subprocess.run(concat_cmd, check=True, capture_output=True)
        else:
            raise ValueError("No segments to render")
        
        executions[execution_id]["progress"] = 40

        narration_url = audio_config.get("narration_url")
        music_url = audio_config.get("music_url")
        music_volume = audio_config.get("music_volume", 0.2)
        
        audio_inputs = []
        filter_complex = []
        
        if narration_url:
            nar_path = os.path.join(temp_dir, "narration.mp3")
            download_file(narration_url, nar_path)
            audio_inputs += ["-i", nar_path]
            filter_complex.append(f"[0:a]volume=1.0[a_narr]")
        
        music_idx = len(audio_inputs)
        if music_url:
            mus_path = os.path.join(temp_dir, "music.mp3")
            download_file(music_url, mus_path)
            audio_inputs += ["-i", mus_path]
            filter_complex.append(f"[{music_idx}:a]volume={music_volume}[a_mus]")

        final_audio = os.path.join(temp_dir, "final_audio.aac")
        
        if filter_complex:
            inputs_to_mix = []
            if narration_url: inputs_to_mix.append("[a_narr]")
            if music_url: inputs_to_mix.append("[a_mus]")
            
            mix_str = "".join(inputs_to_mix)
            mix_str += f"amix=inputs={len(inputs_to_mix)}:duration=first:dropout_transition=2[outa]"
            
            audio_cmd = ["ffmpeg", "-y"] + audio_inputs + ["-filter_complex", ";".join(filter_complex) + ";" + mix_str, "-map", "[outa]", "-c:a", "aac", "-b:a", "128k", final_audio]
            subprocess.run(audio_cmd, check=True, capture_output=True)
        else:
            final_audio = None

        executions[execution_id]["progress"] = 70

        srt_content = captions_config.get("srt_content")
        font_style = captions_config.get("font_style", "")
        
        final_cmd = ["ffmpeg", "-y", "-i", concat_video]
        if final_audio:
            final_cmd += ["-i", final_audio]
            
        video_filters = []
        if captions_config.get("enabled") and srt_content:
            srt_path = os.path.join(temp_dir, "captions.srt")
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            escaped_srt = srt_path.replace("\\", "/").replace(":", "\\\\:")
            sub_filter = f"subtitles='{escaped_srt}'"
            if font_style:
                sub_filter += f":force_style='{font_style}'"
            video_filters.append(sub_filter)
            
        if video_filters:
            final_cmd += ["-vf", ",".join(video_filters)]
            
        if final_audio:
            final_cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"]
        else:
            final_cmd += ["-c:a", "copy"]
            
        final_cmd += ["-c:v", codec, "-r", str(fps), output_path]
        
        subprocess.run(final_cmd, check=True, capture_output=True)
        executions[execution_id] = {"status": "completed", "progress": 100, "output_path": output_path, "output_url": f"/file={output_path}"}
        print(f"Render complete: {output_path}")

    except Exception as e:
        print(f"Render failed: {e}")
        executions[execution_id] = {"status": "failed", "error": str(e)}
    finally:
        pass

@router.post("/render_video")
async def render_video(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    execution_id = data.get("execution_id", str(uuid.uuid4()))
    output_filename = f"render_{execution_id}.mp4"
    output_path = os.path.join("outputs", output_filename)
    os.makedirs("outputs", exist_ok=True)
    
    background_tasks.add_task(render_video_task, data, output_path, execution_id)
    
    return {
        "status": "queued",
        "execution_id": execution_id,
        "output_url": f"/file={output_path}"
    }

@router.get("/render_status/{execution_id}")
async def get_render_status(execution_id: str):
    status = executions.get(execution_id)
    if not status:
        return JSONResponse({"error": "Execution ID not found"}, status_code=404)
    return status

def render_hyperframes_task(data: Dict[str, Any], output_path: str, execution_id: str):
    executions[execution_id] = {"status": "processing", "progress": 0}
    
    html_content = data.get("html")
    html_url = data.get("html_url")
    files = data.get("files", {})
    fps = data.get("fps", 30)
    quality = data.get("quality", "standard")
    format = data.get("format", "mp4")
    
    temp_dir = tempfile.mkdtemp(prefix=f"hyperframes_{execution_id}_")
    
    try:
        ensure_hyperframes_env()
        executions[execution_id]["progress"] = 5
        
        index_path = os.path.join(temp_dir, "index.html")
        if html_url:
            download_file(html_url, index_path)
        elif html_content:
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        else:
            if "index.html" not in files:
                raise ValueError("Either 'html', 'html_url', or 'index.html' in 'files' must be provided")
            
        executions[execution_id]["progress"] = 10
        
        tsx_content = data.get("tsx")
        if tsx_content:
            with open(os.path.join(temp_dir, "Component.tsx"), "w", encoding="utf-8") as f:
                f.write(tsx_content)


        for filename, content in files.items():
            dest_path = os.path.join(temp_dir, filename)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            if isinstance(content, str) and (content.startswith("http://") or content.startswith("https://")):
                download_file(content, dest_path)
            else:
                mode = 'w' if isinstance(content, str) else 'wb'
                encoding = 'utf-8' if isinstance(content, str) else None
                with open(dest_path, mode, encoding=encoding) as f:
                    f.write(content)
        
        executions[execution_id]["progress"] = 20
        
        npx_executable = get_npx_command()
        cmd = [
            npx_executable, "-y", "hyperframes", "render", 
            temp_dir,
            "-o", os.path.abspath(output_path),
            "--fps", str(fps),
            "--quality", quality,
            "--format", format,
            "--quiet"
        ]
        
        print(f"[Hyperframes] Running: {' '.join(cmd)}")
        use_shell = (os.name == "nt")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=use_shell)
        
        for line in process.stdout:
            print(f"[Hyperframes] {line.strip()}")
            if "[INFO] Compiled" in line:
                executions[execution_id]["progress"] = 40
            
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"Hyperframes render failed with exit code {process.returncode}")
            
        executions[execution_id] = {"status": "completed", "progress": 100, "output_path": output_path, "output_url": f"/file={output_path}"}
        print(f"Hyperframes render complete: {output_path}")

    except Exception as e:
        print(f"Hyperframes render failed: {e}")
        executions[execution_id] = {"status": "failed", "error": str(e)}
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

def hyperframes_tts_task(data: Dict[str, Any], output_path: str, execution_id: str):
    executions[execution_id] = {"status": "processing", "progress": 0}
    text = data.get("text", "")
    voice = data.get("voice", "af_heart")
    speed = data.get("speed", 1.0)
    
    LOCAL_TTS_URL = "http://localhost:5556/v1/audio/speech"
    API_TOKEN = "kok_4xK9mP2nQ7wR5vL8jH3fN6yT1sZ0uB4cE2dA9gM7pV5iO8qW3xJ6nK1rY4tU"
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        executions[execution_id]["progress"] = 20
        print(f"[Hyperframes] Calling local TTS with auth: {LOCAL_TTS_URL}")
        
        payload = {
            "model": "kokoro",
            "input": text,
            "voice": voice,
            "speed": speed
        }
        
        try:
            response = requests.post(LOCAL_TTS_URL, json=payload, headers=headers, timeout=60)
        except requests.exceptions.ConnectionError:
            print(f"[Hyperframes] {LOCAL_TTS_URL} connection failed, trying /tts...")
            response = requests.post("http://localhost:5556/tts", json={
                "text": text,
                "voice": voice,
                "speed": speed
            }, headers=headers, timeout=60)

        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            executions[execution_id] = {"status": "completed", "progress": 100, "output_path": output_path, "output_url": f"/file={output_path}"}
            print(f"[Hyperframes] TTS Complete: {output_path}")
        else:
            raise RuntimeError(f"Local TTS failed with status {response.status_code}: {response.text}")

    except Exception as e:
        print(f"[Hyperframes] TTS Error: {e}")
        executions[execution_id] = {"status": "failed", "error": str(e)}

def hyperframes_transcribe_task(data: Dict[str, Any], input_path: str, execution_id: str):
    executions[execution_id] = {"status": "processing", "progress": 0}
    model = data.get("model", "small.en")
    
    try:
        temp_dir = tempfile.mkdtemp()
        npx_executable = get_npx_command()
        cmd = [
            npx_executable, "-y", "hyperframes", "transcribe",
            os.path.abspath(input_path),
            "--dir", temp_dir,
            "-m", model,
            "--json"
        ]
        
        executions[execution_id]["progress"] = 10
        use_shell = (os.name == "nt")
        process = subprocess.run(cmd, capture_output=True, text=True, shell=use_shell)
        
        if process.returncode != 0:
            raise RuntimeError(f"Hyperframes Transcribe failed: {process.stderr}")
            
        transcript_file = None
        for f in os.listdir(temp_dir):
            if f.endswith(".json"):
                transcript_file = os.path.join(temp_dir, f)
                break
        
        if not transcript_file:
            raise RuntimeError("No transcript JSON generated")
            
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
            
        executions[execution_id] = {
            "status": "completed", 
            "progress": 100, 
            "result": transcript_data
        }
    except Exception as e:
        executions[execution_id] = {"status": "failed", "error": str(e)}
    finally:
        try: shutil.rmtree(temp_dir)
        except: pass

@router.post("/hyperframes/render")
async def hyperframes_render(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    execution_id = data.get("execution_id", str(uuid.uuid4()))
    fmt = data.get("format", "mp4")
    output_filename = f"hyper_{execution_id}.{fmt}"
    output_path = os.path.join("outputs", output_filename)
    os.makedirs("outputs", exist_ok=True)
    
    background_tasks.add_task(render_hyperframes_task, data, output_path, execution_id)
    
    return {
        "status": "queued",
        "execution_id": execution_id,
        "output_url": f"/file={output_path}"
    }

@router.post("/hyperframes/tts")
async def hyperframes_tts(request: Request, background_tasks: BackgroundTasks):
    try: data = await request.json()
    except: return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    execution_id = data.get("execution_id", str(uuid.uuid4()))
    output_path = os.path.join("outputs", f"tts_{execution_id}.wav")
    os.makedirs("outputs", exist_ok=True)
    
    background_tasks.add_task(hyperframes_tts_task, data, output_path, execution_id)
    
    return {"status": "queued", "execution_id": execution_id}

@router.post("/hyperframes/transcribe")
async def hyperframes_transcribe(request: Request, background_tasks: BackgroundTasks):
    try: data = await request.json()
    except: return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    input_url = data.get("url")
    if not input_url: return JSONResponse({"error": "Missing 'url'"}, status_code=400)
    
    execution_id = data.get("execution_id", str(uuid.uuid4()))
    temp_input = os.path.join(tempfile.gettempdir(), f"transcribe_{execution_id}")
    
    def transcribe_flow():
        try:
            download_file(input_url, temp_input)
            hyperframes_transcribe_task(data, temp_input, execution_id)
        finally:
            if os.path.exists(temp_input): os.remove(temp_input)
            
    background_tasks.add_task(transcribe_flow)
    
    return {"status": "queued", "execution_id": execution_id}

def setup_workflow_endpoints(app):
    ensure_hyperframes_env()
    app.include_router(router)
