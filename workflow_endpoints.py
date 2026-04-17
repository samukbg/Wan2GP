import os
import json
import tempfile
import subprocess
import requests
import uuid
import shutil
import time
from typing import List, Dict, Any
from fastapi import APIRouter, Request, BackgroundTasks
from fastapi.responses import JSONResponse

router = APIRouter()

# Global dictionary to store execution status
executions = {}

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
            
            # Normalize Video
            # scale, pad to target res, set fps
            vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}"
            
            cmd_v = ["ffmpeg", "-y", "-i", raw_path]
            
            # Trimming / Duration
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

            # Extract audio if sync_audio is true
            if seg.get("sync_audio"):
                cmd_a = ["ffmpeg", "-y", "-i", raw_path]
                if "trim_start" in seg or "trim_end" in seg:
                    start = seg.get("trim_start", 0)
                    cmd_a += ["-ss", str(start)]
                    if "trim_end" in seg:
                        cmd_a += ["-t", str(seg["trim_end"] - start)]
                elif "duration" in seg:
                    cmd_a += ["-t", str(seg["duration"])]
                
                # Check if file has audio
                info = get_media_info(raw_path)
                has_audio = any(s['codec_type'] == 'audio' for s in info.get('streams', []))
                
                if has_audio:
                    cmd_a += ["-vn", "-ac", "2", "-ar", "44100", processed_a_path]
                    subprocess.run(cmd_a, check=True, capture_output=True)
                    # We'll need to know the offset for this audio
                    # Since we are concatenating, the offset is the sum of previous segments' durations
                    offset = 0
                    for prev_i in range(i):
                        # We should ideally get the actual duration of processed segments
                        # But for now we assume they are exactly as requested
                        pass
                    # For now, let's just use a simpler approach: concatenate segments with audio in one go if possible
                    # But mpegts concat is easier for video.
                    # We'll handle segment audio by mixing it into the final audio track.
                else:
                    processed_a_path = None
            else:
                processed_a_path = None
            
            segment_audios.append(processed_a_path)
            executions[execution_id]["progress"] = int(30 * (i + 1) / len(segments))

        # 2. Concatenate Video Segments
        concat_video = os.path.join(temp_dir, "concat.mp4")
        if processed_segments:
            # Use mpegts concat for speed and reliability
            concat_cmd = ["ffmpeg", "-y", "-i", f"concat:{'|'.join(processed_segments)}", "-c", "copy", concat_video]
            subprocess.run(concat_cmd, check=True, capture_output=True)
        else:
            raise ValueError("No segments to render")
        
        executions[execution_id]["progress"] = 40

        # 3. Audio Mixing
        narration_url = audio_config.get("narration_url")
        music_url = audio_config.get("music_url")
        music_volume = audio_config.get("music_volume", 0.2)
        
        audio_inputs = []
        filter_complex = []
        
        # Input 0: Narration (optional)
        if narration_url:
            nar_path = os.path.join(temp_dir, "narration.mp3")
            download_file(narration_url, nar_path)
            audio_inputs += ["-i", nar_path]
            filter_complex.append(f"[0:a]volume=1.0[a_narr]")
        
        # Input 1: Music (optional)
        music_idx = len(audio_inputs)
        if music_url:
            mus_path = os.path.join(temp_dir, "music.mp3")
            download_file(music_url, mus_path)
            audio_inputs += ["-i", mus_path]
            filter_complex.append(f"[{music_idx}:a]volume={music_volume}[a_mus]")

        # Segment audios mixing (this is complex because they have timing offsets)
        # For simplicity, we'll assume narration is the primary track.
        # A better way would be to create an audio track for each segment and amix them.
        
        final_audio = os.path.join(temp_dir, "final_audio.aac")
        
        if filter_complex:
            # Join narration and music
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

        # 4. Captions and Final Pass
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
            
            # Escape for ffmpeg
            escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")
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
        executions[execution_id] = {"status": "completed", "progress": 100, "output_path": output_path}
        print(f"Render complete: {output_path}")

    except Exception as e:
        print(f"Render failed: {e}")
        executions[execution_id] = {"status": "failed", "error": str(e)}
    finally:
        # Cleanup temp files after some time or immediately
        # shutil.rmtree(temp_dir)
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

def setup_workflow_endpoints(app):
    app.include_router(router)
