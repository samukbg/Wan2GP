import asyncio
import base64
import os
import sys
from pathlib import Path
import edge_tts
import requests
import ormsgpack
from pydub import AudioSegment

WORKSPACE_DIR = Path(r"C:\Users\User\Documents\Wan2GP")
ARTIFACT_DIR = Path(r"C:\Users\User\.gemini\antigravity-cli\brain\765ce616-9e97-4e9c-8337-422035f76b0a")

VOICES_CONFIG = {
    "female": {
        "edge_voice": "pt-BR-FranciscaNeural",
        "ref_text": "Olá, eu sou a voz feminina em português do Brasil. Estou pronta para narrar suas histórias com emoção, clareza e ritmo natural.",
        "ref_wav": WORKSPACE_DIR / "ref_pt_br_female.wav",
        "ref_mp3": WORKSPACE_DIR / "ref_pt_br_female.mp3",
    },
    "male": {
        "edge_voice": "pt-BR-AntonioNeural",
        "ref_text": "Olá, eu sou a voz masculina em português do Brasil. Estou pronto para narrar suas histórias com presença, energia e impacto.",
        "ref_wav": WORKSPACE_DIR / "ref_pt_br_male.wav",
        "ref_mp3": WORKSPACE_DIR / "ref_pt_br_male.mp3",
    }
}

async def ensure_reference_voices():
    """Generates reference WAV files for both female and male Brazilian Portuguese voices."""
    for gender, cfg in VOICES_CONFIG.items():
        if not cfg["ref_wav"].exists():
            print(f"Creating reference audio for {gender} voice ({cfg['edge_voice']})...")
            communicate = edge_tts.Communicate(cfg["ref_text"], voice=cfg["edge_voice"])
            await communicate.save(cfg["ref_mp3"])
            
            sound = AudioSegment.from_mp3(cfg["ref_mp3"])
            sound = sound.set_frame_rate(44100).set_channels(1)
            sound.export(cfg["ref_wav"], format="wav")
            
            # Clean up temporary mp3
            if cfg["ref_mp3"].exists():
                os.remove(cfg["ref_mp3"])
            print(f"Saved reference WAV: {cfg['ref_wav']}")

def generate_tts(text: str, voice_type: str = "female", output_filename: str = "output.wav") -> Path:
    """
    Synthesizes Brazilian Portuguese audio using Fish-Speech zero-shot cloning.
    voice_type: 'female' or 'male'
    """
    if voice_type not in VOICES_CONFIG:
        raise ValueError(f"Unknown voice_type '{voice_type}'. Choose 'female' or 'male'.")
    
    asyncio.run(ensure_reference_voices())
    cfg = VOICES_CONFIG[voice_type]
    
    with open(cfg["ref_wav"], "rb") as f:
        ref_audio_bytes = f.read()
        
    url = "http://127.0.0.1:8080/v1/tts"
    payload = {
        "text": text,
        "format": "wav",
        "streaming": False,
        "normalize": True,
        "latency": "normal",
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "temperature": 0.7,
        "references": [
            {
                "audio": ref_audio_bytes,
                "text": cfg["ref_text"],
            }
        ]
    }
    
    print(f"\nGenerating {voice_type.upper()} pt-BR narration...")
    print(f"Text: {text}\n")
    
    packed = ormsgpack.packb(payload)
    headers = {"Content-Type": "application/msgpack"}
    
    response = requests.post(url, data=packed, headers=headers, timeout=300)
    
    if response.status_code == 200:
        out_workspace = WORKSPACE_DIR / output_filename
        out_artifact = ARTIFACT_DIR / output_filename
        
        with open(out_workspace, "wb") as f:
            f.write(response.content)
        with open(out_artifact, "wb") as f:
            f.write(response.content)
            
        print(f"SUCCESS! Saved {voice_type} audio to: {out_workspace}")
        return out_workspace
    else:
        raise RuntimeError(f"Fish-Speech API Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    script_female = (
        "(empolgada) Seja bem-vindo ao nosso espetáculo! Hoje nós vamos revelar segredos fantásticos "
        "que permaneceram ocultos por séculos na nossa história brasileira! (sussurro) Prepare o seu coração!"
    )
    script_male = (
        "(empolgado) Atenção todos! Vocês não vão acreditar no que acabou de acontecer! "
        "(suspiro) Uma descoberta simplesmente inacreditável na floresta amazônica! "
        "(emocionado) É algo surreal e verdadeiramente mágico!"
    )
    
    print("=== Generating Both Female & Male pt-BR Narration Samples ===")
    generate_tts(script_female, voice_type="female", output_filename="portuguese_narration_female.wav")
    generate_tts(script_male, voice_type="male", output_filename="portuguese_narration_male.wav")
