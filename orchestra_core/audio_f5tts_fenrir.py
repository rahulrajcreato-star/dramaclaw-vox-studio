import os
import sys
import wave
import asyncio
import subprocess
from pathlib import Path
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import GenerationConfig, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig
from orchestra_core.config import VALID_GEMINI_KEYS, get_next_gemini_key, FENRIR_REF_AUDIO, DEFAULT_GRIDSHARE_CONFIG

VOICE_MODES = [
    "Indic F5-TTS (Conditioned with Gemini Fenrir Voice)",
    "Gemini 2.5 Male - Fenrir (Deep Epic Narrator - Recommended)",
    "Gemini 2.5 Male - Puck (Fast Vox Documentary)",
    "Madhur (Edge-TTS Neutral Hindi)"
]

def synthesize_indic_f5_tts_remote(text: str, output_path: str, gridshare_cfg: dict = None) -> str:
    """
    Synthesizes narration via Indic F5-TTS conditioned on Gemini Fenrir reference voice on remote GPU.
    """
    cfg = gridshare_cfg or DEFAULT_GRIDSHARE_CONFIG
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        import paramiko
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=cfg.get("host", "relay.gridshare.in"),
            port=int(cfg.get("port", 30000)),
            username=cfg.get("user", "root"),
            key_filename=cfg.get("ssh_key_path"),
            timeout=8.0
        )
        
        # Clean text for python injection
        safe_text = text.replace('"', '\\"').replace("\n", " ")
        remote_script = f'''
import os
import torch
import soundfile as sf

ref_audio = "/workspace/gemini_ref_sample.wav"
ref_text = "2011 me jab Kerala ke Padmanabhaswamy Mandir ke gupt tehkhane khole gaye, toh poori duniya dang reh gayi!"
target_text = "{safe_text}"
out_file = "/workspace/f5_output_temp.wav"

try:
    from f5_tts.api import F5TTS
    f5 = F5TTS(model_type="F5-TTS_Base")
    wav, sr, _ = f5.infer(ref_file=ref_audio, ref_text=ref_text, gen_text=target_text)
    sf.write(out_file, wav, sr)
    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
'''
        stdin, stdout, stderr = client.exec_command(f'python3 -c \'{remote_script}\'')
        out_str = stdout.read().decode()
        if "SUCCESS" in out_str:
            sftp = client.open_sftp()
            sftp.get("/workspace/f5_output_temp.wav", output_path)
            sftp.close()
            client.close()
            return output_path
        client.close()
    except Exception as e:
        print(f"[AUDIO] Remote F5-TTS skipped/failed: {e}")

    # Seamless fallback to Gemini Fenrir
    return synthesize_gemini_fenrir_tts(text, output_path, voice_name="Fenrir")

def synthesize_gemini_fenrir_tts(text: str, output_path: str, voice_name: str = "Fenrir") -> str:
    """
    Synthesizes ultra-expressive documentary narration using Gemini's native voice model
    with round-robin failover across the 5 keys in the pool.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not VALID_GEMINI_KEYS:
        return synthesize_edge_tts(text, output_path)

    errors = []
    # Attempt with Gemini 2.5 Preview SpeechConfig
    for _ in range(len(VALID_GEMINI_KEYS)):
        name, key = get_next_gemini_key()
        if not key:
            continue
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("models/gemini-2.5-flash-preview-tts")
            sc = SpeechConfig(voice_config=VoiceConfig(prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=voice_name)))
            gc = GenerationConfig(response_modalities=["AUDIO"], speech_config=sc)
            resp = model.generate_content(text, generation_config=gc)
            pcm = resp.candidates[0].content.parts[0].inline_data.data

            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(pcm)
            return output_path
        except Exception as e:
            errors.append(f"{name}_2.5: {str(e)[:60]}")
            continue

    # Secondary attempt with Gemini 3.8 Flash TTS
    for _ in range(len(VALID_GEMINI_KEYS)):
        name, key = get_next_gemini_key()
        if not key:
            continue
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("models/gemini-3.8-flash-tts")
            prompt = f"Speak in an authoritative, deep, cinematic Indian documentary male narrator voice: {text}"
            resp = model.generate_content(prompt, generation_config={"response_modalities": ["AUDIO"]})
            pcm = resp.candidates[0].content.parts[0].inline_data.data

            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(pcm)
            return output_path
        except Exception as e:
            errors.append(f"{name}_3.8: {str(e)[:60]}")
            continue

    # Third fallback: Edge-TTS
    print(f"[AUDIO] Gemini TTS exhausted ({errors[:2]}), falling back to Edge-TTS")
    return synthesize_edge_tts(text, output_path)

def synthesize_edge_tts(text: str, output_path: str, voice: str = "hi-IN-MadhurNeural") -> str:
    """Fallback high-intelligibility edge TTS."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        import edge_tts
        async def _run():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
        asyncio.run(_run())
        return output_path
    except Exception as e:
        print(f"[AUDIO] Edge TTS failed: {e}")
        # Generate silence wave as absolute fallback
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00" * 48000)
        return output_path

def get_audio_duration_sec(audio_path: str) -> float:
    """Returns duration of an audio file in seconds."""
    if not os.path.exists(audio_path):
        return 0.0
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        try:
            with wave.open(audio_path, "rb") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            return 15.0

def mix_voice_and_ambient_foley(voice_path: str, ambient_path: str, output_mixed_path: str, ducking_db: float = -12.0) -> str:
    """
    Mixes voiceover narration with ambient background foley bed, applying -12dB ducking.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_mixed_path)), exist_ok=True)
    if not os.path.exists(ambient_path):
        return voice_path
        
    cmd = [
        "ffmpeg", "-y",
        "-i", voice_path,
        "-i", ambient_path,
        "-filter_complex",
        f"[1:a]volume={10**(ducking_db/20.0):.2f}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "[aout]",
        "-c:a", "aac", "-b:a", "192k",
        output_mixed_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return output_mixed_path
    except Exception as e:
        print(f"[AUDIO] Mixing error: {e}, using voice only")
        return voice_path
