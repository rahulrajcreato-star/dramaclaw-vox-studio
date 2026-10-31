import os
import sys
import json
import re
import google.generativeai as genai
from orchestra_core.config import VALID_GEMINI_KEYS, get_next_gemini_key

# -----------------------------------------------------------------------------
# Director Persona & Aesthetic Prompt Directives
# -----------------------------------------------------------------------------
DIRECTOR_SYSTEM_INSTRUCTION = """
You are the Master AI Director & Storyboard Mind for DramaClaw & Vox Documentaries.
Your mandate is to craft viral, high-retention, aesthetically breathtaking short-form & long-form documentaries:
1. Aesthetic: Makoto Shinkai (Your Name / Suzume) meets Vox / Johnny Harris investigative journalism.
   - Twilight blues, celestial golden rims (#F6D365), deep indigo shadows (#0F2027), crisp volumetric lighting.
2. Language: Fluent, dramatic Hindi / Hinglish.
   - Phonetically clear for native text-to-speech without tongue-twisting run-ons or unnatural punctuation.
3. TARGETED MOTION PRINCIPLE: Absolutely NO dizzying, full-frame camera zoom-in/out.
   - For every single shot, you MUST specify:
     * STABLE ELEMENTS: Solid architecture, terrain, maps, rock, tables (remain 100% stable).
     * TARGETED MOTION: Selective focal animation only (torch flames, water ripples, laser scans, radar sweep, drifting particles, eye movement).
4. VRAM Constraint: Designed for sequential execution on 20GB VRAM GPUs (8-9 steps Turbo, clean latents).
"""

def query_gemini_with_failover(prompt: str, preferred_models=None):
    """
    Executes a prompt across the multi-key Gemini pool with automatic model fallback.
    """
    if preferred_models is None:
        preferred_models = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-3.8-flash", "gemini-pro-latest"]

    if not VALID_GEMINI_KEYS:
        return {"error": "No valid Gemini API keys found. Please check your .env or enter a key in Settings."}

    errors = []
    # Try all keys in pool
    for _ in range(len(VALID_GEMINI_KEYS)):
        name, key = get_next_gemini_key()
        if not key:
            continue
        try:
            genai.configure(api_key=key)
        except Exception as e:
            errors.append(f"{name}_config_err: {str(e)[:60]}")
            continue

        for model_name in preferred_models:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=DIRECTOR_SYSTEM_INSTRUCTION
                )
                response = model.generate_content(prompt)
                if response and response.text:
                    return {
                        "success": True,
                        "key_used": name,
                        "model_used": model_name,
                        "text": response.text.strip()
                    }
            except Exception as e:
                errors.append(f"{name}/{model_name}: {str(e)[:80]}")
                continue

    return {"error": "All Gemini API keys and models exhausted", "details": errors}

def analyze_story(topic: str, custom_angle: str = "") -> dict:
    """
    Stage 1: Deep Story & Narrative Analysis (The Mind's first step).
    """
    prompt = f"""
Analyze this story/documentary idea and build an executive narrative blueprint:
Topic: "{topic}"
Specific Angle/Hook: "{custom_angle or 'Vox investigative thriller style with historical and scientific curiosity'}"

Provide a structured briefing in markdown format covering:
1. **Curiosity Hook (0-5 seconds)**: The psychological trigger that stops scrolling.
2. **The Hidden Conflict / Mystery**: The core question to be investigated.
3. **Pacing & Emotional Arc**: How tension builds across scenes.
4. **Visual & Color Palette Direction**: Makoto Shinkai color choices for key visual themes.
5. **Audience Retention Strategy**: Mid-video reveal and climactic interactive closing question.
"""
    res = query_gemini_with_failover(prompt)
    if res.get("success"):
        return {"success": True, "analysis_md": res["text"], "model": res.get("model_used"), "key": res.get("key_used")}
    return {"success": False, "error": res.get("error"), "details": res.get("details", [])}

def generate_storyboard_matrix(topic: str, format_aspect: str = "9:16", style_preset: str = "Makoto Shinkai Vox", num_scenes: int = 6) -> dict:
    """
    Stage 2: Full Shot-by-Shot Storyboard Matrix (Previz Blueprint).
    """
    prompt = f"""
Create a comprehensive shot-by-shot Storyboard Matrix for a documentary on:
"{topic}"

Format: {format_aspect} (1080x1920 vertical reel or 1920x1080 horizontal)
Style Preset: {style_preset}
Scenes Required: Exactly {num_scenes} distinct scenes. Total runtime ~80-100 seconds.

You MUST respond strictly with a valid JSON array of objects.
Do NOT include markdown backticks around the JSON array, or if you do, ensure it is clean ```json [ ... ] ```.
Each scene object must include:
[
  {{
    "scene_num": 1,
    "timestamp_range": "00:00 - 15.00s",
    "duration_sec": 15.0,
    "shot_type": "28mm Wide Angle, Slow 2.5D Dolly-In",
    "narration_hindi": "Clean phonetic Hindi/Hinglish narration line for voiceover.",
    "visual_prompt": "Makoto Shinkai anime aesthetic, twilight cyan and golden lighting, detailed description...",
    "negative_prompt": "low quality, blurry, deformed limbs, text watermarks, plastic gloss, full screen camera zoom",
    "stable_elements": "Granite walls and golden pedestals remain rock-solid and stable",
    "targeted_motion": "torch_flames_and_smoke",
    "sound_design_cue": "Heavy stone grinding, metallic treasure chime, sub-bass pulse"
  }},
  ...
]
"""
    res = query_gemini_with_failover(prompt)
    if not res.get("success"):
        return {"success": False, "error": res.get("error"), "details": res.get("details", [])}

    raw = res.get("text", "")
    # Parse JSON
    json_match = re.search(r'\[\s*\{.*\}\s*\]', raw, re.DOTALL)
    if json_match:
        try:
            scenes = json.loads(json_match.group(0))
            return {
                "success": True,
                "scenes": scenes,
                "model_used": res.get("model_used"),
                "key_used": res.get("key_used")
            }
        except Exception as e:
            return {"success": False, "error": f"JSON parse error: {e}", "raw": raw}
    else:
        try:
            scenes = json.loads(raw)
            return {"success": True, "scenes": scenes, "model_used": res.get("model_used")}
        except Exception as e:
            return {"success": False, "error": f"Could not extract JSON array: {e}", "raw": raw}

def generate_frame_creation_plan(scene: dict) -> dict:
    """
    Stage 3: Generates precise prompt parameters for Qwen Image 2.1 + Turbo LoRA (8-9 steps) + Color Node.
    """
    v_prompt = scene.get("visual_prompt", "")
    t_motion = scene.get("targeted_motion", "")
    
    # Enrich prompt with Makoto Shinkai aesthetic keywords
    enriched_prompt = (
        f"{v_prompt}, masterpiece, 8k resolution, Makoto Shinkai lighting, "
        f"twilight oceanic cyan #0F2027, golden celestial rim light #F6D365, volumetric atmosphere, "
        f"clean anime lineart, hyper-detailed textures, 35mm film grain"
    )
    neg_prompt = scene.get("negative_prompt", (
        "blurry, ugly, distorted, lowres, text, watermark, logo, oversaturated plastic, "
        "dizzy zoom, flickering artifacts, oversaturated colors"
    ))
    
    return {
        "prompt": enriched_prompt,
        "negative_prompt": neg_prompt,
        "steps": 9,
        "cfg": 3.0,
        "denoise": 1.0,
        "lora_name": "qwen_turbo_8step_lora.safetensors",
        "lora_strength": 1.0,
        "color_transfer_method": "mkl_lab",
        "color_ref": "shinkai_roof_ref.jpg",
        "targeted_motion": t_motion
    }
