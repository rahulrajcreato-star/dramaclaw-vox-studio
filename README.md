---
title: DramaClaw Master AI Studio - Vox & Anime Documentary Orchestrator
emoji: 🎬
colorFrom: indigo
colorTo: cyan
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
---

# 🎬 DramaClaw Master AI Studio
### *Autonomous Vox & Anime Documentary Pipeline with Gemini Multi-Key Mind & 20GB GPU ComfyUI*

DramaClaw is an industrialized, autonomous video production system engineered for YouTube Shorts, Reels, and cinematic investigative documentaries (1080x1920 9:16 vertical and 1920x1080 16:9 widescreen).

---

## 🌟 Key Highlights & Architectural Innovations

1. **🧠 Gemini Multi-Key Mind (Zero Rate Limits):**
   - Automatically load-balances across a 5-key Gemini pool (`PRIMARY`, `ALPHA_1`, `ALPHA_2`, `BETA_1`, `BETA_2`).
   - Acts as the director: deep story analysis, hook forensics, shot-by-shot storyboard matrix, and frame prompt generation.

2. **⚡ GridShare 20GB VRAM GPU Integration:**
   - Dedicated connection manager for GridShare remote cloud GPU instances (e.g., `relay.gridshare.in:30000`).
   - Automatic pre-flight inspector and 1-click auto-deployer for ComfyUI custom nodes and quantized models.
   - Strictly engineered to execute within **20GB VRAM** without CUDA OOM errors.

3. **🎨 Makoto Shinkai Color Transfer & 8-9 Step Keyframes:**
   - **Qwen Image 2.1 + Turbo LoRA (8-9 steps)** for high-resolution base keyframe creation.
   - **MKL LAB Color Transfer Node:** Enforces a 100% harmonious color grade matching the signature Makoto Shinkai twilight blues (`#0F2027`) and celestial amber highlights (`#F6D365`).

4. **🎬 Targeted Motion Animation (No Dizzying Zoom):**
   - **Targeted Motion Principle:** Architectural bedrock and geometry stay rock-solid; only selective focal elements (torch embers, sonar sweeps, laser scanlines, water ripples) animate.
   - **MiniMax H3 Omni-Modal Video Engine:** Exclusively used for neural targeted animation and synchronized 32kHz ambient foley bed. Pruned INT8 UNet + NVFP4 AWQ CLIP calibrated for 20GB GPUs.

5. **🎙️ Indic F5-TTS Conditioned with Gemini Fenrir Voice:**
   - High-fidelity Indic F5-TTS conditioned on Gemini's native *Fenrir* male voice sample for natural, authoritative documentary narration.
   - Integrated ambient foley sound design (-12dB ducked under dialogue).

6. **📱 Mobile-First Control & Push Notifications:**
   - Responsive dark-mode interface built for smartphone browsers (iOS Safari, Android Chrome).
   - Generates public `gradio.live` link (`share=True`) to control rendering from anywhere.
   - Real-time Telegram alerts sent to your phone with video previews and download links.

7. **☁️ Hugging Face Cloud Storage Persistence:**
   - Automatically synchronizes finished 1080p master documentaries directly to Hugging Face Datasets/Spaces.

---

## 🏗️ 7-Stage Autonomous Orchestra Pipeline

```
[ User Documentary Topic / Idea ]
                │
                ▼
  [ Stage 1: Story Analysis (Gemini Mind) ]
    - Hook forensics, psychological tension, pacing arc
                │
                ▼
  [ Stage 2: Storyboard Matrix Blueprint (Gemini Mind) ]
    - 6 Scenes: Lens, Shot Type, Phonetic Hindi Script, Stable vs Motion
                │
                ▼
  [ Stage 3: Frame Creation & Color Grading ]
    - Qwen Image 2.1 (8-9 Step Turbo) + MKL LAB Color Transfer
                │
                ▼
  [ Stage 4: Sequential Targeted Animation ]
    - MiniMax H3 Omni-Modal Engine (Strictly < 16GB VRAM)
                │
                ▼
  [ Stage 5: Voiceover & Audio Design ]
    - Indic F5-TTS (Gemini Fenrir Conditioned) + 32kHz Foley Bed
                │
                ▼
  [ Stage 6: Master Video Assembly ]
    - 1080p 30 FPS MP4 Remuxing (FFmpeg)
                │
                ▼
  [ Stage 7: Hugging Face Cloud Sync & Telegram Mobile Alert ]
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install the production dependencies:
```bash
git clone https://github.com/your-username/dramaclaw-studio.git
cd dramaclaw-studio
pip install -r requirements.txt
```

### 2. Configure API Keys (`.env`)
Create a `.env` file in the root directory (see `.env.example`):
```ini
# Multi-Key Gemini Pool
GEMINI_API_KEY=your_primary_gemini_key
GEMINI_API_KEY_ALPHA_1=your_alpha1_key
GEMINI_API_KEY_ALPHA_2=your_alpha2_key
GEMINI_API_KEY_BETA_1=your_beta1_key
GEMINI_API_KEY_BETA_2=your_beta2_key

# Hugging Face Cloud Storage
HF_TOKEN=hf_your_write_token
HF_STORAGE_REPO=your-username/documentary-masters

# GridShare Remote GPU
GRIDSHARE_HOST=relay.gridshare.in
GRIDSHARE_PORT=30000
GRIDSHARE_USER=root
COMFYUI_URL=http://127.0.0.1:8188

# Mobile Telegram Push Alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3. Launching the Studio
```bash
python app.py
```
- Open on Desktop: `http://localhost:7860`
- Open on Mobile: Scan the generated `https://xxxx.gradio.live` link on your smartphone!

---

## 📱 Mobile Control Workflow
1. Open the public Gradio link on your mobile phone.
2. In **Tab 1 (🎬 Gemini Story & Orchestra Studio)**, type your topic or select a project.
3. Tap **"🚀 3. Launch 1-Tap Production Orchestra"**.
4. You can lock your phone — the pipeline runs autonomously and sends you a Telegram message with the download link when the 1080p master video is uploaded to Hugging Face!

---

## 📄 License
MIT License. Created with ❤️ for autonomous AI cinema and investigative journalism.

