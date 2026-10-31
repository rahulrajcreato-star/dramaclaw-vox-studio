import os
import sys

# Force UTF-8 encoding for Windows terminal
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import json
import glob
from pathlib import Path
import gradio as gr

# Ensure root workspace is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from orchestra_core.config import (
    VALID_GEMINI_KEYS,
    HF_TOKEN,
    HF_DEFAULT_DATASET,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    DEFAULT_GRIDSHARE_CONFIG,
    STATIC_DIR,
    OUTPUT_DIR,
    FENRIR_REF_AUDIO,
    F5_CONDITIONED_AUDIO,
    update_gemini_key
)
from orchestra_core.gemini_director import analyze_story, generate_storyboard_matrix
from orchestra_core.gridshare_manager import GridShareManager
from orchestra_core.audio_f5tts_fenrir import synthesize_gemini_fenrir_tts, VOICE_MODES
from orchestra_core.master_orchestrator import run_autonomous_orchestra_pipeline
from orchestra_core.hf_storage_manager import HuggingFaceStorageManager
from orchestra_core.mobile_notifier import send_telegram_alert

# Read custom mobile CSS
css_file = STATIC_DIR / "custom_mobile.css"
custom_css = css_file.read_text(encoding="utf-8") if css_file.exists() else ""

gridshare_mgr = GridShareManager()
hf_mgr = HuggingFaceStorageManager()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_available_completed_videos():
    vids = []
    # Check orchestra_outputs and historical vox projects
    search_dirs = [
        BASE_DIR / "orchestra_outputs",
        BASE_DIR / "vox_video_01_padmanabhaswamy_vault" / "output",
        BASE_DIR / "vox_video_02_siliguri_chokepoint" / "output",
        BASE_DIR / "vox_video_03_kohinoor_peacock_throne_16x9" / "output",
        BASE_DIR / "vox_video_04_vanishing_saraswati_9x16" / "output",
        BASE_DIR / "vox_video_05_pokhran_cia_satellites_16x9" / "output",
        BASE_DIR / "vox_video_06_north_sentinel_island_9x16" / "output",
        BASE_DIR / "vox_video_07_paper_cutout_natwarlal_con" / "output",
        BASE_DIR / "vox_video_08_great_nicobar_geography_9x16" / "output",
        BASE_DIR / "vox_video_09_battle_of_plassey_history_9x16" / "output",
        BASE_DIR / "india_shorts_documentary" / "output",
    ]
    for d in search_dirs:
        if d.exists():
            for f in d.glob("**/*.mp4"):
                name = f.stem.replace("_", " ").title()
                size_mb = f.stat().st_size / (1024 * 1024)
                vids.append((f"{name} ({size_mb:.1f} MB)", str(f)))
    return vids

# -----------------------------------------------------------------------------
# Gradio Interface Callbacks
# -----------------------------------------------------------------------------
def handle_story_analysis(topic: str, angle: str):
    if not topic.strip():
        return "❌ Please enter a documentary topic."
    res = analyze_story(topic.strip(), angle.strip())
    if res.get("success"):
        header = f"### 🧠 Gemini Mind Story Blueprint: `{topic}`\n"
        header += f"*Model: `{res.get('model')}` | Key: `{res.get('key')}`*\n\n"
        return header + res.get("analysis_md", "")
    return f"❌ Analysis Error: {res.get('error')}\n{res.get('details', [])}"

def handle_storyboard_generation(topic: str, aspect: str, style: str):
    if not topic.strip():
        return "❌ Please enter a topic.", ""
    res = generate_storyboard_matrix(topic.strip(), format_aspect=aspect, style_preset=style, num_scenes=6)
    if res.get("success"):
        scenes = res.get("scenes", [])
        table_md = f"### 🎬 DramaClaw Storyboard Blueprint: {topic}\n"
        table_md += f"**Model:** `{res.get('model_used')}` | **Key:** `{res.get('key_used')}`\n\n"
        table_md += "| Scene | Duration | Shot Type | Narration (Phonetic Hindi) | Targeted Motion | Sound FX |\n"
        table_md += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        for s in scenes:
            table_md += f"| **Shot {s.get('scene_num')}** | `{s.get('duration_sec', 15)}s` | {s.get('shot_type')} | {s.get('narration_hindi')} | **{s.get('targeted_motion')}** | {s.get('sound_design_cue')} |\n"
        return table_md, json.dumps(scenes, indent=2, ensure_ascii=False)
    return f"❌ Error: {res.get('error')}", ""

def handle_run_orchestra(topic: str, aspect: str, style: str, voice: str):
    logs = []
    def log_cb(msg):
        logs.append(msg)

    res = run_autonomous_orchestra_pipeline(
        topic=topic,
        format_aspect=aspect,
        style_preset=style,
        voice_choice=voice,
        progress_callback=log_cb
    )
    v_path = res.get("master_video_path")
    log_text = "\n".join(logs)
    if res.get("success") and v_path and os.path.exists(v_path):
        return log_text, v_path, res.get("hf_url", "Saved locally in orchestra_outputs/")
    return log_text, None, "Rendering failed or in progress."

def handle_check_gridshare(host, port, user, ssh_key, comfy_url):
    gridshare_mgr.update_credentials(host, int(port), user, ssh_key, comfy_url)
    status = gridshare_mgr.test_connection()
    workflow_info = gridshare_mgr.inspect_comfyui_workflows()

    report_md = f"### 🖥️ GridShare Remote GPU Telemetry Report\n"
    report_md += f"• **Hardware:** `{status.get('gpu_name')}`\n"
    report_md += f"• **Total VRAM:** `{status.get('vram_total_gb')} GB` (Target: 20GB VRAM)\n"
    report_md += f"• **Free VRAM:** `{status.get('vram_free_gb')} GB`\n"
    report_md += f"• **Temperature:** `{status.get('gpu_temp')}`\n"
    report_md += f"• **Disk Footprint:** `{status.get('disk_info')}`\n"
    report_md += f"• **ComfyUI API:** `{status.get('comfy_status')}`\n\n"

    report_md += "#### 🔍 20GB VRAM Workflow & Node Audit:\n"
    report_md += f"- **Qwen Image 2.1 + Turbo (8-9 steps) + Color Node:** {'✅ Ready' if workflow_info.get('qwen_turbo_colornode_ready') else '❌ Missing'}\n"
    report_md += f"- **MiniMax H3 (Exclusively for 20GB DiT Animation):** {'✅ Ready' if workflow_info.get('minimax_h3_20gb_ready') else '❌ Missing'}\n"
    report_md += f"- **Indic F5-TTS conditioned on Gemini Fenrir:** {'✅ Ready' if workflow_info.get('indic_f5tts_fenrir_ready') else '❌ Missing'}\n\n"

    if workflow_info.get("all_workflows_ready"):
        report_md += "🎉 **Status: All 20GB VRAM Workflows are fully verified and ready for mobile production!**"
    else:
        report_md += "⚠️ **Action Needed:** Some workflows or nodes are missing on the instance. Click **'Auto-Deploy & Setup Workflows'** below to configure them automatically."

    return report_md

def handle_auto_deploy_gridshare():
    logs = []
    def log_cb(msg):
        logs.append(msg)
    res = gridshare_mgr.auto_deploy_and_setup(progress_callback=log_cb)
    return "\n".join(logs)

def handle_test_tts(text_in, voice_mode):
    out_file = str(STATIC_DIR / "temp_live_tts.wav")
    if "Puck" in voice_mode:
        synthesize_gemini_fenrir_tts(text_in, out_file, voice_name="Puck")
    else:
        synthesize_gemini_fenrir_tts(text_in, out_file, voice_name="Fenrir")
    return out_file

def handle_test_telegram(token, chat_id):
    if not token or not chat_id:
        return "❌ Please enter both Bot Token and Chat ID."
    ok = send_telegram_alert("📱 *DramaClaw Mobile Studio*: Connected successfully! You will receive render alerts here.", bot_token=token, chat_id=chat_id)
    return "✅ Test notification sent to your phone!" if ok else "❌ Failed to send Telegram alert. Check token and chat ID."

def handle_hf_connect(token, repo):
    hf_mgr.update_token(token, repo)
    res = hf_mgr.check_connection()
    return f"**Hugging Face Status:** {res.get('message')}"

# -----------------------------------------------------------------------------
# Gradio UI Construction
# -----------------------------------------------------------------------------
def build_interface():
    # Pass title to Blocks; css is provided during launch for Gradio 6 compatibility
    with gr.Blocks(title="DramaClaw Master AI Studio") as app:
        # Hero Banner
        with gr.Row(elem_classes=["hero-banner"]):
            with gr.Column():
                gr.HTML("""
                <h1 class="hero-title">🎬 DramaClaw Master AI Studio</h1>
                <p class="hero-subtitle">
                    Autonomous Vox & Anime Documentary Pipeline • Gemini Multi-Key Mind • 20GB GPU ComfyUI • Indic F5-TTS • Hugging Face Persistence
                </p>
                """)

        # Responsive Telemetry Status Cards
        with gr.Row(elem_classes=["telemetry-grid"]):
            with gr.Column(elem_classes=["telemetry-card"]):
                gr.Markdown(f"""
                <b>🧠 Gemini Mind Pool:</b>
                <code>{len(VALID_GEMINI_KEYS)} Keys Active (Zero Rate Limit)</code>
                """)
            with gr.Column(elem_classes=["telemetry-card"]):
                gr.Markdown("""
                <b>⚡ Remote Cloud GPU:</b>
                <code>20GB VRAM Calibrated (RTX 4000 Ada / 3090)</code>
                """)
            with gr.Column(elem_classes=["telemetry-card"]):
                gr.Markdown("""
                <b>🎙️ Audio Voice Engine:</b>
                <code>Indic F5-TTS + Gemini Fenrir Voice Conditioning</code>
                """)
            with gr.Column(elem_classes=["telemetry-card"]):
                gr.Markdown("""
                <b>🎨 Aesthetic Engine:</b>
                <code>Makoto Shinkai MKL LAB Color Transfer + Targeted Motion</code>
                """)

        with gr.Tabs(elem_classes=["tabs"]):
            # =================================================================
            # TAB 1: GEMINI STORY & ORCHESTRA STUDIO
            # =================================================================
            with gr.Tab("🎬 Gemini Story & Orchestra Studio"):
                with gr.Row():
                    with gr.Column(scale=1):
                        topic_input = gr.Textbox(
                            value="The 22km Siliguri Corridor (Chicken's Neck Geopolitical Mystery)",
                            label="Documentary Topic / Story Idea",
                            lines=2
                        )
                        custom_angle_input = gr.Textbox(
                            label="Specific Angle / Hook (Optional)",
                            placeholder="e.g. Secret underground rail tunnels, satellite radar evidence"
                        )
                        with gr.Row():
                            aspect_ratio = gr.Radio(choices=["9:16", "16:9"], value="9:16", label="Format")
                            style_preset = gr.Dropdown(
                                choices=["Makoto Shinkai Vox", "Dark Geopolitical Thriller", "Ancient Vedic Gold"],
                                value="Makoto Shinkai Vox",
                                label="Visual Aesthetic"
                            )
                        voice_selector = gr.Dropdown(
                            choices=VOICE_MODES,
                            value="Indic F5-TTS (Conditioned with Gemini Fenrir Voice)",
                            label="Narration Voice Engine"
                        )
                        with gr.Row():
                            btn_analyze = gr.Button("🧠 1. Analyze Story & Hook", variant="secondary")
                            btn_storyboard = gr.Button("📋 2. Generate Storyboard", variant="secondary")
                        btn_run_orchestra = gr.Button("🚀 3. Launch 1-Tap Production Orchestra", variant="primary", elem_classes=["primary-touch-btn"])

                    with gr.Column(scale=2):
                        analysis_output = gr.Markdown("### Click '1. Analyze Story & Hook' to reveal Gemini Mind's narrative strategy.")
                        storyboard_table = gr.Markdown("")
                        storyboard_editor = gr.Code(
                            label="Editable Storyboard JSON (Review & customize scene prompts)",
                            language="json",
                            lines=8
                        )

                gr.Markdown("---")
                with gr.Row():
                    with gr.Column():
                        prod_log_output = gr.Textbox(label="Real-Time Production Log Stream", lines=8)
                        hf_cloud_status = gr.Markdown("")
                    with gr.Column():
                        master_video_player = gr.Video(label="1080p Master Documentary Preview")

                btn_analyze.click(
                    fn=handle_story_analysis,
                    inputs=[topic_input, custom_angle_input],
                    outputs=[analysis_output]
                )
                btn_storyboard.click(
                    fn=handle_storyboard_generation,
                    inputs=[topic_input, aspect_ratio, style_preset],
                    outputs=[storyboard_table, storyboard_editor]
                )
                btn_run_orchestra.click(
                    fn=handle_run_orchestra,
                    inputs=[topic_input, aspect_ratio, style_preset, voice_selector],
                    outputs=[prod_log_output, master_video_player, hf_cloud_status]
                )

            # =================================================================
            # TAB 2: GRIDSHARE & COMFYUI 20GB GPU MANAGER
            # =================================================================
            with gr.Tab("⚡ GridShare & ComfyUI 20GB GPU Manager"):
                gr.Markdown("""
                ### 🖥️ Connect GridShare GPU Instance (20GB VRAM Calibrated)
                Manage and audit your remote ComfyUI server. The system automatically inspects whether
                the **Qwen Image 2.1 8-step Turbo + ColorTransfer**, **MiniMax H3 / Wan 2.1**, and **Indic F5-TTS Fenrir** workflows exist.
                """)
                with gr.Row():
                    with gr.Column():
                        gs_host = gr.Textbox(value=DEFAULT_GRIDSHARE_CONFIG["host"], label="GridShare Relay Host / IP")
                        gs_port = gr.Number(value=DEFAULT_GRIDSHARE_CONFIG["port"], label="SSH Port")
                        gs_user = gr.Textbox(value=DEFAULT_GRIDSHARE_CONFIG["user"], label="SSH Username")
                        gs_key = gr.Textbox(value=DEFAULT_GRIDSHARE_CONFIG["ssh_key_path"], label="SSH Private Key Path")
                        gs_comfy_url = gr.Textbox(value=DEFAULT_GRIDSHARE_CONFIG["comfyui_url"], label="ComfyUI Web API URL")
                        with gr.Row():
                            btn_check_gs = gr.Button("🔍 Connect & Inspect Remote GPU", variant="primary")
                            btn_deploy_ws = gr.Button("🛠️ Auto-Deploy & Setup Workflows", variant="secondary")

                    with gr.Column():
                        gs_report = gr.Markdown("### Click 'Connect & Inspect Remote GPU' to check GPU hardware & workflow status.")
                        deploy_log = gr.Textbox(label="Auto-Deployment Console Output", lines=8)

                btn_check_gs.click(
                    fn=handle_check_gridshare,
                    inputs=[gs_host, gs_port, gs_user, gs_key, gs_comfy_url],
                    outputs=[gs_report]
                )
                btn_deploy_ws.click(
                    fn=handle_auto_deploy_gridshare,
                    outputs=[deploy_log]
                )

            # =================================================================
            # TAB 3: 20GB VRAM WORKFLOW HUB & VOICE LAB
            # =================================================================
            with gr.Tab("🎨 20GB VRAM Workflow Hub & Audio Lab"):
                gr.Markdown("### 🧬 20GB VRAM Tuned Generation Workflows & Voice Presets")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        #### 🖼️ Workflow 1: Qwen Image 2.1 + Turbo (8-9 Steps) + Color Node
                        • **Resolution:** 1080x1920 Native Vertical Keyframe  
                        • **Steps:** 8 to 9 steps (Euler / Simple Scheduler)  
                        • **Color Grading:** Native MKL LAB Color Transfer against Makoto Shinkai palette  
                        • **VRAM Usage:** ~7.2 GB (Well within 20GB ceiling)  
                        """)
                        gr.Image(value=str(STATIC_DIR / "shinkai_roof_ref.jpg"), label="Makoto Shinkai Twilight Color Reference Palette")

                    with gr.Column():
                        gr.Markdown("""
                        #### 🎬 Workflow 2: MiniMax H3 Targeted Motion Animation (Exclusively for Video)
                        • **Model:** MiniMax H3 Pruned INT8 UNet + NVFP4 Text Encoder  
                        • **Inference Steps:** 4 steps with Turbo LoRA  
                        • **Targeted Motion:** Dynamic elements animate, background architecture stays 100% solid  
                        • **Audio:** Synchronized 32kHz stereo foley bed generation  
                        • **VRAM Usage:** ~15.4 GB (Safe from CUDA OOM on 20GB GPUs)  
                        """)

                gr.Markdown("---")
                gr.Markdown("#### 🎙️ Voiceover Engine & Audio Reference Samples")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("##### 🌟 Gemini 2.5 Fenrir Male Voice (Deep Epic Narrator)")
                        gr.Audio(value=str(FENRIR_REF_AUDIO) if FENRIR_REF_AUDIO.exists() else None, label="Gemini Fenrir Voice Reference")
                    with gr.Column():
                        gr.Markdown("##### 🧬 Indic F5-TTS Conditioned with Fenrir Voice")
                        gr.Audio(value=str(F5_CONDITIONED_AUDIO) if F5_CONDITIONED_AUDIO.exists() else None, label="F5-TTS Conditioned Audio Sample")

                with gr.Row():
                    with gr.Column():
                        tts_test_text = gr.Textbox(
                            value="2011 me jab Kerala ke Padmanabhaswamy Mandir ke gupt tehkhane khole gaye, toh poori duniya dang reh gayi!",
                            label="Hindi / Hinglish Voice Test Line"
                        )
                        tts_voice_pick = gr.Dropdown(choices=VOICE_MODES, value="Gemini 2.5 Male - Fenrir (Deep Epic Narrator - Recommended)", label="Voice Preset")
                        btn_test_voice = gr.Button("🔊 Test Voice Synthesis")
                    with gr.Column():
                        tts_audio_out = gr.Audio(label="Synthesized Audio Result")

                btn_test_voice.click(
                    fn=handle_test_tts,
                    inputs=[tts_test_text, tts_voice_pick],
                    outputs=[tts_audio_out]
                )

            # =================================================================
            # TAB 4: MOBILE CONTROL & PUSH ALERTS
            # =================================================================
            with gr.Tab("📱 Mobile Control & Push Alerts"):
                gr.Markdown("""
                ### 📲 Remote Mobile Smartphone Control
                You can control the entire autonomous production pipeline directly from your smartphone browser.
                Enable Telegram push notifications to receive real-time alerts and video download links on your phone.
                """)
                with gr.Row():
                    with gr.Column():
                        tg_token_input = gr.Textbox(value=TELEGRAM_BOT_TOKEN, label="Telegram Bot Token", type="password")
                        tg_chat_input = gr.Textbox(value=TELEGRAM_CHAT_ID, label="Telegram Chat ID")
                        btn_test_tg = gr.Button("📲 Send Test Alert to Phone", variant="primary")
                        tg_result = gr.Markdown("")

                    with gr.Column():
                        gr.Markdown("""
                        <div class="card-box">
                            <b>📱 Mobile Control Guidelines:</b><br/>
                            1. Open the Gradio URL on your mobile browser (Safari, Chrome).<br/>
                            2. Select your topic or enter a new documentary idea in Tab 1.<br/>
                            3. Tap <b>'Launch 1-Tap Production Orchestra'</b> and lock your phone.<br/>
                            4. Your phone will receive a Telegram push notification when the 1080p video is rendered and uploaded to Hugging Face!
                        </div>
                        """)

                btn_test_tg.click(
                    fn=handle_test_telegram,
                    inputs=[tg_token_input, tg_chat_input],
                    outputs=[tg_result]
                )

            # =================================================================
            # TAB 5: SCREENING ROOM & HUGGING FACE PERSISTENCE
            # =================================================================
            with gr.Tab("📺 Screening Room & Hugging Face Cloud Storage"):
                gr.Markdown("### 🎬 Watch 1080p Master Documentaries & Manage Hugging Face Storage")
                with gr.Row():
                    with gr.Column(scale=1):
                        hf_token_input = gr.Textbox(value=HF_TOKEN, label="Hugging Face Token (Write Permission)", type="password")
                        hf_repo_input = gr.Textbox(value=HF_DEFAULT_DATASET, label="Hugging Face Storage Dataset / Repo")
                        btn_hf_connect = gr.Button("☁️ Verify Hugging Face Connection")
                        hf_connect_status = gr.Markdown("")

                        all_vids = get_available_completed_videos()
                        video_picker = gr.Dropdown(
                            choices=[name for name, _ in all_vids],
                            value=all_vids[0][0] if all_vids else None,
                            label="Select Master Film to Watch"
                        )
                        btn_refresh_vids = gr.Button("🔄 Refresh Videos List")

                    with gr.Column(scale=2):
                        screening_player = gr.Video(
                            value=all_vids[0][1] if all_vids else None,
                            label="1080p Cinema Player"
                        )

                btn_hf_connect.click(
                    fn=handle_hf_connect,
                    inputs=[hf_token_input, hf_repo_input],
                    outputs=[hf_connect_status]
                )

                def switch_screening(name):
                    vids = get_available_completed_videos()
                    for vname, path in vids:
                        if vname == name and os.path.exists(path):
                            return path
                    return None

                video_picker.change(fn=switch_screening, inputs=[video_picker], outputs=[screening_player])
                btn_refresh_vids.click(
                    fn=lambda: gr.update(choices=[n for n, _ in get_available_completed_videos()]),
                    outputs=[video_picker]
                )

    return app

if __name__ == "__main__":
    app = build_interface()
    # Share=True generates a public gradio.live URL so the user can immediately open it on their mobile phone!
    app.launch(server_name="0.0.0.0", server_port=7860, share=True, css=custom_css)
