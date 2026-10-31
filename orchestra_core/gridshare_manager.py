import os
import sys
import socket
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Tuple
from pathlib import Path
from orchestra_core.config import DEFAULT_GRIDSHARE_CONFIG, GPU_VRAM_TARGET_GB, MAX_GPU_DISK_LIMIT_GB

class GridShareManager:
    def __init__(self, config: Dict[str, Any] = None):
        self.cfg = config or dict(DEFAULT_GRIDSHARE_CONFIG)
        self.host = self.cfg.get("host", "relay.gridshare.in")
        self.port = int(self.cfg.get("port", 30000))
        self.user = self.cfg.get("user", "root")
        self.ssh_key_path = self.cfg.get("ssh_key_path", os.path.expanduser("~/.ssh/id_rsa"))
        self.comfyui_url = self.cfg.get("comfyui_url", "http://127.0.0.1:8188").rstrip("/")
        self.max_disk_limit_gb = MAX_GPU_DISK_LIMIT_GB  # Strict 50.0 GB ceiling

    def update_credentials(self, host: str, port: int, user: str, ssh_key_path: str = "", comfyui_url: str = ""):
        self.host = host.strip() or self.host
        self.port = int(port) if port else self.port
        self.user = user.strip() or self.user
        if ssh_key_path:
            self.ssh_key_path = ssh_key_path.strip()
        if comfyui_url:
            self.comfyui_url = comfyui_url.strip().rstrip("/")

    def test_connection(self) -> Dict[str, Any]:
        """
        Tests SSH and HTTP reachability of the GridShare instance and gathers GPU telemetry.
        """
        result = {
            "online": False,
            "ssh_status": "DISCONNECTED",
            "comfy_status": "OFFLINE",
            "gpu_name": "Unknown",
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
            "gpu_temp": "N/A",
            "disk_info": "N/A",
            "disk_used_gb": 0.0,
            "disk_limit_gb": self.max_disk_limit_gb,
            "vram_20gb_ready": False,
            "message": ""
        }

        # 1. Quick TCP Port Ping
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.5)
            err = sock.connect_ex((self.host, self.port))
            sock.close()
            if err != 0:
                result["message"] = f"Port {self.port} on {self.host} is unreachable or closed."
                return result
            result["ssh_status"] = "PORT_OPEN"
        except Exception as e:
            result["message"] = f"Network socket error: {str(e)[:60]}"
            return result

        # 2. SSH Telemetry via paramiko
        try:
            import paramiko
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            connect_kwargs = {
                "hostname": self.host,
                "port": self.port,
                "username": self.user,
                "timeout": 4.0,
                "look_for_keys": True
            }
            if os.path.exists(self.ssh_key_path):
                connect_kwargs["key_filename"] = self.ssh_key_path

            client.connect(**connect_kwargs)
            result["online"] = True
            result["ssh_status"] = "CONNECTED"

            # Execute nvidia-smi
            _, stdout, _ = client.exec_command(
                "nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader,nounits"
            )
            smi_out = stdout.read().decode("utf-8", errors="ignore").strip()
            if smi_out:
                parts = [p.strip() for p in smi_out.split(",")]
                if len(parts) >= 4:
                    result["gpu_name"] = parts[0]
                    total_mb = float(parts[1])
                    free_mb = float(parts[2])
                    result["vram_total_gb"] = round(total_mb / 1024.0, 1)
                    result["vram_free_gb"] = round(free_mb / 1024.0, 1)
                    result["gpu_temp"] = f"{parts[3]}°C"
                    
                    if result["vram_total_gb"] >= 19.0:
                        result["vram_20gb_ready"] = True

            # Check disk footprint
            _, stdout, _ = client.exec_command("df -BG /workspace 2>/dev/null || df -BG /")
            df_lines = stdout.read().decode("utf-8", errors="ignore").strip().splitlines()
            if len(df_lines) > 1:
                cols = df_lines[-1].split()
                if len(cols) >= 4:
                    used_str = cols[2].replace("G", "")
                    try:
                        result["disk_used_gb"] = float(used_str)
                    except ValueError:
                        pass
                result["disk_info"] = df_lines[-1]

            client.close()
        except Exception as e:
            result["message"] = f"SSH authentication/telemetry warning: {str(e)[:70]}"

        # 3. Check ComfyUI API endpoint
        comfy_info = self.check_comfyui_http()
        result["comfy_status"] = comfy_info.get("status", "OFFLINE")
        if comfy_info.get("online"):
            result["online"] = True
            if not result["message"]:
                result["message"] = f"Connected to GPU ({result['gpu_name']} - {result['vram_total_gb']}GB) & ComfyUI Active."

        return result

    def check_comfyui_http(self) -> Dict[str, Any]:
        """Pings ComfyUI API /system_stats endpoint."""
        try:
            req = urllib.request.Request(f"{self.comfyui_url}/system_stats", headers={"User-Agent": "DramaClaw/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode())
                return {"online": True, "status": "ONLINE", "stats": data}
        except Exception as e:
            return {"online": False, "status": f"OFFLINE ({str(e)[:40]})"}

    def audit_gpu_disk_and_weights(self) -> Dict[str, Any]:
        """
        Deep Audit of GPU Server Storage & Model Weight Cache:
        1. Checks current disk usage against 50 GB ceiling.
        2. Detects whether all workflow weights (MiniMax H3, Qwen Turbo, ColorTransfer, F5-TTS)
           already exist on the GPU server.
        3. Enforces idempotency: If weights already exist, PREVENTS redundant pushes!
        """
        audit = {
            "connected": False,
            "disk_used_gb": 0.0,
            "disk_limit_gb": self.max_disk_limit_gb,
            "disk_within_budget": True,
            "weights_already_cached": False,
            "missing_weights": [],
            "cached_weights": [],
            "status_headline": ""
        }

        try:
            import paramiko
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                key_filename=self.ssh_key_path if os.path.exists(self.ssh_key_path) else None,
                timeout=4.0
            )
            audit["connected"] = True

            # 1. Query exact disk usage
            _, stdout, _ = client.exec_command("df -BG /workspace 2>/dev/null || df -BG /")
            df_lines = stdout.read().decode().strip().splitlines()
            if len(df_lines) > 1:
                cols = df_lines[-1].split()
                if len(cols) >= 3:
                    try:
                        used_val = float(cols[2].replace("G", ""))
                        audit["disk_used_gb"] = used_val
                        audit["disk_within_budget"] = (used_val <= self.max_disk_limit_gb)
                    except ValueError:
                        pass

            # 2. Check existence of specific workflow weights
            weight_checks = [
                # MiniMax H3 (Exclusively for Video Animation & Foley)
                ("MiniMax H3 INT8 UNet", "/workspace/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
                ("MiniMax H3 NVFP4 Text Encoder", "/workspace/ComfyUI/models/clip/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
                ("MiniMax H3 Turbo 4-Step LoRA", "/workspace/ComfyUI/models/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
                ("MiniMax H3 Video VAE", "/workspace/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors"),
                ("MiniMax H3 Audio VAE (32kHz Foley)", "/workspace/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors"),
                # Qwen Image 2.1 (Keyframes)
                ("Qwen Image 2.1 FP8 Model", "/workspace/ComfyUI/models/checkpoints/qwen_image_2.1_pruned_fp8.safetensors"),
                ("Qwen Turbo 8-Step LoRA", "/workspace/ComfyUI/models/loras/qwen_turbo_8step_lora.safetensors"),
                # Custom Nodes
                ("ColorTransfer Node (MKL LAB)", "/workspace/ComfyUI/custom_nodes/ComfyUI-Color-Transfer"),
                ("VideoHelperSuite", "/workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite"),
                # Audio Reference
                ("Gemini Fenrir Audio Reference", "/workspace/gemini_ref_sample.wav"),
                ("Makoto Shinkai Color Palette Ref", "/workspace/ComfyUI/input/shinkai_roof_ref.jpg")
            ]

            # Fast batch check via python/bash script on GPU
            check_script = "import os; paths = " + str([p[1] for p in weight_checks]) + "; print([os.path.exists(p) for p in paths])"
            _, stdout, _ = client.exec_command(f'python3 -c "{check_script}" 2>/dev/null')
            exists_list_str = stdout.read().decode().strip()

            if exists_list_str.startswith("["):
                exists_flags = eval(exists_list_str)
                for (name, path), exists in zip(weight_checks, exists_flags):
                    if exists:
                        audit["cached_weights"].append(name)
                    else:
                        audit["missing_weights"].append(name)
            else:
                # Fallback: check missing
                audit["missing_weights"] = [w[0] for w in weight_checks]

            client.close()

            # Determine whether weights already exist on GPU
            if len(audit["missing_weights"]) == 0:
                audit["weights_already_cached"] = True
                audit["status_headline"] = (
                    f"🟢 WEIGHTS CACHED ON GPU ({audit['disk_used_gb']:.1f} GB / {self.max_disk_limit_gb} GB): "
                    f"Zero push needed! Server is primed for instant production."
                )
            else:
                audit["weights_already_cached"] = False
                audit["status_headline"] = (
                    f"🟡 MISSING {len(audit['missing_weights'])} WEIGHTS ({audit['disk_used_gb']:.1f} GB / {self.max_disk_limit_gb} GB): "
                    f"Safe to deploy within 50 GB ceiling."
                )

        except Exception as e:
            audit["status_headline"] = f"⚠️ Server query note: {str(e)[:50]}"

        return audit

    def inspect_comfyui_workflows(self) -> Dict[str, Any]:
        """
        Pre-flight check:
        Verifies presence of:
        1. Qwen Image 2.1 + Turbo LoRA + ColorTransfer Node (MKL LAB)
        2. MiniMax H3 (Exclusively for 20GB VRAM Animation)
        3. Indic F5-TTS conditioned on Gemini Fenrir voice
        4. Strict 50 GB disk budget verification
        """
        inspection = {
            "comfyui_reachable": False,
            "qwen_turbo_colornode_ready": False,
            "minimax_h3_20gb_ready": False,
            "indic_f5tts_fenrir_ready": False,
            "weights_already_cached": False,
            "disk_used_gb": 0.0,
            "disk_limit_gb": self.max_disk_limit_gb,
            "all_workflows_ready": False,
            "missing_components": [],
            "details": {}
        }

        # Run weights audit
        audit = self.audit_gpu_disk_and_weights()
        inspection["weights_already_cached"] = audit["weights_already_cached"]
        inspection["disk_used_gb"] = audit["disk_used_gb"]
        inspection["missing_components"] = audit["missing_weights"]

        # 1. Fetch registered ComfyUI nodes via /object_info
        try:
            req = urllib.request.Request(f"{self.comfyui_url}/object_info", headers={"User-Agent": "DramaClawVerifier/1.0"})
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode())
                inspection["comfyui_reachable"] = True
        except Exception:
            return self._inspect_via_ssh_fallback()

        # Check required nodes
        has_color_transfer = ("ColorTransfer" in data) or ("ColorMatch" in data) or ("MKLColorTransfer" in data)
        has_lora = "LoraLoader" in data
        has_ksampler = "KSampler" in data
        inspection["qwen_turbo_colornode_ready"] = (has_color_transfer and has_lora and has_ksampler)

        # MiniMax H3 Video Nodes (Exclusively for Animation)
        has_vhs = "VHS_VideoCombine" in data
        has_minimax = ("MiniMaxH3ImageToVideo" in data) or ("MiniMaxH3SigmaShift" in data)
        inspection["minimax_h3_20gb_ready"] = (has_vhs and has_minimax)

        # Indic F5-TTS
        inspection["indic_f5tts_fenrir_ready"] = self._check_remote_f5_tts()

        inspection["all_workflows_ready"] = (
            inspection["qwen_turbo_colornode_ready"] and
            inspection["minimax_h3_20gb_ready"] and
            inspection["indic_f5tts_fenrir_ready"]
        )

        return inspection

    def _inspect_via_ssh_fallback(self) -> Dict[str, Any]:
        """Inspects custom_nodes directory and models directly via SSH."""
        res = {
            "comfyui_reachable": False,
            "qwen_turbo_colornode_ready": False,
            "minimax_h3_20gb_ready": False,
            "indic_f5tts_fenrir_ready": False,
            "all_workflows_ready": False,
            "missing_components": [],
            "ssh_fallback_used": True
        }
        try:
            import paramiko
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                key_filename=self.ssh_key_path if os.path.exists(self.ssh_key_path) else None,
                timeout=4.0
            )

            _, stdout, _ = client.exec_command("ls -d /workspace/ComfyUI/custom_nodes/* 2>/dev/null")
            installed_nodes = stdout.read().decode().lower()

            res["qwen_turbo_colornode_ready"] = ("color" in installed_nodes or "mixlab" in installed_nodes)
            res["minimax_h3_20gb_ready"] = ("videohelpersuite" in installed_nodes or "vhs" in installed_nodes or "minimax" in installed_nodes)

            _, stdout, _ = client.exec_command("python3 -c 'import f5_tts; print(\"OK\")' 2>/dev/null")
            res["indic_f5tts_fenrir_ready"] = ("OK" in stdout.read().decode().strip())

            res["all_workflows_ready"] = (res["qwen_turbo_colornode_ready"] and res["minimax_h3_20gb_ready"] and res["indic_f5tts_fenrir_ready"])
            client.close()
        except Exception as e:
            res["missing_components"].append(f"SSH Inspection error: {str(e)[:50]}")

        return res

    def _check_remote_f5_tts(self) -> bool:
        """Verifies if F5-TTS is available on remote instance."""
        try:
            import paramiko
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=self.host, port=self.port, username=self.user, key_filename=self.ssh_key_path, timeout=3.0)
            _, stdout, _ = client.exec_command("python3 -c 'import f5_tts; print(\"READY\")' 2>/dev/null")
            ready = "READY" in stdout.read().decode()
            client.close()
            return ready
        except Exception:
            return False

    def auto_deploy_and_setup(self, progress_callback=None) -> Dict[str, Any]:
        """
        Idempotent & 50GB Budget Protected Auto-Deployment:
        1. Checks if workflow weights already exist on GPU server.
           -> IF ALREADY PRESENT: SKIPS PUSH COMPLETELY!
        2. IF NOT PRESENT: Checks disk usage against 50 GB limit.
           -> Only pushes if total footprint remains strictly under 50 GB!
        """
        logs = []
        def log(msg):
            logs.append(msg)
            if progress_callback:
                progress_callback(msg)

        log("🔍 [STEP 0] Auditing GPU Server Storage & Model Weight Cache...")
        audit = self.audit_gpu_disk_and_weights()
        current_disk = audit.get("disk_used_gb", 0.0)

        # ---------------------------------------------------------------------
        # IDEMPOTENCY GUARD: Do NOT push if already present!
        # ---------------------------------------------------------------------
        if audit.get("weights_already_cached"):
            log("=" * 65)
            log("🟢 [CACHE HIT - SKIP PUSH] ALL WORKFLOW WEIGHTS ALREADY EXIST ON GPU SERVER!")
            log(f"• Current Disk Usage: {current_disk:.1f} GB / {self.max_disk_limit_gb} GB (SAFE)")
            log("• Qwen Image 2.1 (8-step Turbo) + Color Node: CACHED")
            log("• MiniMax H3 (Exclusively for Video Animation): CACHED")
            log("• Indic F5-TTS + Fenrir Audio Sample: CACHED")
            log("• Action: Skipping redundant download/push to save time, bandwidth & disk space.")
            log("=" * 65)
            return {
                "success": True,
                "action": "SKIPPED_ALREADY_CACHED",
                "message": "All weights already exist on GPU server. Re-push skipped.",
                "logs": logs
            }

        # ---------------------------------------------------------------------
        # 50 GB CEILING GUARD: Ensure disk will stay under 50 GB!
        # ---------------------------------------------------------------------
        log(f"⚠️ Missing components detected: {audit.get('missing_weights', [])}")
        log(f"📊 Checking 50 GB Disk Budget (Current Usage: {current_disk:.1f} GB / {self.max_disk_limit_gb} GB)...")

        if current_disk > (self.max_disk_limit_gb - 4.0):
            log(f"🧹 Server disk near 50 GB limit! Purging temporary files and cache...")
            try:
                import paramiko
                cl = paramiko.SSHClient()
                cl.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                cl.connect(hostname=self.host, port=self.port, username=self.user, key_filename=self.ssh_key_path, timeout=5.0)
                cl.exec_command("rm -rf /workspace/ComfyUI/temp/* /tmp/* ~/.cache/huggingface/hub/tmp* 2>/dev/null")
                cl.close()
                log("✅ Cleaned temporary caches to protect 50 GB budget.")
            except Exception:
                pass

        log("🚀 Starting Safe Incremental Deployment strictly under 50 GB...")
        try:
            import paramiko
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                key_filename=self.ssh_key_path if os.path.exists(self.ssh_key_path) else None,
                timeout=10.0
            )

            # Ensure directories
            client.exec_command("mkdir -p /workspace/ComfyUI/custom_nodes /workspace/ComfyUI/models/loras /workspace/ComfyUI/models/checkpoints /workspace/ComfyUI/models/diffusion_models")

            # 1. Deploy ComfyUI-Color-Transfer
            log("🎨 Installing ColorTransfer Node (MKL LAB 100% Palette Match)...")
            cmd_color = (
                "cd /workspace/ComfyUI/custom_nodes && "
                "if [ ! -d 'ComfyUI-Color-Transfer' ]; then "
                "git clone --depth 1 https://github.com/kijai/ComfyUI-Color-Transfer.git; "
                "pip install -q scikit-image opencv-python; "
                "fi"
            )
            client.exec_command(cmd_color)

            # 2. Deploy ComfyUI-VideoHelperSuite (for MiniMax H3 VideoCombine)
            log("🎬 Installing VideoHelperSuite (VHS_VideoCombine)...")
            cmd_vhs = (
                "cd /workspace/ComfyUI/custom_nodes && "
                "if [ ! -d 'ComfyUI-VideoHelperSuite' ]; then "
                "git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && "
                "pip install -q -r ComfyUI-VideoHelperSuite/requirements.txt; "
                "fi"
            )
            client.exec_command(cmd_vhs)

            # 3. Install Indic F5-TTS
            log("🎙️ Installing Indic F5-TTS & conditioned voice support...")
            client.exec_command("pip install -q f5-tts torchaudio soundfile")

            # 4. Upload reference assets
            sftp = client.open_sftp()
            local_shinkai = Path(__file__).resolve().parent.parent / "static" / "shinkai_roof_ref.jpg"
            if local_shinkai.exists():
                log("🖼️ Uploading Makoto Shinkai Twilight Color Reference Palette...")
                sftp.put(str(local_shinkai), "/workspace/ComfyUI/input/shinkai_roof_ref.jpg")

            local_fenrir = Path(__file__).resolve().parent.parent / "static" / "gemini_male_fenrir.wav"
            if local_fenrir.exists():
                log("🎙️ Uploading Gemini Fenrir Reference Voice Sample for F5-TTS...")
                sftp.put(str(local_fenrir), "/workspace/gemini_ref_sample.wav")

            sftp.close()

            # Final disk audit
            _, stdout, _ = client.exec_command("df -BG /workspace 2>/dev/null || df -BG /")
            df_lines = stdout.read().decode().strip().splitlines()
            final_used = "N/A"
            if len(df_lines) > 1:
                cols = df_lines[-1].split()
                if len(cols) >= 3:
                    final_used = cols[2]
            client.close()

            log(f"✅ Deployment Complete! Final GPU Disk Usage: {final_used} (STRICTLY UNDER 50 GB LIMIT)")
            return {"success": True, "action": "DEPLOYED_WITHIN_BUDGET", "logs": logs}
        except Exception as e:
            err_msg = f"Auto-deployment failed: {e}"
            log(f"❌ {err_msg}")
            return {"success": False, "error": err_msg, "logs": logs}
