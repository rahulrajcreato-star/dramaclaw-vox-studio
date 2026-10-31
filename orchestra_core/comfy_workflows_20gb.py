import os
import json
from typing import Dict, Any

"""
ComfyUI API Workflow Generators strictly calibrated for 20GB VRAM GPUs
(e.g., NVIDIA RTX 4000 Ada 20GB, RTX 3090 24GB, RTX 4090 24GB).

Key optimizations for 20GB VRAM:
- Qwen Image 2.1 / SDXL Turbo: 8-9 steps fast Euler, memory footprint ~6-8GB.
- ColorTransfer Node: Native MKL LAB color grading against reference (0 extra VRAM, CPU/fast CUDA).
- MiniMax H3: Pruned INT8 ConvRot unet, NVFP4 AWQ clip text encoder, 4-step turbo lora, memory footprint ~14-16GB.
- Wan 2.1: 14B Q4_K_M GGUF, fp8 text encoder, memory footprint ~15GB.
"""

def build_qwen_image_turbo_workflow(
    prompt: str,
    negative_prompt: str,
    width: int = 1080,
    height: int = 1920,
    steps: int = 9,
    cfg: float = 3.5,
    seed: int = 42,
    ref_image_name: str = "shinkai_roof_ref.jpg",
    output_prefix: str = "qwen_keyframe"
) -> Dict[str, Any]:
    """
    Builds the Qwen Image 2.1 + Turbo LoRA + ColorTransfer Node workflow.
    Executes in 8-9 steps with Makoto Shinkai color transfer.
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "qwen_image_2.1_pruned_fp8.safetensors"
            }
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": "qwen_turbo_8step_lora.safetensors",
                "strength_model": 1.0,
                "strength_clip": 1.0
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 1],
                "text": prompt
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 1],
                "text": negative_prompt or "ugly, blurry, low quality, deformed, oversaturated"
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            }
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["1", 2]
            }
        },
        "8": {
            "class_type": "LoadImage",
            "inputs": {
                "image": ref_image_name
            }
        },
        "9": {
            "class_type": "ColorTransfer",
            "inputs": {
                "image_target": ["7", 0],
                "image_ref": ["8", 0],
                "method": "mkl_lab",
                "strength": 0.85,
                "source_stats": "uniform"
            }
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["9", 0],
                "filename_prefix": output_prefix
            }
        }
    }

def build_minimax_h3_20gb_workflow(
    prompt: str,
    ref_keyframe_filename: str,
    width: int = 448,
    height: int = 768,
    num_frames: int = 56,
    seed: int = 12345,
    steps: int = 4,
    output_prefix: str = "minimax_animated_shot"
) -> Dict[str, Any]:
    """
    MiniMax H3 Omni-modal Video + Audio Foley workflow tuned for 20GB VRAM.
    Uses INT8 pruned UNet and NVFP4 CLIP.
    """
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                "weight_dtype": "default"
            }
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "type": "minimax",
                "device": "cpu"  # CPU offload preserves VRAM on 20GB GPUs
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "minimax_h3_video_vae_fp16.safetensors"
            }
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "minimax_h3_audio_vae_fp32.safetensors"
            }
        },
        "5": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["2", 0],
                "lora_name": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
                "strength_model": 1.0,
                "strength_clip": 1.0
            }
        },
        "6": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": ["5", 0],
                "shift_video": 12.0,
                "shift_audio": 3.0
            }
        },
        "7": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["5", 1],
                "vae": ["3", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": num_frames
            }
        },
        "8": {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": ["7", 0]
            }
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["6", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["7", 1],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            }
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["3", 0]
            }
        },
        "11": {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["4", 0]
            }
        },
        "12": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["10", 0],
                "audio": ["11", 0],
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": output_prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 18,
                "save_output": True
            }
        }
    }

def build_wan21_i2v_20gb_workflow(
    prompt: str,
    ref_image_filename: str,
    width: int = 832,
    height: int = 480,
    num_frames: int = 49,
    steps: int = 9,
    seed: int = 2026,
    output_prefix: str = "wan_prod_shot"
) -> Dict[str, Any]:
    """
    Wan 2.1 I2V 14B Q4_K_M GGUF Workflow for 20GB VRAM.
    Includes ColorTransfer node for 100% color consistency with input keyframe.
    """
    return {
        "1": {
            "class_type": "UnetLoaderGGUFAdvanced",
            "inputs": {
                "unet_name": "wan2.1-i2v-14b-480p-Q4_K_M.gguf",
                "dequant_dtype": "default",
                "patch_dtype": "default",
                "patch_on_device": False
            }
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan"
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "wan_2.1_vae.safetensors"
            }
        },
        "4": {
            "class_type": "CLIPVisionLoader",
            "inputs": {
                "clip_name": "clip_vision_h.safetensors"
            }
        },
        "5": {
            "class_type": "LoadImage",
            "inputs": {
                "image": ref_image_filename
            }
        },
        "6": {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["4", 0],
                "image": ["5", 0],
                "crop": "center"
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": prompt
            }
        },
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": "static still frame, frozen picture, low quality, jerky, deformed limbs, blurry, glitches"
            }
        },
        "9": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["7", 0],
                "negative": ["8", 0],
                "vae": ["3", 0],
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
                "clip_vision_output": ["6", 0],
                "start_image": ["5", 0]
            }
        },
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 5.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["9", 0],
                "negative": ["9", 1],
                "latent_image": ["9", 2],
                "denoise": 1.0
            }
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["10", 0],
                "vae": ["3", 0]
            }
        },
        "12": {
            "class_type": "ColorTransfer",
            "inputs": {
                "image_target": ["11", 0],
                "image_ref": ["5", 0],
                "method": "mkl_lab",
                "strength": 1.0,
                "source_stats": "uniform"
            }
        },
        "13": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["12", 0],
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": output_prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 18,
                "pingpong": False,
                "save_output": True
            }
        }
    }
