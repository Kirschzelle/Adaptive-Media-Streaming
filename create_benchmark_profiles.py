from benchmark import EncodingConfig
from benchmark import USE_SVT
from benchmark import USE_VPX

def create():
    configs = []

    # =========================================================================
    # LIBVPX-VP9
    # =========================================================================

    # --- Speed Preset Comparison: slow, balanced, fast ---
    for cpu_used in [0, 2, 5]:
        configs.append(EncodingConfig(
            name=f"vpx_2000k_cpu{cpu_used}",
            encoder=USE_VPX,
            bitrate="2000k",
            speed_preset=cpu_used,
            keyframe_interval=240,
            extra_params={}
        ))

    # --- Bitrate Comparison at Medium Speed ---
    for bitrate in ["1500k", "3000k"]:
        configs.append(EncodingConfig(
            name=f"vpx_{bitrate}_cpu2",
            encoder=USE_VPX,
            bitrate=bitrate,
            speed_preset=2,
            keyframe_interval=240,
            extra_params={}
        ))

    # --- CRF Mode Comparison ---
    for crf in [25, 30, 35]:
        configs.append(EncodingConfig(
            name=f"vpx_crf{crf}_cpu2",
            encoder=USE_VPX,
            crf=crf,
            speed_preset=2,
            keyframe_interval=240,
            extra_params={}
        ))

    # High quality preset
    configs.append(EncodingConfig(
        name="vpx_high_quality",
        encoder=USE_VPX,
        bitrate="3000k",
        speed_preset=0,
        keyframe_interval=240,
        extra_params={
            "-quality": "good",
            "-auto-alt-ref": "1",
            "-lag-in-frames": "25",
            "-arnr-maxframes": "7",
            "-arnr-strength": "5",
            "-aq-mode": "0"
        }
    ))

    # Fast preset for comparison
    configs.append(EncodingConfig(
        name="vpx_fast",
        encoder=USE_VPX,
        bitrate="2000k",
        speed_preset=5,
        keyframe_interval=240,
        extra_params={
            "-quality": "good",
            "-deadline": "realtime"
        }
    ))

    # --- Keyframe Interval Comparison ---
    for kf_interval in [120, 240]:
        configs.append(EncodingConfig(
            name=f"vpx_2000k_kf{kf_interval}",
            encoder=USE_VPX,
            bitrate="2000k",
            speed_preset=2,
            keyframe_interval=kf_interval,
            extra_params={}
        ))

    # =========================================================================
    # SVT-VP9
    # =========================================================================

    # --- Speed Preset Comparison: slow, balanced, fast ---
    for preset in [1, 5, 8]:
        configs.append(EncodingConfig(
            name=f"svt_2000k_preset{preset}",
            encoder=USE_SVT,
            bitrate="2000k",
            speed_preset=preset,
            keyframe_interval=240,
            extra_params={
                "-rc": "1"  # VBR mode
            }
        ))

    # --- Bitrate Comparison at Medium Speed ---
    for bitrate in ["1500k", "2000k", "3000k"]:
        configs.append(EncodingConfig(
            name=f"svt_{bitrate}_preset5",
            encoder=USE_SVT,
            bitrate=bitrate,
            speed_preset=5,
            keyframe_interval=240,
            extra_params={
                "-rc": "1"  # VBR mode
            }
        ))

    # --- CQP Mode (Constant Quality) ---
    for qp in [25, 30, 35]:
        configs.append(EncodingConfig(
            name=f"svt_qp{qp}_preset5",
            encoder=USE_SVT,
            crf=qp,
            speed_preset=5,
            keyframe_interval=240,
            extra_params={
                "-rc": "0"  # CQP mode
            }
        ))

    # --- Keyframe Interval Comparison ---
    for kf_interval in [120, 240]:
        configs.append(EncodingConfig(
            name=f"svt_2000k_kf{kf_interval}",
            encoder=USE_SVT,
            bitrate="2000k",
            speed_preset=5,
            keyframe_interval=kf_interval,
            extra_params={}
        ))

    # =========================================================================
    # Head-to-head
    # =========================================================================

    # Ultra
    configs.append(EncodingConfig(
        name="comparison_ultra_vpx",
        encoder=USE_VPX,
        bitrate="3000k",
        speed_preset=0,
        keyframe_interval=240,
        extra_params={
            "-quality": "good",
            "-auto-alt-ref": "1",
            "-lag-in-frames": "25"
        }
    ))
    
    configs.append(EncodingConfig(
        name="comparison_ultra_svt",
        encoder=USE_SVT,
        bitrate="3000k",
        speed_preset=0,
        keyframe_interval=240,
        extra_params={
            "-rc": "1",
            "-scd": "1"
        }
    ))

    # Balanced
    configs.append(EncodingConfig(
        name="comparison_balanced_vpx",
        encoder=USE_VPX,
        bitrate="2000k",
        speed_preset=2,
        keyframe_interval=240,
        extra_params={
            "-quality": "good"
        }
    ))
    
    configs.append(EncodingConfig(
        name="comparison_balanced_svt",
        encoder=USE_SVT,
        bitrate="2000k",
        speed_preset=5,
        keyframe_interval=240,
        extra_params={
            "-rc": "1"
        }
    ))

    # Fast
    configs.append(EncodingConfig(
        name="comparison_fast_vpx",
        encoder=USE_VPX,
        bitrate="2000k",
        speed_preset=4,
        keyframe_interval=240,
        extra_params={
            "-quality": "good"
        }
    ))
    
    configs.append(EncodingConfig(
        name="comparison_fast_svt",
        encoder=USE_SVT,
        bitrate="2000k",
        speed_preset=8,
        keyframe_interval=240,
        extra_params={
            "-rc": "1"
        }
    ))

    print(f"Created {len(configs)} encoding configurations:")
    print(f"  - {sum(1 for c in configs if c.encoder == USE_VPX)} libvpx-vp9 configs")
    print(f"  - {sum(1 for c in configs if c.encoder == USE_SVT)} SVT-VP9 configs")
    return configs