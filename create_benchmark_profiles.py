from benchmark import EncodingConfig
from benchmark import USE_SVT
from benchmark import USE_VPX

def create():
    configs = []

    configs.append(EncodingConfig(
        name="vpx_fast",
        encoder=USE_VPX,
        bitrate="2000k",
        speed_preset=4,
        keyframe_interval=240,
        extra_params={"-deadline": "realtime"}
    ))

    configs.append(EncodingConfig(
        name=f"vpx_2000k_cpu3",
        encoder=USE_VPX,
        bitrate="2000k",
        speed_preset=3,
        keyframe_interval=240,
        extra_params={}
    ))

    configs.append(EncodingConfig(
        name="vpx_2000k_cpu3_2pass",
        encoder=USE_VPX,
        bitrate="2000k",
        speed_preset=4,
        keyframe_interval=240,
        extra_params={
            "-pass": "2",
            "-passlogfile": "/tmp/vpx_fast_2pass"
        }
    ))

    for crf in [20, 35, 50]:
        for speed in [2,4]:
            if (crf == 20 and speed == 4) or (crf == 50 and speed == 4):
                continue
            configs.append(EncodingConfig(
                name=f"vpx_crf{crf}_cpu{speed}",
                encoder=USE_VPX,
                crf=crf,
                speed_preset=speed,
                keyframe_interval=240,
                extra_params={}
            ))

    print(f"Created {len(configs)} encoding configurations:")
    print(f"  - {sum(1 for c in configs if c.encoder == USE_VPX)} libvpx-vp9 configs")
    print(f"  - {sum(1 for c in configs if c.encoder == USE_SVT)} SVT-VP9 configs")
    return configs