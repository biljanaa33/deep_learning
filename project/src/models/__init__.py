from src.models.rt_monodepth import (
    RTMonoDepth,
    RTMonoDepthLiteEncoder,
    RTMonoDepthMobileNetV3,
)

MODEL_NAMES = ("baseline", "lite_encoder", "mobilenet_v3")

DEFAULT_CHECKPOINTS = {
    "baseline": "checkpoints/rt_monodepth_baseline.pth",
    "lite_encoder": "checkpoints/rt_monodepth_lite_encoder.pth",
    "mobilenet_v3": "checkpoints/rt_monodepth_mobilenet_v3.pth",
}


def build_depth_model(name="baseline", max_depth=10.0, pretrained_backbone=False):
    if name == "baseline":
        return RTMonoDepth(max_depth=max_depth)

    if name == "lite_encoder":
        return RTMonoDepthLiteEncoder(max_depth=max_depth)

    if name == "mobilenet_v3":
        return RTMonoDepthMobileNetV3(
            max_depth=max_depth,
            pretrained_backbone=pretrained_backbone,
        )

    raise ValueError(f"Unknown model name: {name}")
