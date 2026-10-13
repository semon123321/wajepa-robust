"""Causal camera faults: no future image is ever substituted into the history."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Fault:
    kind: str = "clean"
    camera: int = 1  # CAM_F0, index in the OFFICIAL camera ordering
    delay: int = 1  # 0.5-second history intervals

    @property
    def name(self):
        return "clean" if self.kind == "clean" else f"{self.kind}_v{self.camera}_d{self.delay}"


def apply_fault(features, fault):
    images = features["history_images"].clone()
    b, t, v = images.shape[:3]
    # Metadata is per frame/view: [available, age in seconds].
    status = images.new_zeros((b, t, v, 2))
    status[..., 0] = 1
    if fault.kind == "blackout":
        images[:, :, fault.camera] = -1  # physical black RGB, before ImageNet normalization
        status[:, :, fault.camera, 0] = 0
    elif fault.kind == "stale":
        if fault.delay < 1 or fault.delay >= t:
            raise ValueError("delay must fit inside the observed history")
        # Startup boundary: hold the oldest available historical frame.
        for target in range(t):
            source = max(0, target - fault.delay)
            images[:, target, fault.camera] = features["history_images"][:, source, fault.camera]
            status[:, target, fault.camera, 1] = (target - source) * 0.5
    elif fault.kind == "hold":
        # Last two frames fail; hold frame 1, mimicking a burst packet loss.
        images[:, 2:, fault.camera] = features["history_images"][:, 1:2, fault.camera]
        status[:, 2, fault.camera, 1] = 0.5
        status[:, 3, fault.camera, 1] = 1.0
    elif fault.kind != "clean":
        raise ValueError(fault.kind)
    return {**features, "history_images": images}, status
