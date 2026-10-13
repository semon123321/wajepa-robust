"""Load the exact official model. No replacement planning backbone.

An optional context hook modifies only observed-history scene tokens; the
official joint future-latent / action flow sampler remains untouched.
"""
import sys
import time
from pathlib import Path
import torch
from omegaconf import OmegaConf

UPSTREAM_COMMIT = "bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad"


class OfficialModel:
    def __init__(self, upstream, checkpoint):
        upstream = Path(upstream).resolve()
        sys.path.insert(0, str(upstream))
        from models import build_world_model
        cfg = OmegaConf.load(upstream / "configs/wa_jepa_hugsim.yaml")
        torch.manual_seed(42)
        self.model = build_world_model(cfg)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        for key in ("model_state_dict", "state_dict"):
            if key in weights:
                weights = weights[key]
                break
        self.model.load_state_dict(weights, strict=True, assign=True)
        del weights
        self.model.eval().requires_grad_(False).to("cuda", torch.bfloat16)
        self.adapter = None
        self.status = None
        self.captured = None
        self.capture = False
        self._original_encode = self.model._encode_scene_context

        def hook(*args, **kwargs):
            context = self._original_encode(*args, **kwargs)
            if self.capture:
                self.captured = context.detach().cpu()
            if self.adapter is not None:
                context = self.adapter(context.float(), self.status.float()).to(context.dtype)
            return context

        self.model._encode_scene_context = hook

    def encode(self, features):
        """Cache frozen context only; same history/zero-future path as official inference."""
        m = self.model
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            f = m.prepare_batch(features)
            history = m._ensure_multiview(f["history_images"])
            b, _, v, c, h, w = history.shape
            future = history.new_zeros((b, m.num_future_frames, v, c, h, w))
            clip = m._make_full_clip(history, future)
            masks = m.mask_sampler.full_future_mask(b, device=clip.device)
            context = self._original_encode(clip, masks.masks_x[0], b)
        return context.detach().cpu()

    def predict(self, features, status=None):
        self.status = status.to("cuda") if status is not None else None
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            trajectory = self.model.predict_trajectory(features)
        torch.cuda.synchronize()
        return trajectory[0].float().cpu().numpy(), time.perf_counter() - start

    def teacher_rollout(self, features):
        """Record the clean model's actual four-step sampler states for distillation.

        No recorded future is used: these are the model's imagined latent states.
        Targets belong only to the training cache and never enter evaluation.
        """
        records = []
        indices = torch.linspace(0, 8191, 256, device="cuda").long()

        def before(module, args, kwargs):
            records.append({"scene": kwargs["noisy_future_scene"].detach().cpu(),
                            "trajectory": kwargs["trajectory_inputs"]["noisy_trajectory"].detach().cpu(),
                            "t": kwargs["t_cont"].detach().cpu()})

        def after(module, args, kwargs, outputs):
            records[-1]["target_action"] = outputs[1].detach().cpu()
            records[-1]["target_scene"] = outputs[0][:, indices].detach().cpu()

        pre = self.model.predictor.register_forward_pre_hook(before, with_kwargs=True)
        post = self.model.predictor.register_forward_hook(after, with_kwargs=True)
        try:
            self.predict(features)
        finally:
            pre.remove()
            post.remove()
        return {"steps": records, "scene_indices": indices.cpu(),
                "history_trajectory": features["history_trajectory"].cpu(),
                "ego_status": features["ego_status"].cpu()}
