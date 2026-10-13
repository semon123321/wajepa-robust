"""Minimal OpenScene reader matching NAVSIM's official scene selection.

Only deserialize pickle files downloaded from the pinned official dataset.
Model inputs are constructed from history only. Future poses are labels only.
"""
from pathlib import Path
import hashlib
import json
import pickle

import cv2
import numpy as np
import torch
import yaml
from pyquaternion import Quaternion

CAMERAS = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0")


def relative_poses(poses):
    poses = np.asarray(poses, dtype=np.float64)
    origin = poses[3].copy()
    result = poses - origin
    c, s = np.cos(origin[2]), np.sin(origin[2])
    result[:, :2] = result[:, :2] @ np.array([[c, -s], [s, c]])
    result[:, 2] = np.arctan2(np.sin(result[:, 2]), np.cos(result[:, 2]))
    return result.astype(np.float32)


def pose(frame):
    # Match NAVSIM/PyQuaternion's intrinsic convention exactly, including roll/pitch.
    yaw = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]
    return [*frame["ego2global_translation"][:2], yaw]


class OpenSceneSubset:
    def __init__(self, metadata, sensors, scene_filter, limit=None, stride=1, logs=None):
        self.sensors = Path(sensors)
        self.filter_path = Path(scene_filter)
        filt = yaml.safe_load(self.filter_path.read_text(encoding="utf-8"))
        self.filter_sha256 = hashlib.sha256(self.filter_path.read_bytes()).hexdigest()
        allowed_logs = set(filt["log_names"]) if filt.get("log_names") else None
        allowed_tokens = set(filt["tokens"]) if filt.get("tokens") else None
        if filt["num_history_frames"] != 4:
            raise ValueError("This adapter expects the official four-frame history.")
        total = 4 + int(filt["num_future_frames"])
        self.scenes = []
        for log in sorted(self.sensors.iterdir()):
            if not log.is_dir() or (logs and log.name not in logs):
                continue
            if allowed_logs is not None and log.name not in allowed_logs:
                continue
            meta = Path(metadata) / (log.name + ".pkl")
            if not meta.exists():
                continue
            with meta.open("rb") as stream:
                frames = pickle.load(stream)
            selected = []
            for i in range(0, len(frames) - total + 1, int(filt["frame_interval"])):
                window = frames[i:i + total]
                current = window[3]
                if filt.get("has_route") and not len(current["roadblock_ids"]):
                    continue
                if allowed_tokens is not None and current["token"] not in allowed_tokens:
                    continue
                # All four historical camera views must exist. Future RGB is never needed.
                if not all((self.sensors / f["cams"][cam]["data_path"]).is_file()
                           for f in window[:4] for cam in CAMERAS):
                    continue
                ts = np.array([f["timestamp"] for f in window[:12]], dtype=np.float64)
                if not np.allclose(np.diff(ts) / 1e6, 0.5, atol=0.06):
                    raise ValueError(f"Unexpected temporal spacing in {log.name}")
                selected.append(window)
            self.scenes.extend(selected[::stride])
        # Round-robin logs to prevent the small smoke subset coming from only one log.
        groups = {}
        for scene in self.scenes:
            groups.setdefault(scene[3]["log_name"], []).append(scene)
        self.scenes = [g[i] for i in range(max(map(len, groups.values()), default=0))
                       for g in groups.values() if i < len(g)]
        if limit is not None:
            self.scenes = self.scenes[:limit]
        if not self.scenes:
            raise ValueError("No scenes match the official filter and downloaded sensors.")

    def __len__(self):
        return len(self.scenes)

    def manifest(self):
        return {"filter_sha256": self.filter_sha256, "cameras": CAMERAS,
                "history_frames": 4, "prediction_frames": 8, "interval_s": 0.5,
                "scenes": [{"token": s[3]["token"], "log": s[3]["log_name"]}
                           for s in self.scenes]}

    def __getitem__(self, i):
        window = self.scenes[i]
        images = []
        for frame in window[:4]:
            views = []
            for cam in CAMERAS:
                filename = self.sensors / frame["cams"][cam]["data_path"]
                bgr = cv2.imread(str(filename))
                if bgr is None:
                    raise IOError(filename)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (512, 256), interpolation=cv2.INTER_AREA)
                views.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 127.5 - 1)
            images.append(torch.stack(views))
        current = window[3]
        cmd = np.asarray(current["driving_command"], dtype=np.float32).ravel()
        if cmd.size != 4 or not np.isfinite(cmd).all() or cmd.max() <= 0:
            raise ValueError("Missing navigation command; never derive it from future labels.")
        onehot = np.eye(4, dtype=np.float32)[cmd.argmax()]
        dynamics = np.asarray(current["ego_dynamic_state"], dtype=np.float32).ravel()
        if dynamics.size != 4:
            raise ValueError("Expected body-frame vx,vy,ax,ay.")
        positions = relative_poses([pose(f) for f in window[:12]])
        features = {"history_images": torch.stack(images).unsqueeze(0),
                    "history_trajectory": torch.from_numpy(positions[:4]).unsqueeze(0),
                    "ego_status": torch.from_numpy(np.concatenate([onehot, dynamics])).unsqueeze(0)}
        return features, positions[4:12], self.manifest()["scenes"][i]
