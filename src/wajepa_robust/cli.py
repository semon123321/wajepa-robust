import argparse
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch

from .adapter import ReliabilityAdapter
from .corruptions import Fault, apply_fault
from .data import OpenSceneSubset
from .metrics import displacement, summarize
from .model import OfficialModel, UPSTREAM_COMMIT


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def fault_list():
    return [Fault(), Fault("blackout"), Fault("stale", delay=1), Fault("stale", delay=2), Fault("hold")]


def dataset(config, limit=None):
    return OpenSceneSubset(config["metadata"], config["sensors"], config["scene_filter"],
                           limit=limit, stride=config.get("stride", 1), logs=config.get("logs"))


def evaluate(args, config):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = dataset(config, args.limit)
    dump(out / "manifest.json", data.manifest())
    model = OfficialModel(config["upstream"], config["checkpoint"])
    if args.adapter:
        saved = torch.load(args.adapter, map_location="cpu", weights_only=True)
        eval_logs = {s["log"] for s in data.manifest()["scenes"]}
        adaptation_logs = set(saved["train_logs"]) | set(saved["val_logs"])
        if eval_logs & adaptation_logs:
            raise ValueError("Evaluation logs overlap adapter training/validation logs.")
        adapter = ReliabilityAdapter(**saved["architecture"])
        adapter.load_state_dict(saved["state_dict"], strict=True)
        model.adapter = adapter.eval().to("cuda")
    rows = []
    # Warmup separately, excluded from timings.
    model.predict(data[0][0], apply_fault(data[0][0], Fault())[1])
    torch.cuda.reset_peak_memory_stats()
    for i in range(len(data)):
        features, target, info = data[i]
        for fault in fault_list():
            corrupted, status = apply_fault(features, fault)
            prediction, latency = model.predict(corrupted, status)
            row = {**info, "condition": fault.name, **displacement(prediction, target),
                   "latency_s": latency, "prediction": prediction.tolist(), "target": target.tolist()}
            rows.append(row)
        # Incremental output survives interruptions. No selection based on test errors.
        dump(out / "predictions.json", rows)
        print(f"eval {i+1}/{len(data)} {info['token']}", flush=True)
    result = {"scope": "OpenScene shard subset; offline displacement, NOT official EPDMS or closed-loop score",
              "model_commit": UPSTREAM_COMMIT, "precision": "bfloat16", "sampling_steps": 4,
              "inference_seed": 1, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "python": platform.python_version(), "adapter": bool(args.adapter),
              "peak_reserved_gb": torch.cuda.max_memory_reserved()/1e9,
              "metrics": summarize(rows)}
    dump(out / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def cache(args, config):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = dataset(config, args.limit)
    dump(out / "manifest.json", data.manifest())
    model = OfficialModel(config["upstream"], config["checkpoint"])
    for i in range(len(data)):
        features, _, info = data[i]
        destination = out / (info["token"] + ".pt")
        if destination.exists():
            continue
        clean = model.encode(features)
        contexts, statuses = [], []
        # All cameras trained; unseen combinations reserved for later testing.
        for camera in range(4):
            for fault in [Fault("blackout", camera), Fault("stale", camera, 1),
                          Fault("stale", camera, 2), Fault("hold", camera)]:
                corrupted, status = apply_fault(features, fault)
                contexts.append(model.encode(corrupted))
                statuses.append(status.cpu())
        torch.save({"clean": clean, "corrupt": torch.cat(contexts),
                    "status": torch.cat(statuses), **info}, destination)
        print(f"cache {i+1}/{len(data)} {info['log']}", flush=True)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest = json.loads((Path(args.cache)/"manifest.json").read_text())
    # The manifest defines the experiment. Ignore previous runs' extra cached files.
    files = [Path(args.cache)/(s["token"]+".pt") for s in manifest["scenes"]]
    if not files or not all(f.is_file() for f in files):
        raise ValueError("Complete every manifest scene's training cache first.")
    logs = sorted({s["log"] for s in manifest["scenes"]})
    if len(logs) < 2:
        raise ValueError("Need separate training and validation logs.")
    val_logs = set(logs[-max(1, len(logs)//5):])
    train_data, val_data = [], []
    for file in files:
        value = torch.load(file, map_location="cpu", weights_only=True)
        (val_data if value["log"] in val_logs else train_data).append(value)
    if not train_data or not val_data:
        raise ValueError("Empty training/validation partition.")
    architecture = {"dim": 512, "rank": args.rank, "use_age": not args.no_age}
    adapter = ReliabilityAdapter(**architecture).to("cuda")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")

    def loss_for(item, index, training):
        x = item["corrupt"][index:index+1].to("cuda", torch.float32)
        y = item["clean"].to("cuda", torch.float32)
        status = item["status"][index:index+1].to("cuda", torch.float32)
        pred = adapter(x, status)
        meta = status.reshape(1, 2, 2, 4, 2).mean(2).permute(0, 2, 1, 3)
        mask = ((meta[..., 0]<1) | (meta[..., 1]>0)).float()[..., None, None]
        delta = (pred-y).reshape(1,4,2,512,512)
        loss = (delta.square()*mask).sum() / (mask.sum()*512*512).clamp_min(1)
        raw = ((x-y).reshape_as(delta).square()*mask).sum() / (mask.sum()*512*512).clamp_min(1)
        return loss, raw

    for epoch in range(args.epochs):
        adapter.train()
        losses = []
        for idx in rng.permutation(len(train_data)):
            item = train_data[idx]
            fault_index = int(rng.integers(len(item["corrupt"])))
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_for(item, fault_index, True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        adapter.eval()
        validation, raw_errors = [], []
        with torch.no_grad():
            for item in val_data:
                for idx in range(len(item["corrupt"])):
                    loss, raw = loss_for(item, idx, False)
                    validation.append(float(loss))
                    raw_errors.append(float(raw))
        row = {"epoch": epoch+1, "train_mse": float(np.mean(losses)),
               "val_mse": float(np.mean(validation)), "val_uncorrected_mse": float(np.mean(raw_errors))}
        history.append(row)
        if row["val_mse"] < best:
            best = row["val_mse"]
            torch.save({"architecture": architecture, "state_dict": adapter.state_dict(),
                        "seed": args.seed, "epoch": epoch+1, "selection_metric": "held-out-log context MSE",
                        "train_logs": [l for l in logs if l not in val_logs], "val_logs": sorted(val_logs)},
                       out/"adapter.pt")
        dump(out/"training.json", {"parameters": sum(p.numel() for p in adapter.parameters()),
                                   "train_scenes": len(train_data), "val_scenes": len(val_data),
                                   "train_logs": [l for l in logs if l not in val_logs],
                                   "val_logs": sorted(val_logs), "history": history})
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "cache"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--limit", type=int)
        if name == "evaluate":
            p.add_argument("--adapter")
    p = sub.add_parser("train")
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-age", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "train":
        train(args)
    else:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        (evaluate if args.command == "evaluate" else cache)(args, config)


if __name__ == "__main__":
    main()
