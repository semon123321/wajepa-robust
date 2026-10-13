import numpy as np


def source_drive(log):
    """OpenScene segments from the same original nuPlan drive are correlated."""
    parts = log.rsplit("_", 2)
    return parts[0] if len(parts)==3 and parts[1].isdigit() and parts[2].isdigit() else log


def displacement(prediction, target):
    error = np.linalg.norm(np.asarray(prediction)[..., :2] - np.asarray(target)[..., :2], axis=-1)
    return {"ade_m": float(error.mean()), "fde_m": float(error[-1]),
            "l2_1s_m": float(error[1]), "l2_2s_m": float(error[3]), "l2_3s_m": float(error[5])}


def summarize(rows):
    output = {}
    for condition in sorted({r["condition"] for r in rows}):
        group = [r for r in rows if r["condition"] == condition]
        output[condition] = {"n": len(group), "logs": len({r["log"] for r in group}),
                             "source_drives": len({source_drive(r["log"]) for r in group})}
        for key in ("ade_m", "fde_m", "l2_1s_m", "l2_2s_m", "l2_3s_m", "latency_s"):
            output[condition][key] = float(np.mean([r[key] for r in group]))
    return output


def paired_log_bootstrap(rows, baseline, candidate, repetitions=2000):
    """Paired delta candidate - baseline, resampling LOGS (overlapping clips correlate)."""
    left = {r["token"]: r for r in rows if r["condition"] == baseline}
    right = {r["token"]: r for r in rows if r["condition"] == candidate}
    if left.keys() != right.keys() or not left:
        raise ValueError("Paired comparison requires identical nonempty token sets.")
    groups = {}
    for token, l in left.items():
        groups.setdefault(source_drive(l["log"]), []).append(right[token]["ade_m"] - l["ade_m"])
    rng = np.random.default_rng(42)
    logs = list(groups)
    samples = []
    for _ in range(repetitions):
        chosen = rng.choice(logs, len(logs), replace=True)
        samples.append(np.mean([d for log in chosen for d in groups[log]]))
    return {"delta_ade_m": float(np.mean([d for g in groups.values() for d in g])),
            "log_bootstrap_95pct": np.quantile(samples, [.025, .975]).tolist(),
            "source_drive_clusters": len(logs), "log_segments": len({r['log'] for r in left.values()}),
            "caution": "Few logs give unreliable intervals; smoke evidence only." if len(logs) < 10 else None}
