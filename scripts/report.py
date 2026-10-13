"""Build paired tables and trajectory examples from committed per-scene results."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wajepa_robust.metrics import paired_log_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.baseline/"predictions.json").read_text())
    candidate = json.loads((args.candidate/"predictions.json").read_text())
    left = {(r["token"],r["condition"]):r for r in baseline}
    right = {(r["token"],r["condition"]):r for r in candidate}
    if left.keys() != right.keys():
        raise ValueError("Baseline/candidate tokens and faults must match exactly.")
    args.output.mkdir(parents=True, exist_ok=True)
    conditions = sorted({r["condition"] for r in baseline})
    report = {}
    for condition in conditions:
        raw = [r for r in baseline if r["condition"]==condition]
        adapted = [r for r in candidate if r["condition"]==condition]
        paired = [{**r,"condition":"baseline"} for r in raw]
        paired += [{**r,"condition":"adapter"} for r in adapted]
        report[condition] = paired_log_bootstrap(paired,"baseline","adapter")
        report[condition].update({"baseline_ade_m":float(np.mean([r['ade_m'] for r in raw])),
                                  "adapter_ade_m":float(np.mean([r['ade_m'] for r in adapted]))})
    clean_drift = max(np.max(np.abs(np.array(left[k]["prediction"])-np.array(right[k]["prediction"])))
                      for k in left if k[1]=="clean")
    report["clean_prediction_max_abs_delta"] = float(clean_drift)
    (args.output/"paired_comparison.json").write_text(json.dumps(report,indent=2))
    fig, axes = plt.subplots(1,2,figsize=(12,4.6))
    x = np.arange(len(conditions));width=.36
    axes[0].bar(x-width/2,[report[c]['baseline_ade_m'] for c in conditions],width,label='Official WA-JEPA')
    axes[0].bar(x+width/2,[report[c]['adapter_ade_m'] for c in conditions],width,label='Context adapter')
    axes[0].set_xticks(x,[c.replace('_v1_d',' / ') for c in conditions],rotation=20,ha='right')
    axes[0].set_ylabel('4-second ADE (m)');axes[0].legend(fontsize=8)
    # Fixed selection: lexicographically first token, burst-loss condition.
    token=sorted({r['token'] for r in baseline})[0];key=(token,'hold_v1_d1')
    for label,values in [('Ground truth',left[key]['target']),('Official WA-JEPA',left[key]['prediction']),('Context adapter',right[key]['prediction'])]:
        trajectory=np.asarray(values);axes[1].plot(trajectory[:,0],trajectory[:,1],'.-',label=label)
    axes[1].set_aspect('equal',adjustable='datalim');axes[1].set_xlabel('Ego-forward x (m)');axes[1].set_ylabel('Ego-left y (m)')
    axes[1].set_title(f'Burst loss: {token}',fontsize=9);axes[1].legend(fontsize=8)
    fig.suptitle('Development smoke subset — offline metrics, not official NAVSIM EPDMS')
    fig.tight_layout();fig.savefig(args.output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
