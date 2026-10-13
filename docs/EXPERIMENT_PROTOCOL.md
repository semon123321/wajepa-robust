# Experiment protocol, version 0.1 (2026-10-13)

## Scope and attribution

Reproduce the **official pretrained inference path** of WA-JEPA (August 21,
2026; arXiv 2608.20974). This project does not reproduce foundation pretraining
and does not claim the paper's full NAVSIM EPDMS result. The upstream model,
checkpoint, latent flow sampler and planning architecture belong to its authors.
Our code supplies data/protocol checks, causal camera corruption, a small
reliability-conditioned context adapter, and paired evaluation.

## Hypothesis and intervention

Missing or stale camera histories may distort encoder scene tokens and planned
ego trajectories. Freeze the complete official model. Train a rank-32 residual
adapter to reconstruct clean **history** scene tokens from corrupted histories,
using pooled unaffected-camera context, camera identity, availability and frame
age. The adapter's clean-input path is exactly identity; its output remains the
input to the unchanged official joint scene/action flow sampler.

This is context distillation, not future-latent pretraining. Future RGB and future
ego poses are absent from the adapter training cache. The approach is an
engineering research hypothesis, without a global novelty claim. A reduction in
latent MSE does not establish a reduction in trajectory error.

## Data and leakage controls

- Official OpenScene, based on nuPlan; four views in upstream order L0/F0/R0/B0.
- Training smoke data: mini camera shard 5; intersect with NAVSIM v2.2 navtrain.
- Evaluation smoke data: test camera shard 30; intersect with v2.2 navtest.
- Four history frames at 2 Hz; eight predicted poses (4 seconds). Preserve the
  official filter's additional future frames for scene eligibility.
- Record every token/log and filter hash. Manifest overlaps must be rejected.
- Adapter train/validation division is by **log**, never random overlapping
  windows. Select checkpoints using validation context MSE only. Test trajectory
  metrics must not choose training epochs or hyperparameters.
- Navigation is the available metadata command; no inference from future poses.
- Apply the identical faults to paired baseline/adapter evaluations and keep
  inference seed 1, four sampling steps, BF16, original image size 256x512.

The initial small shards are deliberately development smoke data. Do not report
these as representative full NAVSIM performance. Expansion to >=10 independent
test logs and separate held-out development logs is a gate before resume claims.

## Faults

1. Clean input.
2. Entire front camera history blacked out (availability=0).
3. Front camera histories delayed by one observed interval (0.5 s).
4. Delay by two intervals (1 s).
5. Last two front frames missing; repeat the most recently available frame.

Delays clamp at the oldest historical frame at startup. Report this boundary
policy rather than pretending a missing pre-window frame is available. Metadata
age is the realized source-to-target offset. Faults are synthetic interventions
on real driving RGB; they are not naturally recorded fault events.

## Baselines and ablations

- Official model, clean and each corruption.
- Simple last-frame hold (the burst fault input policy).
- Reliability adapter with binary fault metadata (no continuous age).
- Full availability/age adapter.
- Three training seeds; identical logs and official sampling seed.
- Additional future experiments: unseen cameras/combinations, two-camera loss,
  variable delay, RGB photometric changes without known metadata.

A simple global replacement of fault tokens by healthy-camera means may be added
as a non-learned comparator; never transfer front-left pixels as front pixels
without camera geometry.

## Metrics and decision rules

- Offline ADE/FDE in meters, eight 0.5-second points; L2 at 1/2/3 s.
- Clean trajectory equality and clean-regression check.
- Paired log-bootstrap delta ADE, resampling logs due to correlated windows.
- Inference wall time after warmup, CUDA synchronized, excluding disk/image IO;
  GPU/precision/peak reserved memory recorded. No real-time deployment claim.
- Official EPDMS with NAVSIM maps/scorer is a separate later gate; offline
  displacement is not a safety, collision, simulation or closed-loop score.

Provisional acceptance target: reduce degraded ADE relative to the exact official
model, show consistent gains over the binary fault adapter across three seeds,
and preserve clean predictions. Confidence intervals on three logs are weak.
If feature reconstruction improves but planning does not, report the negative
result and investigate objectives; do not substitute latent MSE for planning.

## Four-week delivery gates

1. Week 1: strict checkpoint reproduction, real data manifests, baseline faults,
   cache/adapter training, experiment log and failure examples.
2. Week 2: expand training/log diversity; held-out-log development; improve
   objective with frozen predictor action consistency if local memory allows.
3. Week 3: three seeds, age/camera/context ablations, official EPDMS scoring on a
   disclosed subset, speed/memory profiling. Stop changes before final test.
4. Week 4: frozen test run, limitations, reproducible demo, 5-minute explanation,
   interview questions and factual resume entry.
