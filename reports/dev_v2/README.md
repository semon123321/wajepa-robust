# Development experiment 2: task alignment and non-learned control

**Small changes, no demonstrated continuous-age advantage.**

Same 27 development scenes, three logs, five conditions, fixed seed and official
model as dev_v1. These logs are now explicitly development data because their
results motivated the following hypothesis. Held-out test logs are separate.

The action adapters use a larger training cache: 60 scenes from four navtrain
logs, split into 45 training scenes / three logs and 15 validation scenes / one
log. Both models have 52,032 parameters, rank 32, AdamW lr=1e-4, weight decay=.01,
seed 42, 240 optimizer updates, and the identical sampled fault/flow-step schedule.
Both checkpoints were selected at epoch 2 by held-out-log teacher action loss.
The continuous-age version selected loss 0.102050; binary fault version 0.101942.

| Front-camera condition | Official | Context only¹ | Action + age | Action + binary fault | Fixed latent reliability² |
|---|---:|---:|---:|---:|---:|
| Clean | .477950 | .477950 | .477950 | .477950 | .477950 |
| Blackout | 1.167779 | 1.358287 | 1.158435 | 1.160489 | 1.331696 |
| Stale 0.5 s | .599846 | .583475 | .588709 | .588412 | .609827 |
| Stale 1.0 s | .829519 | .768469 | .814164 | .811127 | .909693 |
| Burst loss / hold | 1.308216 | 1.246887 | 1.321781 | 1.321046 | 1.132050 |

All values are 4-second ADE in meters. ¹ Context-only is the smaller training
experiment from dev_v1; it is a separate pilot, not an objective-only ablation
with matched training data. ² The fixed control scales latent tokens by
availability × exp(-age/0.5 s), filling removed information with zero latent
vectors. This is distinct from black raw pixels and from repeating raw frames.

Mean corrupted-condition ADE: official 0.976340, task-aligned age adapter
approximately 0.970772, binary adapter 0.970269. These small differences do not
establish a useful continuous-age advantage. The fixed control helps burst loss
but hurts blackout and persistent stale history. No single intervention wins
every condition. Clean predictions remain identical.

## Frozen next hypothesis

Use **observable sensor metadata** to route the interventions: missing camera
→ action-distilled adapter; early history current and recent history stale
→ fixed reliability suppression; other stale history → context-distilled
adapter. This is a hand-written, development-selected rule, not a learned fault
detector or a globally novel architecture. It currently supports batch size 1.

The two adapters together contain 104,064 parameters. The routing uses no future
RGB, target trajectory, scene category, or fault-name argument. Its behavior and
weights were frozen in `reports/holdout_v1/frozen_protocol.json` before examining
new-log model predictions. All-condition results on the new logs determine
whether development-selected routing generalizes. Unseen fault patterns and
larger independent-log counts remain necessary follow-up tests.

The original context-only blackout regression is retained in dev_v1. The present
tables must not be described as full NAVSIM EPDMS or closed-loop driving results.
