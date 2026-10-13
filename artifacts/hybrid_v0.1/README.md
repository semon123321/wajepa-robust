# Personally trained adapter, research snapshot v0.1

`adapter.safetensors` contains only 104,064 personally trained adapter parameters
(two 52,032-parameter branches). It does not contain the official encoder,
world/action predictor, scene projector or other upstream model parameters.

The branches were trained using the official frozen WA-JEPA model. Metadata in
`adapter_config.json` records their train/validation logs and disclosed routing.
Original world/action weights must be downloaded separately from their official
source. See NOTICE.md for attribution and licensing boundaries.

Use this artifact to reproduce the **snapshot**, not as a validated vehicle
deployment model. The hand-written routing was selected on development logs.
Its clean input path is identity. Current supported batch size is 1.

```bash
pip install -e ".[export]"
wajepa-robust evaluate --config local_test.json --output runs/artifact_eval --adapter artifacts/hybrid_v0.1/adapter.safetensors
```
