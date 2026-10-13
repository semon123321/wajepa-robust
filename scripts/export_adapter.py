"""Export only personally trained adapter tensors, excluding upstream weights."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    checkpoint=torch.load(args.adapter,map_location='cpu',weights_only=True)
    state=checkpoint['state_dict']
    # No encoder, predictor, official scene projectors or future tokens permitted.
    if any(any(name in key for name in ['encoder','predictor','scene_projector','future_mask']) for key in state):
        raise ValueError('Refusing to export official model weights as personal adapter.')
    args.output.mkdir(parents=True,exist_ok=True)
    np_state={k:v.cpu().contiguous() for k,v in state.items()}
    from safetensors.torch import save_file
    save_file(np_state,str(args.output/'adapter.safetensors'))
    metadata={k:v for k,v in checkpoint.items() if k!='state_dict'}
    metadata['source_pt_sha256']=hashlib.sha256(args.adapter.read_bytes()).hexdigest()
    metadata['safetensors_sha256']=hashlib.sha256((args.output/'adapter.safetensors').read_bytes()).hexdigest()
    (args.output/'adapter_config.json').write_text(json.dumps(metadata,indent=2))
    print(args.output)


if __name__=='__main__':main()
