"""Package two validation-selected adapters and a disclosed development routing."""
import argparse
from pathlib import Path
import torch
from wajepa_robust.hybrid import HybridAdapter


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--context",type=Path,required=True)
    parser.add_argument("--action",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    context=torch.load(args.context,map_location="cpu",weights_only=True)
    action=torch.load(args.action,map_location="cpu",weights_only=True)
    model=HybridAdapter(context['architecture'],action['architecture'])
    model.context.load_state_dict(context['state_dict'],strict=True)
    model.action.load_state_dict(action['state_dict'],strict=True)
    bundle={'kind':'hybrid','context_architecture':context['architecture'],
            'action_architecture':action['architecture'],'state_dict':model.state_dict(),
            'train_logs':sorted(set(context['train_logs'])|set(action['train_logs'])),
            'val_logs':sorted(set(context['val_logs'])|set(action['val_logs'])),
            'routing':'missing -> action distillation; burst old frames -> fixed reliability; other stale -> context distillation',
            'development_selection':'Routing chosen with disclosed dev_v1 three logs; held-out logs not used.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(bundle,args.output)
    print(f'packaged {sum(p.numel() for p in model.parameters())} parameters')


if __name__=='__main__':
    main()
