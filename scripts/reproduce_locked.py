"""Reproduce published inference using exact test tokens and adapter weights.

Weights may be trained locally or exported from this project's local runs. This
script validates their published checksums before invoking inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--report',type=Path,default=Path('reports/holdout_v1'))
    parser.add_argument('--output',type=Path,default=Path('runs/reproduce_locked'))
    parser.add_argument('--adapter',type=Path)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    manifest=json.loads((args.report/'locked_manifest.json').read_text())
    config['tokens']=[s['token'] for s in manifest['scenes']]
    config['logs']=sorted({s['log'] for s in manifest['scenes']})
    config['stride']=1
    args.output.mkdir(parents=True,exist_ok=True)
    local=args.output/'local_locked.json';local.write_text(json.dumps(config,indent=2))
    if args.adapter:
        protocol=json.loads((args.report/'frozen_protocol.json').read_text())
        expected=protocol['hybrid_safetensors_sha256'] if args.adapter.suffix=='.safetensors' else protocol['hybrid_sha256']
        if hashlib.sha256(args.adapter.read_bytes()).hexdigest()!=expected:
            raise ValueError('Adapter checksum differs from published locked experiment.')
    command=[sys.executable,'-m','wajepa_robust.cli','evaluate','--config',str(local),'--output',str(args.output)]
    if args.adapter:command+=['--adapter',str(args.adapter)]
    subprocess.run(command,check=True)
    actual=json.loads((args.output/'manifest.json').read_text())
    if {s['token'] for s in actual['scenes']}!={s['token'] for s in manifest['scenes']}:
        raise ValueError('Incomplete sensor download: evaluated token set differs from locked manifest.')


if __name__=='__main__':main()
