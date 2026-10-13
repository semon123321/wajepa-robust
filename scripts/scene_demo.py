"""Local real-camera/trajectory figure. Does not redistribute dataset imagery."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wajepa_robust.data import OpenSceneSubset, CAMERAS
from wajepa_robust.corruptions import Fault,apply_fault


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--token');args=p.parse_args()
    config=json.loads(Path(args.config).read_text())
    b=json.loads((args.baseline/'predictions.json').read_text());a=json.loads((args.candidate/'predictions.json').read_text())
    token=args.token or sorted({r['token'] for r in b})[0]
    data=OpenSceneSubset(config['metadata'],config['sensors'],config['scene_filter'],tokens=[token])
    features,target,_=data[0]
    corrupted,status=apply_fault(features,Fault('hold'))
    fig=plt.figure(figsize=(14,7));grid=fig.add_gridspec(2,4,height_ratios=[1,1.5])
    for view,cam in enumerate(CAMERAS):
        ax=fig.add_subplot(grid[0,view]);rgb=corrupted['history_images'][0,-1,view].permute(1,2,0).numpy()
        ax.imshow(np.clip((rgb+1)/2,0,1));ax.axis('off')
        age=float(status[0,-1,view,1]);ax.set_title(f'{cam} / image age {age:.1f}s',fontsize=10)
    ax=fig.add_subplot(grid[1,:2])
    condition='hold_v1_d1'
    br=next(r for r in b if r['token']==token and r['condition']==condition)
    ar=next(r for r in a if r['token']==token and r['condition']==condition)
    for label,trajectory in [('Ground truth',target),('Official WA-JEPA',br['prediction']),('Sensor-aware adapter',ar['prediction'])]:
        trajectory=np.asarray(trajectory);ax.plot(trajectory[:,0],trajectory[:,1],'.-',label=label)
    ax.set_xlabel('Ego-forward x (m)');ax.set_ylabel('Ego-left y (m)');ax.grid(alpha=.2);ax.legend(fontsize=9)
    ax.set_title('Four-second trajectory / front-camera burst loss')
    ax2=fig.add_subplot(grid[1,2:]);ax2.axis('off')
    ax2.text(.03,.9,'WA-JEPA Robust — real OpenScene data',fontsize=14)
    lines=[f'Token: {token}',f'Official ADE: {br["ade_m"]:.3f} m',f'Adapter ADE: {ar["ade_m"]:.3f} m',
           'Four cameras, four historical frames at 2 Hz.',
           'Last two front-camera frames repeat an older image.',
           'Teacher/model parameters are frozen.',
           'Displayed example chosen by token order, not best gain.',
           'Offline trajectory demonstration; no collision/EPDMS claim.']
    ax2.text(.03,.76,'\n\n'.join(lines),fontsize=10,va='top')
    fig.tight_layout();args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160);print(args.output)


if __name__=='__main__':main()
