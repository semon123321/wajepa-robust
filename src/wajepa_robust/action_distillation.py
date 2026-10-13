"""Train through the frozen joint world/action predictor on clean teacher states.

Only the small context adapter receives parameter gradients. Teacher futures are
model-generated, not ground truth future RGB. This supplements context MSE with
physical-unit action consistency and sampled future-latent consistency.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .adapter import ReliabilityAdapter
from .model import OfficialModel


class FrozenFlowTeacher:
    def __init__(self, official):
        self.model = official.model
        # Encoding was cached; offload the two unused large encoders for backprop memory.
        self.model.encoder.cpu()
        self.model.target_encoder.cpu()
        torch.cuda.empty_cache()
        masks = self.model.mask_sampler.full_future_mask(1, device=torch.device("cuda"))
        self.indices = self.model._history_indices_from_masks(masks.masks_x[0], 1)
        self.condition = self.model._future_mask_condition(1, device=torch.device("cuda"), dtype=torch.bfloat16)

    def predict(self, context, rollout, step):
        m = self.model
        state = rollout["steps"][step]
        features = {k: rollout[k].to("cuda", torch.bfloat16) for k in ("history_trajectory","ego_status")}
        conditions = m.driving_condition_adapter.inference_conditions(features)
        inputs = m.driving_condition_adapter.prepare_inference_inputs(
            m, conditions, dtype=torch.bfloat16, device=torch.device("cuda"))
        inputs["noisy_trajectory"] = state["trajectory"].to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m.predictor(context_scene=context.to(torch.bfloat16), context_token_indices=self.indices,
                               noisy_future_scene=state["scene"].to("cuda"), future_condition_scene=self.condition,
                               t_cont=state["t"].to("cuda"), trajectory_inputs=inputs)


def losses(adapter, teacher, item, fault_index, step):
    x = item["corrupt"][fault_index:fault_index+1].to("cuda", torch.float32)
    y = item["clean"].to("cuda", torch.float32)
    status = item["status"][fault_index:fault_index+1].to("cuda", torch.float32)
    context = adapter(x, status)
    scene, action = teacher.predict(context, item["teacher_rollout"], step)
    target = item["teacher_rollout"]["steps"][step]
    predicted_action = teacher.model._denormalize_trajectory(action.float())
    target_action = teacher.model._denormalize_trajectory(target["target_action"].to("cuda", torch.float32))
    action_loss = F.smooth_l1_loss(predicted_action[..., :2], target_action[..., :2])
    future_indices = item["teacher_rollout"]["scene_indices"].to("cuda")
    world_loss = F.mse_loss(scene[:, future_indices].float(), target["target_scene"].to("cuda", torch.float32))
    meta = status.reshape(1,2,2,4,2).mean(2).permute(0,2,1,3)
    mask = ((meta[...,0]<1)|(meta[...,1]>0)).float()[...,None,None]
    context_loss = ((context-y).reshape(1,4,2,512,512).square()*mask).sum() / (mask.sum()*512*512).clamp_min(1)
    return action_loss, world_loss, context_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-age", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads(Path(args.config).read_text())
    manifest = json.loads((args.cache/"manifest.json").read_text())
    logs = sorted({s["log"] for s in manifest["scenes"]})
    if len(logs)<2:
        raise ValueError("Need independent train and validation logs.")
    val_logs = set(logs[-max(1,len(logs)//5):])
    train_files, val_files = [], []
    for info in manifest["scenes"]:
        (val_files if info["log"] in val_logs else train_files).append(args.cache/(info["token"]+".pt"))
    official = OfficialModel(config["upstream"],config["checkpoint"])
    teacher = FrozenFlowTeacher(official)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    architecture = {"dim":512,"rank":32,"use_age":not args.no_age}
    adapter = ReliabilityAdapter(**architecture).to("cuda")
    optimizer = torch.optim.AdamW(adapter.parameters(),lr=args.lr,weight_decay=.01)
    args.output.mkdir(parents=True,exist_ok=True)
    history = []
    best = float('inf');updates=0;started=time.monotonic()

    def read(file):
        item = torch.load(file,map_location="cpu",weights_only=True)
        if "teacher_rollout" not in item:
            raise ValueError("Run cache --with-action first.")
        return item

    if args.probe_only:
        item=read(train_files[0]);rollout=item['teacher_rollout']
        with torch.no_grad():
            _,action=teacher.predict(item['clean'].to('cuda'),rollout,0)
            max_delta=float((action.cpu()-rollout['steps'][0]['target_action']).abs().max())
        torch.cuda.reset_peak_memory_stats()
        a,w,c=losses(adapter,teacher,item,4,0)
        (a+.1*w+.1*c).backward()
        grad=float(adapter.up.weight.grad.norm())
        result={'clean_teacher_forward_max_delta':max_delta,'adapter_gradient_norm':grad,
                'frozen_model_gradient_tensors':sum(p.grad is not None for p in teacher.model.parameters()),
                'peak_reserved_gb':torch.cuda.max_memory_reserved()/1e9}
        if max_delta > .02 or not np.isfinite(grad) or grad<=0:
            raise RuntimeError(result)
        (args.output/'probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        return

    for epoch in range(args.epochs):
        adapter.train();training=[]
        for index in rng.permutation(len(train_files)):
            item=read(train_files[index]);fault=int(rng.integers(16));step=int(rng.integers(4))
            optimizer.zero_grad(set_to_none=True)
            a,w,c=losses(adapter,teacher,item,fault,step)
            total=a+.1*w+.1*c
            if not torch.isfinite(total):
                raise RuntimeError("Non-finite loss; abort rather than save an invalid adapter.")
            total.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(),1);optimizer.step();updates+=1
            training.append([float(a.detach()),float(w.detach()),float(c.detach())])
            if updates%10==0:
                print(f'updates {updates}: action={float(a.detach()):.4f}',flush=True)
            if args.max_steps and updates>=args.max_steps:
                break
        adapter.eval();validation=[]
        with torch.no_grad():
            # Fixed faults/steps, chosen without trajectory test labels.
            for index,file in enumerate(val_files):
                item=read(file);a,w,c=losses(adapter,teacher,item,index%16,index%4)
                validation.append([float(a),float(w),float(c)])
        train_mean=np.mean(training,axis=0);val_mean=np.mean(validation,axis=0)
        row={'epoch':epoch+1,'updates':updates,'train_action':float(train_mean[0]),
             'val_action':float(val_mean[0]),'val_future_mse':float(val_mean[1]),
             'val_context_mse':float(val_mean[2])};history.append(row)
        # Selection is by held-out-log teacher action consistency, never test ADE.
        if row['val_action']<best:
            best=row['val_action']
            torch.save({'architecture':architecture,'state_dict':adapter.state_dict(),
                        'seed':args.seed,'epoch':epoch+1,'objective':'clean-teacher flow action/world/context distillation',
                        'selection_metric':'held-out-log teacher action smooth L1',
                        'train_logs':[l for l in logs if l not in val_logs],'val_logs':sorted(val_logs)},args.output/'adapter.pt')
        result={'parameters':sum(p.numel() for p in adapter.parameters()),'train_scenes':len(train_files),
                'seed':args.seed,'lr':args.lr,'use_age':not args.no_age,'weight_decay':.01,
                'val_scenes':len(val_files),'train_logs':[l for l in logs if l not in val_logs],
                'val_logs':sorted(val_logs),'history':history,'wall_time_s':time.monotonic()-started,
                'loss_weights':{'action':1,'sampled_future':.1,'context':.1},
                'peak_reserved_gb':torch.cuda.max_memory_reserved()/1e9}
        (args.output/'training.json').write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
        if args.max_steps and updates>=args.max_steps:
            break


if __name__=='__main__':
    main()
