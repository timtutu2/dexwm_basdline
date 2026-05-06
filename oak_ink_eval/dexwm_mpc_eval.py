#!/usr/bin/env python3
"""
DexWM CEM-MPC evaluation — no IsaacGym required.

Uses pre-rendered RGB frames for context.
Outputs per-step frame selection and CEM loss to a CSV.

Usage:
    PYTHONPATH=. python oak_ink_eval/dexwm_mpc_eval.py \
        --data_root      /home/yulun/projects/ManipTrans/data \
        --data_idx       083f7@0 \
        --dexwm_checkpoint data/checkpoints/oakink2_maniptrans_multistep_ft_epoch44.pth.tar \
        --dexwm_config   configs/oakink2_multistep_finetune.yaml \
        --goal_image_path data/oakink2_processed/rgb/001196.png \
        --rgb_dir        data/oakink2_processed/rgb \
        --output_dir     output/dexwm_mpc_eval/083f7

"""

import sys, os, argparse, pickle, yaml, csv, time
from pathlib import Path

import numpy as np
import torch
import cv2
import torchvision.transforms.functional as TF
from scipy.spatial.transform import Rotation as ScipyR

DEXWM_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, DEXWM_ROOT)
from models.model import DexWM


# ─────────────────────────────── backbone ─────────────────────────────────────

def get_patch_size_from_backbone(backbone_name):
    if 'dinov2' in backbone_name or 'siglip' in backbone_name or 'webssl' in backbone_name:
        return 14, 448
    elif 'dinov3' in backbone_name or 'vjepa' in backbone_name:
        return 16, 336
    raise ValueError(f'Backbone {backbone_name} not supported')


# ─────────────────────────────── camera / pose ────────────────────────────────

_CAM_POS    = np.array([-0.25, -0.12, 0.95], dtype=np.float64)
_CAM_TARGET = np.array([-0.15, -0.12, 0.70], dtype=np.float64)
_WORLD_UP   = np.array([ 0.00,  0.00,  1.00], dtype=np.float64)

def _build_cam_extrinsic():
    z = _CAM_TARGET - _CAM_POS;  z /= np.linalg.norm(z)
    x = np.cross(_WORLD_UP, z);  x /= np.linalg.norm(x)
    y = np.cross(z, x);          y /= np.linalg.norm(y)
    rot = np.stack([x, y, z], axis=0)
    extr = np.eye(4, dtype=np.float64)
    extr[:3, :3] = rot
    extr[:3,  3] = -rot @ _CAM_POS
    return extr.astype(np.float32)

_CAM_EXTR      = _build_cam_extrinsic()
_CAM_POS_F32   = _CAM_POS.astype(np.float32)
_CAM_ROT_EULER = ScipyR.from_matrix(_CAM_EXTR[:3, :3].T).as_euler('xyz').astype(np.float32)

def joints_to_cam(joints_world):
    ones  = np.ones((len(joints_world), 1), dtype=np.float32)
    j_hom = np.concatenate([joints_world.astype(np.float32), ones], axis=1)
    return (_CAM_EXTR @ j_hom.T).T[:, :3]

def get_pose44(rh_joints, lh_joints):
    rh_cam = joints_to_cam(rh_joints)
    lh_cam = joints_to_cam(lh_joints)
    rh_pad = np.concatenate([rh_cam, rh_cam[-3:]], axis=0)
    lh_pad = np.concatenate([lh_cam, lh_cam[-3:]], axis=0)
    return np.concatenate([lh_pad, rh_pad, _CAM_POS_F32[None], _CAM_ROT_EULER[None]], axis=0)


# ─────────────────────────────── data loading ─────────────────────────────────

def _resolve_anno_stem(seq_hash, anno_dir):
    matches = [f for f in os.listdir(anno_dir) if seq_hash in f]
    assert len(matches) == 1
    enc_stem = os.path.splitext(matches[0])[0]
    return enc_stem, enc_stem.replace("%2B", "+")

def load_retargeted_pkl(data_idx, side, data_root):
    seq_hash, stage = data_idx.split("@")
    anno_dir = os.path.join(data_root, "OakInk-v2", "anno_preview")
    enc_stem, dec_stem = _resolve_anno_stem(seq_hash, anno_dir)
    pkl_dir = os.path.join(data_root, "retargeting", "OakInk-v2", f"mano2inspire_{side}")
    for stem in (dec_stem, enc_stem):
        p = os.path.join(pkl_dir, f"{stem}@{stage}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"pkl not found in {pkl_dir}")


# ─────────────────────────────── DexWM ────────────────────────────────────────

def load_dexwm(config_path, checkpoint_path, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    backbone_name = cfg["model"]["backbone_name"]
    patch_size, num_patches = get_patch_size_from_backbone(backbone_name)
    img_width = 392 if patch_size == 14 else 384
    model = DexWM(
        backbone_name=backbone_name,
        num_patches=num_patches,
        patch_size=patch_size,
        hidden_dim=cfg["model"]["hidden_dim"],
        action_dim=cfg["model"]["action_dim"],
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
        is_eval=True,
        num_context=cfg["data"]["num_context"],
        emb_loss_fn=torch.nn.MSELoss(reduction="mean"),
    )
    ckpt  = torch.load(checkpoint_path, map_location="cpu")["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(state)
    model.eval().to(device)
    return model, cfg, img_width, backbone_name

def img_to_tensor(img_bgr, img_size, img_width, backbone_name):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if h != img_size:
        img = cv2.resize(img, (int(img_size * w / h), img_size), interpolation=cv2.INTER_LINEAR)
    cur_w = img.shape[1]
    if cur_w < img_width:
        pad = img_width - cur_w
        img = np.pad(img, ((0,0),(pad//2, pad-pad//2),(0,0)), mode="constant")
    elif cur_w > img_width:
        crop = cur_w - img_width
        img  = img[:, crop//2: crop//2+img_width]
    t = torch.from_numpy(img.copy()).permute(2,0,1).float() / 255.0
    if "siglip" in backbone_name:
        t = TF.normalize(t, [0.5,0.5,0.5], [0.5,0.5,0.5])
    else:
        t = TF.normalize(t, [0.485,0.456,0.406], [0.229,0.224,0.225])
    return t


# ─────────────────────────────── CEM MPC ──────────────────────────────────────

@torch.inference_mode()
def score_candidates(model, context_imgs, curr_pose44, cand_poses, goal_emb, num_context, device):
    """Score one candidate at a time to keep GPU memory bounded."""
    deltas = (cand_poses - curr_pose44[None]).astype(np.float32)
    ctx    = context_imgs.unsqueeze(0).to(device)          # (1, T+1, 3, H, W)
    rel_t  = torch.zeros(1, num_context, dtype=torch.float32, device=device)
    losses = []
    for d in deltas:
        actions = torch.zeros(1, num_context, 44, 3, dtype=torch.float32, device=device)
        actions[0, -1] = torch.from_numpy(d).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            x_pred, _, _, _, _ = model(ctx, actions, rel_t, action_diff=True)
        loss = torch.nn.functional.mse_loss(
            x_pred[0, -1].float(), goal_emb[0], reduction="mean"
        )
        losses.append(loss.item())
    return np.array(losses, dtype=np.float32)

def cem_plan(model, context_imgs, curr_pose44, goal_emb, num_context, device,
             num_samples=512, opt_steps=5, topk=10, sigma=0.02):
    mu  = np.zeros((44, 3), dtype=np.float32)
    std = np.full((44, 3), sigma, dtype=np.float32)
    best_loss = np.inf
    for _ in range(opt_steps):
        deltas     = (mu[None] + std[None] * np.random.randn(num_samples, 44, 3)).astype(np.float32)
        cand_poses = curr_pose44[None] + deltas
        losses     = score_candidates(model, context_imgs, curr_pose44, cand_poses,
                                      goal_emb, num_context, device)
        elite_idx  = np.argsort(losses)[:topk]
        elite      = deltas[elite_idx]
        mu         = elite.mean(0)
        std        = elite.std(0) + 1e-6
        best_loss  = float(losses[elite_idx[0]])
    return (curr_pose44 + mu).astype(np.float32), best_loss

def find_nearest_gt_frame(target_pose44, all_poses, lo, hi):
    lo = max(lo, 0); hi = min(hi, len(all_poses))
    if lo >= hi:
        return lo
    dists = np.mean((all_poses[lo:hi] - target_pose44[None]) ** 2, axis=(1, 2))
    return lo + int(np.argmin(dists))


# ─────────────────────────────── main ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",         default="/home/yulun/projects/ManipTrans/data")
    p.add_argument("--data_idx",          default="083f7@0")
    p.add_argument("--dexwm_checkpoint",  required=True)
    p.add_argument("--dexwm_config",      required=True)
    p.add_argument("--goal_image_path",   required=True)
    p.add_argument("--rgb_dir",           required=True,
                   help="Directory of pre-rendered frames named 000000.png, 000001.png, …")
    p.add_argument("--output_dir",        default="output/dexwm_mpc_eval")
    p.add_argument("--search_window",     type=int,   default=50)
    p.add_argument("--cem_samples",       type=int,   default=64)
    p.add_argument("--cem_steps",         type=int,   default=5)
    p.add_argument("--cem_topk",          type=int,   default=10)
    p.add_argument("--cem_sigma",         type=float, default=0.02)
    p.add_argument("--gpu_id",            type=int,   default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading DexWM …")
    model, cfg, img_width, backbone_name = load_dexwm(
        args.dexwm_config, args.dexwm_checkpoint, device)
    num_context = cfg["data"]["num_context"]
    img_size    = cfg["data"]["img_size"]

    def to_tensor(img_bgr):
        return img_to_tensor(img_bgr, img_size, img_width, backbone_name)

    # ── goal embedding ─────────────────────────────────────────────────────────
    goal_bgr = cv2.imread(args.goal_image_path)
    assert goal_bgr is not None, f"Cannot read {args.goal_image_path}"
    goal_t   = to_tensor(goal_bgr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        goal_emb = model.encode_image(goal_t).squeeze(1)   # (1, P, F)
    print(f"Goal emb: {goal_emb.shape}")

    # ── load retargeted data ───────────────────────────────────────────────────
    print("Loading inspire retargeted pkls …")
    rh_data = load_retargeted_pkl(args.data_idx, "rh", args.data_root)
    lh_data = load_retargeted_pkl(args.data_idx, "lh", args.data_root)
    num_frames = rh_data["opt_dof_pos"].shape[0]

    # pre-compute pose44 for every GT frame
    all_poses = np.stack([
        get_pose44(rh_data["opt_joints_pos"][i], lh_data["opt_joints_pos"][i])
        for i in range(num_frames)
    ])   # (T, 44, 3)

    # ── load pre-rendered frames ───────────────────────────────────────────────
    rgb_files = sorted([
        os.path.join(args.rgb_dir, f)
        for f in os.listdir(args.rgb_dir) if f.endswith(".png")
    ])
    num_frames = min(num_frames, len(rgb_files))
    print(f"  {num_frames} frames  ({len(rgb_files)} RGB available)")

    def load_frame_tensor(fi):
        bgr = cv2.imread(rgb_files[fi])
        if bgr is None:
            return torch.zeros(3, img_size, img_width)
        return to_tensor(bgr)

    # ── init context buffer ────────────────────────────────────────────────────
    f0_t         = load_frame_tensor(0)
    context_imgs = f0_t.unsqueeze(0).repeat(num_context + 1, 1, 1, 1)   # (T+1,3,H,W)

    # ── CEM MPC loop ───────────────────────────────────────────────────────────
    print("Running DexWM CEM-MPC …")
    cur_frame  = 0
    rows       = []
    step_times = []

    for step_idx in range(num_frames):
        t0 = time.perf_counter()

        curr_t = load_frame_tensor(cur_frame)
        context_imgs = torch.cat([context_imgs[1:], curr_t.unsqueeze(0)], dim=0)

        best_pose44, cem_loss = cem_plan(
            model, context_imgs, all_poses[cur_frame], goal_emb, num_context, device,
            num_samples=args.cem_samples, opt_steps=args.cem_steps,
            topk=args.cem_topk, sigma=args.cem_sigma,
        )
        lo = cur_frame + 1
        hi = min(cur_frame + args.search_window + 1, num_frames)
        if lo >= hi:
            lo = cur_frame
        best_frame = find_nearest_gt_frame(best_pose44, all_poses, lo, hi)

        step_ms = (time.perf_counter() - t0) * 1000
        step_times.append(step_ms)
        rows.append({"step": step_idx, "cur_frame": cur_frame,
                     "best_frame": best_frame, "cem_loss": cem_loss, "step_ms": step_ms})

        print(f"[{step_idx:4d}/{num_frames}]  cur={cur_frame}  "
              f"best={best_frame}  cem_loss={cem_loss:.5f}  step={step_ms:.1f}ms")

        cur_frame = best_frame
        if cur_frame >= num_frames - 1:
            print("Reached last frame.")
            break

    # ── save results ───────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step","cur_frame","best_frame","cem_loss","step_ms"])
        writer.writeheader()
        writer.writerows(rows)

    if step_times:
        print(f"\n── Results saved to {csv_path}")
        print(f"── Timing: mean={sum(step_times)/len(step_times):.1f}ms  "
              f"min={min(step_times):.1f}ms  max={max(step_times):.1f}ms")
        final_loss = rows[-1]["cem_loss"]
        print(f"── Final step loss: {final_loss:.5f}  (goal = 0)")


if __name__ == "__main__":
    main()
