#!/usr/bin/env python3
"""
DexWM discrete-search MPC visualised in IsaacGym (OakInk2 bimanual).

At each step DexWM scores candidate next GT frames and the best one is
applied in IsaacGym so you can watch the hands move in the viewer.

Portable: all machine-specific paths are passed as CLI args or env vars.

Install IsaacGym in the dexwm env first:
    pip install -e <path-to-isaacgym>/python   # Preview 4

Usage (interactive viewer on david):
    ISAACGYM_PATH=/path/to/isaacgym/python \\
    PYTHONPATH=. python oak_ink_eval/dexwm_mpc_replay.py \\
        --data_root      /path/to/maniptrans_lib \\
        --asset_root     /path/to/assets \\
        --data_idx        083f7@0 \\
        --dexwm_checkpoint /mnt/data/tim_data/dexwm/runs/oakink2_ft_wandb/checkpoints/oakink2_maniptrans_ft_249.pth.tar \\
        --dexwm_config    /mnt/data/tim_data/dexwm/configs/oakink2_finetune.yaml \\
        --goal_image_path /mnt/data/tim_data/dexwm/data/oakink2_processed/rgb/001196.png \\
        --search_window   10

Usage (headless, save frames):
    ... same flags + --headless --output_dir output/dexwm_mpc/083f7
"""

import sys
import os
import argparse
import json
import pickle
import yaml
from pathlib import Path
import time

# ── IsaacGym path: env var or well-known default ──────────────────────────────
_igym_path = os.environ.get("ISAACGYM_PATH", "/home/yulun/isaacgym/python")
sys.path.insert(0, _igym_path)
from isaacgym import gymapi, gymtorch  # noqa: E402  (must come before torch)

import numpy as np
import torch
import cv2
import torchvision.transforms.functional as TF
from scipy.spatial.transform import Rotation as ScipyR

# ── DexWM: auto-detect root from this script's location ──────────────────────
DEXWM_ROOT = str(Path(__file__).resolve().parent.parent)   # oak_ink_eval/../
sys.path.insert(0, DEXWM_ROOT)
from models.model import DexWM                    # noqa: E402


def get_patch_size_from_backbone(backbone_name):
    if 'dinov2' in backbone_name or 'siglip' in backbone_name or 'webssl' in backbone_name:
        patch_size = 14
        num_patches = 448
    elif 'dinov3' in backbone_name or 'vjepa' in backbone_name:
        patch_size = 16
        num_patches = 336
    else:
        raise ValueError(f'Backbone {backbone_name} not supported')
    return patch_size, num_patches


# ─────────────────────────────── quaternion math (no ManipTrans needed) ───────

def aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    return ScipyR.from_rotvec(aa).as_matrix().astype(np.float32)


def aa_to_isaac_quat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle → [x, y, z, w] for IsaacGym."""
    q_xyzw = ScipyR.from_rotvec(aa).as_quat()   # scipy gives [x,y,z,w]
    return q_xyzw.astype(np.float32)


def rotmat_to_isaac_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → [x, y, z, w] for IsaacGym."""
    q_xyzw = ScipyR.from_matrix(R).as_quat()    # scipy gives [x,y,z,w]
    return q_xyzw.astype(np.float32)


# ─────────────────────────────── camera constants ─────────────────────────────
_CAM_POS    = np.array([-0.25, -0.12, 0.95], dtype=np.float64)
_CAM_TARGET = np.array([-0.15, -0.12, 0.70], dtype=np.float64)
_WORLD_UP   = np.array([ 0.00,  0.00,  1.00], dtype=np.float64)
HEAD_CAM_WIDTH  = 294
HEAD_CAM_HEIGHT = 224

TABLE_POS_Z       = 0.4
TABLE_HALF_HEIGHT = 0.015
TABLE_SURFACE_Z   = TABLE_POS_Z + TABLE_HALF_HEIGHT
TABLE_HALF_WIDTH  = 0.4
TABLE_WIDTH_OFFSET = 0.2
ROBOT_HEIGHT      = 0.00214874


def _build_cam_extrinsic() -> np.ndarray:
    z = _CAM_TARGET - _CAM_POS;  z /= np.linalg.norm(z)
    x = np.cross(_WORLD_UP, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1., 0., 0.])
    x /= np.linalg.norm(x)
    y = np.cross(z, x);  y /= np.linalg.norm(y)
    rot = np.stack([x, y, z], axis=0)
    extr = np.eye(4, dtype=np.float64)
    extr[:3, :3] = rot
    extr[:3,  3] = -rot @ _CAM_POS
    return extr.astype(np.float32)


_CAM_EXTR      = _build_cam_extrinsic()
_CAM_POS_F32   = _CAM_POS.astype(np.float32)
_CAM_ROT_EULER = ScipyR.from_matrix(_CAM_EXTR[:3, :3].T).as_euler('xyz').astype(np.float32)


def build_mujoco2gym() -> np.ndarray:
    """Mujoco → IsaacGym coordinate transform (from ManipTrans bih.py)."""
    rot = (
        aa_to_rotmat(np.array([0.0, 0.0, -np.pi / 2]))
        @ aa_to_rotmat(np.array([np.pi / 2, 0.0, 0.0]))
    )
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3,  3] = [0.0, 0.0, TABLE_SURFACE_Z]
    return T


# ─────────────────────────────── keypoint helpers ─────────────────────────────

def joints_to_cam(joints_world: np.ndarray) -> np.ndarray:
    ones  = np.ones((len(joints_world), 1), dtype=np.float32)
    j_hom = np.concatenate([joints_world.astype(np.float32), ones], axis=1)
    return (_CAM_EXTR @ j_hom.T).T[:, :3]


def get_pose44(rh_joints: np.ndarray, lh_joints: np.ndarray) -> np.ndarray:
    """(44,3) camera-frame pose matching OakInk2ManipTransDataset._get_pose."""
    rh_cam = joints_to_cam(rh_joints)
    lh_cam = joints_to_cam(lh_joints)
    rh_pad = np.concatenate([rh_cam, rh_cam[-3:]], axis=0)   # 18→21
    lh_pad = np.concatenate([lh_cam, lh_cam[-3:]], axis=0)
    return np.concatenate(
        [lh_pad, rh_pad, _CAM_POS_F32[None], _CAM_ROT_EULER[None]], axis=0
    )  # (44, 3)


# ─────────────────────────────── data loading ─────────────────────────────────

def _resolve_anno_stem(seq_hash: str, anno_dir: str):
    matches = [f for f in os.listdir(anno_dir) if seq_hash in f]
    assert len(matches) == 1, f"Expected 1 annotation for '{seq_hash}', got {matches}"
    enc_stem = os.path.splitext(matches[0])[0]
    return enc_stem, enc_stem.replace("%2B", "+")


def load_retargeted_pkl(data_idx: str, side: str, data_root: str) -> dict:
    seq_hash = data_idx.split("@")[0]
    stage    = data_idx.split("@")[1]
    anno_dir = os.path.join(data_root, "OakInk-v2", "anno_preview")
    enc_stem, dec_stem = _resolve_anno_stem(seq_hash, anno_dir)
    pkl_dir = os.path.join(data_root, "retargeting", "OakInk-v2", f"mano2inspire_{side}")
    for stem in (dec_stem, enc_stem):
        p = os.path.join(pkl_dir, f"{stem}@{stage}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Inspire pkl not found in {pkl_dir}")


def load_annotation(data_idx: str, data_root: str):
    seq_hash = data_idx.split("@")[0]
    anno_dir = os.path.join(data_root, "OakInk-v2", "anno_preview")
    enc_stem, dec_stem = _resolve_anno_stem(seq_hash, anno_dir)
    with open(os.path.join(anno_dir, f"{enc_stem}.pkl"), "rb") as f:
        anno = pickle.load(f)
    return anno, dec_stem


def get_frame_list(anno: dict, dec_stem: str, data_root: str, stage: int = 0):
    prog_path = os.path.join(
        data_root, "OakInk-v2", "program", "program_info", f"{dec_stem}.json"
    )
    with open(prog_path) as f:
        raw = json.load(f)
    program_info = {eval(k): v for k, v in raw.items()}
    key = list(program_info.keys())[stage]
    lr, rr = key[0], key[1]
    begin, end = max(lr[0], rr[0]), min(lr[1], rr[1])
    frames = [f for f in anno["mocap_frame_id_list"][::2] if begin <= f <= end]
    return frames, program_info[key]


# ─────────────────────────────── DexWM model ──────────────────────────────────

def load_dexwm(config_path: str, checkpoint_path: str, device: torch.device):
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


def img_to_tensor(img_bgr: np.ndarray, img_size: int, img_width: int,
                  backbone_name: str) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if h != img_size:
        img = cv2.resize(img, (int(img_size * w / h), img_size), interpolation=cv2.INTER_LINEAR)
    cur_w = img.shape[1]
    if cur_w < img_width:
        pad = img_width - cur_w
        img = np.pad(img, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), mode="constant")
    elif cur_w > img_width:
        crop = cur_w - img_width
        img  = img[:, crop // 2: crop // 2 + img_width]
    t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
    if "siglip" in backbone_name:
        t = TF.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    else:
        t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return t


# ─────────────────────────────── MPC scoring ──────────────────────────────────

@torch.inference_mode()
def score_candidates(model, context_imgs, curr_pose44, cand_poses,
                     goal_emb, num_context, device):
    """Returns (K,) MSE loss — lower = closer to goal."""
    K      = len(cand_poses)
    deltas = (cand_poses - curr_pose44[None]).astype(np.float32)   # (K,44,3)
    actions = torch.zeros(K, num_context, 44, 3, dtype=torch.float32, device=device)
    actions[:, -1] = torch.from_numpy(deltas).to(device)
    ctx   = context_imgs.unsqueeze(0).expand(K, -1, -1, -1, -1).contiguous().to(device)
    rel_t = torch.zeros(K, num_context, dtype=torch.float32, device=device)
    x_pred, _, _, _, _ = model(ctx, actions, rel_t, action_diff=True)
    pred_last = x_pred[:, -1]   # (K, P, F)
    loss = torch.nn.functional.mse_loss(
        pred_last, goal_emb.expand(K, -1, -1), reduction="none"
    ).mean(dim=[1, 2])
    return loss.cpu().numpy()


def cem_plan(model, context_imgs, curr_pose44, goal_emb, num_context, device,
             num_samples=512, opt_steps=5, topk=10, sigma=0.02):
    """CEM over continuous pose44 delta space. Returns (best_pose44, best_loss)."""
    mu  = np.zeros((44, 3), dtype=np.float32)
    std = np.full((44, 3), sigma, dtype=np.float32)
    best_loss = np.inf
    for _ in range(opt_steps):
        deltas     = (mu[None] + std[None] * np.random.randn(num_samples, 44, 3)).astype(np.float32)
        cand_poses = curr_pose44[None] + deltas                   # (N, 44, 3)
        losses     = score_candidates(model, context_imgs, curr_pose44, cand_poses,
                                      goal_emb, num_context, device)
        elite_idx  = np.argsort(losses)[:topk]
        elite      = deltas[elite_idx]
        mu         = elite.mean(0)
        std        = elite.std(0) + 1e-6
        best_loss  = float(losses[elite_idx[0]])
    return (curr_pose44 + mu).astype(np.float32), best_loss


def find_nearest_gt_frame(target_pose44, all_poses, lo, hi):
    """GT frame index in [lo, hi) whose pose44 has minimum MSE to target."""
    lo = max(lo, 0); hi = min(hi, len(all_poses))
    if lo >= hi:
        return lo
    dists = np.mean((all_poses[lo:hi] - target_pose44[None]) ** 2, axis=(1, 2))
    return lo + int(np.argmin(dists))


# ─────────────────────────────── IsaacGym setup ───────────────────────────────

def init_gym(gpu_id: int):
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sp.dt, sp.substeps = 1.0 / 60.0, 2
    sp.use_gpu_pipeline = False
    sp.physx.use_gpu   = False
    sp.physx.num_threads = 4
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 8
    sp.physx.num_velocity_iterations = 1
    sp.physx.contact_offset = 0.002
    sp.physx.rest_offset = 0.0
    sim = gym.create_sim(gpu_id, gpu_id, gymapi.SIM_PHYSX, sp)
    assert sim is not None
    pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, pp)
    return gym, sim


def load_hand_asset(gym, sim, side: str, asset_root: str):
    hand_dir = os.path.join(asset_root, "inspire_hand")
    fname    = f"inspire_hand_{'right' if side == 'rh' else 'left'}.urdf"
    opts = gymapi.AssetOptions()
    opts.fix_base_link = False
    opts.disable_gravity = True
    opts.angular_damping = opts.linear_damping = 20.0
    opts.max_linear_velocity = 50.0; opts.max_angular_velocity = 100.0
    opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
    opts.collapse_fixed_joints = False
    opts.use_mesh_materials = True
    asset = gym.load_asset(sim, hand_dir, fname, opts)
    assert asset is not None, f"Cannot load {hand_dir}/{fname}"
    return asset


def load_obj_asset(gym, sim, obj_id: str, data_root: str):
    obj_dir = os.path.join(data_root, "OakInk-v2", "coacd_object_preview", "align_ds", obj_id)
    fname   = [f for f in os.listdir(obj_dir) if f.endswith(".urdf")][0]
    opts = gymapi.AssetOptions()
    opts.override_com = opts.override_inertia = True
    opts.convex_decomposition_from_submeshes = False
    opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    opts.use_mesh_materials = True
    opts.vhacd_enabled = True
    opts.vhacd_params = gymapi.VhacdParams()
    opts.vhacd_params.resolution = 1000
    opts.density = 200.0
    # URDFs reference meshes as "data/OakInk-v2/..." relative to the ManipTrans
    # project root (parent of data/). Pass that root so IsaacGym resolves correctly.
    maniptrans_root = os.path.dirname(data_root)
    rel_urdf = os.path.join("data", "OakInk-v2", "coacd_object_preview", "align_ds", obj_id, fname)
    asset = gym.load_asset(sim, maniptrans_root, rel_urdf, opts)
    assert asset is not None
    return asset


def set_pos_drive(gym, env, actor, n_dof):
    props = gym.get_actor_dof_properties(env, actor)
    for i in range(n_dof):
        props["driveMode"][i] = gymapi.DOF_MODE_POS
        props["stiffness"][i] = 500.0
        props["damping"][i]   = 30.0
    gym.set_actor_dof_properties(env, actor, props)


# ──────────────────────────────────── main ────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="/home/yulun/projects/ManipTrans/data",
                   help="Root containing OakInk-v2/ and retargeting/")
    p.add_argument("--asset_root", default="/mnt/drive/yulun/projects/ManipTrans/maniptrans_envs/assets",
                   help="Root containing inspire_hand/")
    p.add_argument("--data_idx",         default="083f7@0",
                   help="OakInk-V2 sequence index, e.g. 083f7@0")
    p.add_argument("--dexwm_checkpoint", required=True)
    p.add_argument("--dexwm_config",     required=True)
    p.add_argument("--goal_image_path",  required=True,
                   help="Path to goal PNG (last frame of the pre-rendered RGB dir)")
    p.add_argument("--output_dir",       default="output/dexwm_mpc")
    p.add_argument("--headless",         action="store_true")
    p.add_argument("--search_window",    type=int, default=50,
                   help="GT frame look-ahead range for nearest-frame snapping after CEM")
    p.add_argument("--cem_samples",     type=int,   default=512)
    p.add_argument("--cem_steps",       type=int,   default=5)
    p.add_argument("--cem_topk",        type=int,   default=10)
    p.add_argument("--cem_sigma",       type=float, default=0.02,
                   help="Initial std for CEM sampling in pose44 space (metres/rad)")
    p.add_argument("--fps",              type=float, default=30.0)
    p.add_argument("--gpu_id",           type=int, default=0,
                   help="CUDA/IsaacGym GPU id to use")
    return p.parse_args()


def main():
    args    = parse_args()
    device  = torch.device(f"cuda:{args.gpu_id}")
    stage   = int(args.data_idx.split("@")[1])

    ASSET_ROOT = args.asset_root
    DATA_ROOT  = args.data_root

    os.makedirs(args.output_dir, exist_ok=True)

    # ── DexWM ─────────────────────────────────────────────────────────────────
    print("Loading DexWM …")
    model, cfg, img_width, backbone_name = load_dexwm(
        args.dexwm_config, args.dexwm_checkpoint, device
    )
    num_context  = cfg["data"]["num_context"]
    img_size     = cfg["data"]["img_size"]

    def to_tensor(img_bgr):
        return img_to_tensor(img_bgr, img_size, img_width, backbone_name)

    # ── Goal embedding ─────────────────────────────────────────────────────────
    goal_bgr = cv2.imread(args.goal_image_path)
    assert goal_bgr is not None
    goal_t   = to_tensor(goal_bgr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        goal_emb = model.encode_image(goal_t).squeeze(1)   # (1, P, F)
    print(f"Goal emb: {goal_emb.shape}")

    # ── Load retargeted pkls ───────────────────────────────────────────────────
    print("Loading inspire retargeted pkls …")
    rh_data    = load_retargeted_pkl(args.data_idx, "rh", DATA_ROOT)
    lh_data    = load_retargeted_pkl(args.data_idx, "lh", DATA_ROOT)
    num_frames = rh_data["opt_dof_pos"].shape[0]
    print(f"  {num_frames} frames")

    # Pre-compute camera-frame poses for every GT frame
    all_poses = np.stack([
        get_pose44(rh_data["opt_joints_pos"][i], lh_data["opt_joints_pos"][i])
        for i in range(num_frames)
    ])   # (T, 44, 3)

    # ── Annotation + object trajectories ─────────────────────────────────────
    anno, dec_stem = load_annotation(args.data_idx, DATA_ROOT)
    frame_id_list, prog_info = get_frame_list(anno, dec_stem, DATA_ROOT, stage)
    assert len(frame_id_list) == num_frames, "Frame / pkl count mismatch"

    M = build_mujoco2gym()
    def build_obj_traj(obj_id):
        return np.stack([M @ anno["obj_transf"][obj_id][fid].astype(np.float32)
                         for fid in frame_id_list])

    obj_rh_id   = prog_info["obj_list_rh"][0]
    obj_lh_id   = prog_info["obj_list_lh"][0]
    obj_rh_traj = build_obj_traj(obj_rh_id)
    obj_lh_traj = build_obj_traj(obj_lh_id)

    # ── IsaacGym ──────────────────────────────────────────────────────────────
    print("Initialising IsaacGym …")
    gym, sim = init_gym(args.gpu_id)

    rh_asset     = load_hand_asset(gym, sim, "rh", ASSET_ROOT)
    lh_asset     = load_hand_asset(gym, sim, "lh", ASSET_ROOT)
    obj_rh_asset = load_obj_asset(gym, sim, obj_rh_id, DATA_ROOT)
    obj_lh_asset = load_obj_asset(gym, sim, obj_lh_id, DATA_ROOT)
    n_rh = gym.get_asset_dof_count(rh_asset)
    n_lh = gym.get_asset_dof_count(lh_asset)

    table_opts = gymapi.AssetOptions(); table_opts.fix_base_link = True
    table_asset = gym.create_box(sim, 0.8 + TABLE_WIDTH_OFFSET, 1.6, 0.03, table_opts)

    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

    hand_init = gymapi.Transform()
    hand_init.p = gymapi.Vec3(-TABLE_HALF_WIDTH, 0.0, TABLE_SURFACE_Z + ROBOT_HEIGHT)
    hand_init.r = gymapi.Quat.from_euler_zyx(0.0, -np.pi / 2, 0.0)

    rh_actor = gym.create_actor(env, rh_asset,  hand_init, "dexhand_r", 0, 0)
    lh_actor = gym.create_actor(env, lh_asset,  hand_init, "dexhand_l", 0, 0)
    set_pos_drive(gym, env, rh_actor, n_rh)
    set_pos_drive(gym, env, lh_actor, n_lh)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(-TABLE_WIDTH_OFFSET / 2, 0.0, TABLE_POS_Z)
    gym.create_actor(env, table_asset, table_pose, "table", 0, 0)

    def make_obj_pose(T4):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(float(T4[0,3]), float(T4[1,3]), float(T4[2,3]))
        q = rotmat_to_isaac_quat(T4[:3,:3])
        pose.r = gymapi.Quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        return pose

    obj_rh_actor = gym.create_actor(env, obj_rh_asset, make_obj_pose(obj_rh_traj[0]), "obj_rh", 0, 0)
    obj_lh_actor = gym.create_actor(env, obj_lh_asset, make_obj_pose(obj_lh_traj[0]), "obj_lh", 0, 0)

    # Chest camera matching the DexWM training setup
    cam_props = gymapi.CameraProperties()
    cam_props.width, cam_props.height = HEAD_CAM_WIDTH, HEAD_CAM_HEIGHT
    cam_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(
        cam_handle, env,
        gymapi.Vec3(*_CAM_POS.tolist()),
        gymapi.Vec3(*_CAM_TARGET.tolist()),
    )

    gym.prepare_sim(sim)
    root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof_state  = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))

    rh_idx     = gym.get_actor_index(env, rh_actor,     gymapi.DOMAIN_SIM)
    lh_idx     = gym.get_actor_index(env, lh_actor,     gymapi.DOMAIN_SIM)
    obj_rh_idx = gym.get_actor_index(env, obj_rh_actor, gymapi.DOMAIN_SIM)
    obj_lh_idx = gym.get_actor_index(env, obj_lh_actor, gymapi.DOMAIN_SIM)
    gym_device = "cpu"   # gymtorch tensors live on CPU when use_gpu_pipeline=False
    all_idxs   = torch.tensor([rh_idx, lh_idx, obj_rh_idx, obj_lh_idx],
                               dtype=torch.int32, device=gym_device)

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        assert viewer is not None
        gym.viewer_camera_look_at(viewer, None,
                                  gymapi.Vec3(0.8, -0.5, 1.2),
                                  gymapi.Vec3(0.0,  0.0, 0.5))

    def apply_frame(fi: int):
        for (idx, pos, rot) in [
            (rh_idx, rh_data["opt_wrist_pos"][fi], rh_data["opt_wrist_rot"][fi]),
            (lh_idx, lh_data["opt_wrist_pos"][fi], lh_data["opt_wrist_rot"][fi]),
        ]:
            root_state[idx, :3]  = torch.tensor(pos.astype(np.float32), device=gym_device)
            root_state[idx, 3:7] = torch.tensor(aa_to_isaac_quat(rot), device=gym_device)
            root_state[idx, 7:]  = 0.0
        for actor_idx, traj in [(obj_rh_idx, obj_rh_traj), (obj_lh_idx, obj_lh_traj)]:
            T = traj[fi]
            root_state[actor_idx, :3]  = torch.tensor(T[:3, 3], device=gym_device)
            root_state[actor_idx, 3:7] = torch.tensor(rotmat_to_isaac_quat(T[:3, :3]), device=gym_device)
            root_state[actor_idx, 7:]  = 0.0
        rd = rh_data["opt_dof_pos"][fi].astype(np.float32)
        ld = lh_data["opt_dof_pos"][fi].astype(np.float32)
        dof_state[:n_rh, 0] = torch.tensor(rd, device=gym_device); dof_state[:n_rh, 1] = 0.0
        dof_state[n_rh:n_rh+n_lh, 0] = torch.tensor(ld, device=gym_device); dof_state[n_rh:n_rh+n_lh, 1] = 0.0
        gym.set_actor_root_state_tensor_indexed(
            sim, gymtorch.unwrap_tensor(root_state),
            gymtorch.unwrap_tensor(all_idxs), len(all_idxs)
        )
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))

    def get_cam_bgr() -> np.ndarray:
        gym.render_all_camera_sensors(sim)
        raw  = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
        rgba = raw.reshape(HEAD_CAM_HEIGHT, HEAD_CAM_WIDTH, 4)
        return cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)

    # ── Init context buffer ────────────────────────────────────────────────────
    apply_frame(0)
    gym.simulate(sim); gym.fetch_results(sim, True); gym.step_graphics(sim)
    f0_t = to_tensor(get_cam_bgr())
    context_imgs = f0_t.unsqueeze(0).repeat(num_context + 1, 1, 1, 1)   # (T+1,3,H,W)

    rgb_out = os.path.join(args.output_dir, "rgb")
    os.makedirs(rgb_out, exist_ok=True)
    frame_dt = 1.0 / args.fps

    # ── MPC loop ──────────────────────────────────────────────────────────────
    print("Running DexWM MPC …")
    cur_frame = 0
    step_times: list[float] = []
    t_loop_start = time.perf_counter()

    for step_idx in range(num_frames):
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break
        t0 = time.perf_counter()

        curr_bgr = get_cam_bgr()
        curr_t   = to_tensor(curr_bgr)
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
        print(f"[{step_idx:4d}/{num_frames}]  cur={cur_frame}  "
              f"best={best_frame}  cem_loss={cem_loss:.5f}  step={step_ms:.1f}ms")

        apply_frame(best_frame)
        cur_frame = best_frame

        gym.simulate(sim); gym.fetch_results(sim, True); gym.step_graphics(sim)

        if viewer is not None:
            gym.poll_viewer_events(viewer)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
            remaining = frame_dt - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)

        cv2.imwrite(os.path.join(rgb_out, f"{step_idx:06d}.png"), curr_bgr)

        if cur_frame >= num_frames - 1:
            print("Reached last frame.")
            break

    total_s = time.perf_counter() - t_loop_start
    if step_times:
        print(f"\n── Timing ──────────────────────────────────────────")
        print(f"  Steps run   : {len(step_times)}")
        print(f"  Total time  : {total_s:.2f}s")
        print(f"  Per-step    : mean={sum(step_times)/len(step_times):.1f}ms  "
              f"min={min(step_times):.1f}ms  max={max(step_times):.1f}ms")

    if viewer is not None:
        print("Done. Close the viewer to exit.")
        while not gym.query_viewer_has_closed(viewer):
            gym.poll_viewer_events(viewer); gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)
        gym.destroy_viewer(viewer)

    gym.destroy_sim(sim)
    print(f"Frames saved to {rgb_out}")


if __name__ == "__main__":
    main()
