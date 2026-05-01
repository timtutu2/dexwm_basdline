"""
OakInk2 ManipTrans dataset for fine-tuning DexWM.

Loads rendered RGB frames from a ManipTrans replay (static chest camera) together
with retargeted hand joint positions from the mano2inspire pkl files.  The images
come from dexhandmanip_bih_gt_v2.py's chest camera:

    cam_pos    = (-0.25, -0.12, 0.95)   world frame (Isaac Gym, Z-up)
    cam_target = (-0.15, -0.12,  0.70)
    hfov       = 60°
    width=294, height=224

Joints are taken from opt_joints_pos (18 bodies per hand, world frame) stored in the
retargeted pkl files.  They are transformed to camera frame and padded from 18→21
keypoints per hand so the action vector matches the EgoDex action_dim=132 format:

    action = [lh_joints(21,3), rh_joints(21,3), cam_pos(1,3), cam_rot(1,3)] → (44,3)
    input to model: np.concatenate([curr_poses, next_poses]) → (88, 3)
    prepare_actions computes (next - curr).flatten() → (132,)

Keypoint heatmaps are set to zero (kp_weight should be 0 in the training config).
"""

import os
import glob
import pickle
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
import torchvision.transforms.v2 as transforms
from scipy.spatial.transform import Rotation as R


# ──────────────────────────────────────────────────────────────────────────────
# ManipTrans chest-camera constants (from dexhandmanip_bih_gt_v2.py)
# ──────────────────────────────────────────────────────────────────────────────
_CAM_POS    = np.array([-0.25, -0.12, 0.95], dtype=np.float64)
_CAM_TARGET = np.array([-0.15, -0.12, 0.70], dtype=np.float64)
_WORLD_UP   = np.array([ 0.00,  0.00,  1.00], dtype=np.float64)

# Number of inspire hand bodies in the pkl
_N_BODIES_PER_HAND = 18
# DexWM EgoDex-format keypoints per hand (21 left + 21 right + cam_pos + cam_rot = 44)
_N_KP_PER_HAND = 21
_N_KP_TOTAL    = 44   # must give action_dim = 44*3 = 132 after next-curr diff


def _build_cam_extrinsic() -> np.ndarray:
    """Compute 4×4 world-to-camera matrix for the static ManipTrans chest camera."""
    cam_pos    = _CAM_POS
    cam_target = _CAM_TARGET
    world_up   = _WORLD_UP

    # Camera Z axis (points INTO the scene, OpenCV convention)
    z_cam = cam_target - cam_pos
    z_cam /= np.linalg.norm(z_cam)

    # Camera X axis (right), camera Y axis (down in OpenCV, but here we keep Z-up
    # compatible by using world_up as reference)
    x_cam = np.cross(world_up, z_cam)
    if np.linalg.norm(x_cam) < 1e-6:
        x_cam = np.array([1.0, 0.0, 0.0])
    x_cam /= np.linalg.norm(x_cam)

    y_cam = np.cross(z_cam, x_cam)
    y_cam /= np.linalg.norm(y_cam)

    # Rotation rows = camera axes expressed in world frame
    rot = np.stack([x_cam, y_cam, z_cam], axis=0)   # (3, 3)
    trans = -rot @ cam_pos                           # (3,)

    extr = np.eye(4, dtype=np.float64)
    extr[:3, :3] = rot
    extr[:3,  3] = trans
    return extr


# Pre-compute once at import time
_CAM_EXTR = _build_cam_extrinsic().astype(np.float32)

# Constant camera pose entries for the action vector (same every frame → delta = 0)
_CAM_ROT_MAT_WORLD = _CAM_EXTR[:3, :3].T            # camera orientation in world frame
_CAM_ROT_EULER     = R.from_matrix(_CAM_ROT_MAT_WORLD).as_euler('xyz').astype(np.float32)
_CAM_POS_F32       = _CAM_POS.astype(np.float32)


class OakInk2ManipTransDataset(Dataset):
    """
    Single-sequence fine-tuning dataset built from:
      • RGB images  – rendered by the ManipTrans chest camera
      • Joint poses – opt_joints_pos in mano2inspire_{rh,lh} pkl files

    Returns the same tuple as EgoDexDataset:
        curr_frames  (num_context+1, 3, 224, img_width)  float32
        actions      (num_context,  88, 3)               float32  (curr+next stacked)
        rel_t        (num_context,)                      int array
        heatmaps     (num_context,  12, 224, img_width)  zeros
        valid_kp     (num_context,  12)                  zeros
        metadata     dict
    """

    def __init__(
        self,
        rh_pkl_path: str,
        lh_pkl_path: str,
        rgb_dir: str,
        max_context_len: int = 120,
        num_context: int = 8,
        patch_size: int = 14,
        img_size: int = 224,
        aug: bool = False,
        backbone_name: str = 'dinov2',
        train: bool = True,
        var_time: bool = False,
    ):
        super().__init__()

        # ── Load pkl joint data ──────────────────────────────────────────────
        with open(rh_pkl_path, 'rb') as f:
            rh_data = pickle.load(f)
        with open(lh_pkl_path, 'rb') as f:
            lh_data = pickle.load(f)

        # opt_joints_pos: (T, 18, 3)  world frame (gym coordinates)
        self.rh_joints = rh_data['opt_joints_pos'].astype(np.float32)
        self.lh_joints = lh_data['opt_joints_pos'].astype(np.float32)

        # ── Load image paths ─────────────────────────────────────────────────
        all_images = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        n_pkl = min(len(self.rh_joints), len(self.lh_joints))
        total_frames = min(len(all_images), n_pkl)
        self.image_files = all_images[:total_frames]

        # ── Train / val split (90 % / 10 %) ─────────────────────────────────
        split_idx = int(0.9 * total_frames)
        if train:
            self.avail_frames = list(range(split_idx))
        else:
            self.avail_frames = list(range(split_idx, total_frames))

        self.max_context_len = max_context_len
        self.num_context     = num_context
        self.img_size        = img_size
        self.patch_size      = patch_size
        self.var_time        = var_time
        self.train           = train
        self.backbone_name   = backbone_name

        # Target image width after padding / cropping
        self.img_width = 392 if patch_size == 14 else 384

        # ── Augmentation pipeline ────────────────────────────────────────────
        self.aug = self._build_aug(aug, backbone_name)

        # ── Build sampling index ─────────────────────────────────────────────
        # Produce one entry per non-overlapping half-window inside avail_frames
        n_avail = len(self.avail_frames)
        step    = max(1, max_context_len // num_context)
        # Positions (as indices into avail_frames) used as the "goal" frame
        self.idx_to_data = [
            pos for pos in range(max_context_len, n_avail, step)
        ]
        if not self.idx_to_data:
            # Very short sequence – at least one sample
            self.idx_to_data = [max(0, n_avail - 1)]

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_aug(aug, backbone_name):
        if not aug:
            return None
        norm = (
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            if 'siglip' in backbone_name
            else transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
        return transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            norm,
        ])

    def _joints_to_cam(self, joints_world: np.ndarray) -> np.ndarray:
        """Transform (N, 3) world-frame joints to camera frame via _CAM_EXTR."""
        ones   = np.ones((len(joints_world), 1), dtype=np.float32)
        j_hom  = np.concatenate([joints_world, ones], axis=1)   # (N, 4)
        j_cam  = (_CAM_EXTR @ j_hom.T).T                        # (N, 4)
        return j_cam[:, :3]                                      # (N, 3)

    def _get_pose(self, frame_idx: int) -> np.ndarray:
        """Return (44, 3) pose array for one frame in camera-frame coords."""
        rh_j = self.rh_joints[frame_idx]                        # (18, 3)
        lh_j = self.lh_joints[frame_idx]                        # (18, 3)

        rh_cam = self._joints_to_cam(rh_j)                      # (18, 3)
        lh_cam = self._joints_to_cam(lh_j)                      # (18, 3)

        # Pad 18 → 21 by repeating the last 3 body positions
        rh_padded = np.concatenate([rh_cam, rh_cam[-3:]], axis=0)   # (21, 3)
        lh_padded = np.concatenate([lh_cam, lh_cam[-3:]], axis=0)   # (21, 3)

        # Constant camera entries (delta will be 0 since camera is static)
        all_poses = np.concatenate([
            lh_padded,                    # (21, 3)
            rh_padded,                    # (21, 3)
            _CAM_POS_F32[None],           # (1,  3)
            _CAM_ROT_EULER[None],         # (1,  3)
        ], axis=0)                        # (44, 3)
        return all_poses

    def image_transform(self, img: np.ndarray) -> torch.Tensor:
        """Resize to img_size height and pad/crop width to img_width."""
        h, w = img.shape[:2]
        if h != self.img_size:
            ar  = w / h
            img = cv2.resize(
                img,
                (int(self.img_size * ar), self.img_size),
                interpolation=cv2.INTER_LINEAR,
            )

        cur_w = img.shape[1]
        if cur_w < self.img_width:
            pad_total = self.img_width - cur_w
            pad_l = pad_total // 2
            pad_r = pad_total - pad_l
            img = np.pad(img, ((0, 0), (pad_l, pad_r), (0, 0)), mode='constant')
        elif cur_w > self.img_width:
            crop = cur_w - self.img_width
            cl   = crop // 2
            img  = img[:, cl: cl + self.img_width]

        return torch.from_numpy(img.copy()).permute(2, 0, 1).float()   # (3, H, W)

    def _load_image(self, frame_idx: int) -> torch.Tensor:
        img = cv2.imread(self.image_files[frame_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.image_transform(img)

    def process_data(self, indices):
        """
        indices: list of len (num_context + 1), actual frame indices.
        Returns curr_frames, actions, heatmaps, valid_kp.
        """
        T = len(indices) - 1
        curr_frames = []
        actions     = []
        heatmaps    = torch.zeros(T, 12, self.img_size, self.img_width)
        valid_kp    = torch.zeros(T, 12)

        for ii in range(T):
            curr_frames.append(self._load_image(indices[ii]))
            curr_poses  = self._get_pose(indices[ii])       # (44, 3)
            next_poses  = self._get_pose(indices[ii + 1])   # (44, 3)
            actions.append(np.concatenate([curr_poses, next_poses], axis=0))  # (88, 3)

        # Goal frame (last)
        curr_frames.append(self._load_image(indices[-1]))

        # Stack and normalise
        curr_frames = torch.stack(curr_frames) / 255.           # (T+1, 3, H, W)
        curr_frames = tv_tensors.Video(curr_frames)
        if self.aug is not None:
            curr_frames = self.aug(curr_frames)
        else:
            if 'siglip' in self.backbone_name:
                curr_frames = transforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                )(curr_frames)
            else:
                curr_frames = transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )(curr_frames)

        actions = torch.from_numpy(np.array(actions, dtype=np.float32))  # (T, 88, 3)
        return curr_frames, actions, heatmaps, valid_kp

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.idx_to_data)

    def __getitem__(self, idx):
        goal_pos = self.idx_to_data[idx]         # index into self.avail_frames
        ctx_len  = self.max_context_len
        n_ctx    = self.num_context

        frame_id_min = max(0, goal_pos - ctx_len)
        frame_id     = goal_pos

        if self.var_time:
            pool    = list(range(frame_id_min, frame_id))
            sampled = sorted(random.sample(pool, min(n_ctx, len(pool))))
        else:
            skip    = max(1, ctx_len // n_ctx)
            sampled = list(range(frame_id_min, frame_id, skip))[:n_ctx]

        if not sampled:
            sampled = [frame_id_min]

        if len(sampled) < n_ctx:
            pad     = n_ctx - len(sampled)
            sampled = [sampled[0]] * pad + sampled

        # Convert positions (indices into avail_frames) to actual frame indices
        sampled.append(frame_id)
        actual = [self.avail_frames[i] for i in sampled]

        rel_t = np.array(
            [actual[i + 1] - actual[i] for i in range(len(actual) - 1)],
            dtype=np.float32,
        )

        curr_frames, actions, heatmaps, valid_kp = self.process_data(actual)
        metadata = {'frame_indices': torch.tensor(actual)}

        return curr_frames, actions, rel_t, heatmaps, valid_kp, metadata
