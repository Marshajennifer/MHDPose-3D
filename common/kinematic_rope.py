# kinematic_rope.py
import torch
import torch.nn.functional as F

def rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
#x1 = first half of the last dimension, x2 = second half.

@torch.no_grad()
def build_phi_bias_from_parents(
    parents_np,
    joints_left=None,
    joints_right=None,
    depth_weight=0.5,
    side_weight=0.15,
):
    """
    parents_np: numpy array (N,), parent index per joint, -1 for root.
    joints_left/right: optional joint indices used to add side-aware asymmetry.
    Returns torch.FloatTensor (N,) with a stable kinematic phase prior.
    """
    import numpy as np
    parents = parents_np.copy() # parents_np = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    N = len(parents) # 17
    depth = np.zeros(N, dtype=np.float32) # depth[i] will store how far joint i is from the root.
    # Allocate an array depth to store, for each joint i, the number of edges from the root to i (tree depth).

# parents: [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
# index:     0  1  2  3  4  5  6  7  8  9 10 11  12  13  14  15  16
# depth.max() = 5

    for i in range(N):
        d, j = 0, i
        while parents[j] != -1:
            j = parents[j]; d += 1
        depth[i] = d
    if depth.max() > 0:
        depth = depth / depth.max()  # [0,1]
    side = np.zeros(N, dtype=np.float32)
    if joints_left is not None:
        side[np.asarray(joints_left, dtype=np.int64)] = -1.0
    if joints_right is not None:
        side[np.asarray(joints_right, dtype=np.int64)] = 1.0

    # Side-aware Kin-RoPE: preserve root-to-limb depth while separating
    # mirrored anatomical branches at the same hierarchy level.
    phi_np = (depth_weight * depth) + (side_weight * side * depth)
    phi = torch.from_numpy(phi_np.astype(np.float32))
    return phi
#depth = [0,1,2,3, 1,2,3, 1,2,3, 4,3, 4,5, 3,4,5]
#depth-only phi = [0,.1,.2,.3, .1,.2,.3, .1,.2,.3, .4,.3, .4,.5, .3,.4,.5]
#later passed as phi_bias to apply_rope_kinematics function from mambahpe.py

def apply_rope_kinematic(q, k, pos_ids, phi_bias, base=10000.0, eps=0.01):
    """
    q,k: (B,H,N,Dh)
    pos_ids: (N,) int tensor 0..N-1 (token order == joint order)
    phi_bias: (N,) float tensor, per-joint phase (radians)
    eps: small scalar; 0 → vanilla RoPE
    """
    B,H,N,Dh = q.shape
    rot_dim = Dh
    half_dim = rot_dim // 2
    device, dtype = q.device, q.dtype

    freqs = torch.arange(half_dim, device=device, dtype=dtype)
    theta = base ** (-2.0 * freqs / max(1, rot_dim))  # (Dh/2,)
    pos = pos_ids.to(device=device, dtype=dtype)[:, None]  # (N,1)
    angles_half = pos * theta[None, :]                  # (N,Dh/2)

    if (phi_bias is not None) and (eps != 0.0):
        angles_half = angles_half + (phi_bias.to(device, dtype)[:, None] * eps)

    # rotate_half() pairs the first and second halves of the head dimension,
    # so both dimensions in each pair must receive the same angle.
    angles = torch.cat((angles_half, angles_half), dim=-1)
    cos = torch.cos(angles); sin = torch.sin(angles)
    if Dh > rot_dim:  # pad if odd Dh
        cos = F.pad(cos, (0, Dh-rot_dim)); sin = F.pad(sin, (0, Dh-rot_dim))

    cos = cos.unsqueeze(0).unsqueeze(0)            # (1,1,N,Dh)
    sin = sin.unsqueeze(0).unsqueeze(0)            # (1,1,N,Dh)

    q_ro = (q * cos) + (rotate_half(q) * sin)
    k_ro = (k * cos) + (rotate_half(k) * sin)
    return q_ro, k_ro
