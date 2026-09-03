import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from common.arguments import parse_args


args = parse_args()
# ---------------- 1. Load GT ----------------
npz_path = "./data/data_test_3dhp_ori.npz"
data = np.load(npz_path, allow_pickle=True)["data"].item()

seq = args.seq
gt_3d_all = data[seq]["data_3d"]          # [F,17,3]
valid = data[seq]["valid"].astype(bool)
valid_indices = np.where(valid)[0]
gt_3d = gt_3d_all[valid]

# ---------------- 2. Load prediction --------
mat_path = (
    f"./checkpoint/inference_data_P_Agg_{args.p_agg_mode}"
    f"_H{args.num_proposals}_K{args.sampling_timesteps}.mat"
)
print("Loading prediction:", mat_path)
pred_mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

pred_seq = pred_mat[seq]
print("pred_seq shape:", pred_seq.shape)

if pred_seq.shape[0] == 3:
    pred_3d = pred_seq[..., -1]              # (3,17,F)
    pred_3d = np.transpose(pred_3d, (2, 1, 0))
elif pred_seq.shape[0] == 17:
    pred_3d = pred_seq[..., -1]              # (17,3,F)
    pred_3d = np.transpose(pred_3d, (2, 0, 1))
else:
    raise ValueError(f"Unexpected prediction shape: {pred_seq.shape}")

pred_3d = pred_3d[valid]

print("GT shape:", gt_3d.shape, "Pred shape:", pred_3d.shape)

frame_idx = args.frame
frame_idx = min(frame_idx, gt_3d.shape[0] - 1)
orig_frame_idx = valid_indices[frame_idx]
print(f"Visualising valid frame {frame_idx} (original {orig_frame_idx})")

gt = gt_3d[frame_idx].copy()
pr = pred_3d[frame_idx].copy()

root_idx = 14
gt -= gt[root_idx]
pr -= pr[root_idx]

# Convert 3DHP camera coordinates to Matplotlib visualization coordinates.
# Matplotlib uses Z as the vertical axis; this keeps the body upright when
# rotating the visualization view with ax.view_init(...).
gt = np.stack([gt[:, 0], gt[:, 2], -gt[:, 1]], axis=1)
pr = np.stack([pr[:, 0], pr[:, 2], -pr[:, 1]], axis=1)

# skeleton connections

bones = [
    # spine + head
    (14, 15), (15, 1), (1, 0),
    # right arm
    (1, 2), (2, 3), (3, 4),
    # left arm
    (1, 5), (5, 6), (6, 7),
    # right leg
    (14, 8), (8, 9), (9, 10),
    # left leg
    (14, 11), (11, 12), (12, 13),
]

# ----------------------------------------------------
# 6. Plot GT (blue) and Prediction (red) skeletons
# ----------------------------------------------------
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# If you want to show joint positions uncomment below lines
# ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=15, label="Groundtruth", depthshade=True)
# ax.scatter(pr[:, 0], pr[:, 1], pr[:, 2], s=15, marker="^", label="Predicted", depthshade=True)

# Draw skeleton
for i, j in bones:
    # GT in blue
    ax.plot(
        [gt[i, 0], gt[j, 0]],
        [gt[i, 1], gt[j, 1]],
        [gt[i, 2], gt[j, 2]],
        linewidth=2,
        color='b'
    )
    # Prediction in red
    ax.plot(
        [pr[i, 0], pr[j, 0]],
        [pr[i, 1], pr[j, 1]],
        [pr[i, 2], pr[j, 2]],
        linewidth=2,
        color='r'
    )

# Uncomment below 2 lines if you want to see joint index numbers
# for j in range(gt.shape[0]):
#     ax.text(gt[j, 0], gt[j, 1], gt[j, 2], str(j), fontsize=8)

ax.set_title(f"MPI-INF-3DHP {seq} – frame {frame_idx} (orig {orig_frame_idx})\n"
             "GT (blue) vs Prediction (red)")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# To remove numbering on axis
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')


stacked = np.vstack((gt, pr))
max_range = (stacked.max(axis=0) - stacked.min(axis=0)).max() / 2.0
mid = (stacked.max(axis=0) + stacked.min(axis=0)) / 2.0
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
# ax.set_box_aspect([1, 1, 1])


#Adjust orientation as your preference

# ax.view_init(elev=100, azim=90) #original
ax.view_init(elev=20, azim=-90) 
# ax.invert_xaxis()


# ax.legend()
# plt.tight_layout()
if args.viz_output:
    plt.savefig(args.viz_output, dpi=200, bbox_inches='tight')
    print("Saved visualization:", args.viz_output)
else:
    plt.show()
