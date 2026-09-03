import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


BONES = [
    (14, 15), (15, 1), (1, 0),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (14, 8), (8, 9), (9, 10),
    (14, 11), (11, 12), (12, 13),
]


def find_sequence_dir(dataset_root, seq):
    root = Path(dataset_root)
    candidates = [root / seq, root / seq.lower(), root / seq.upper()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find sequence folder. Tried:\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


def find_image(seq_dir, image_number):
    candidates = [
        seq_dir / "imageSequence" / f"img_{image_number:06d}.jpg",
        seq_dir / "imageSequence" / f"img_{image_number:06d}.png",
        seq_dir / "image_sequence" / f"img_{image_number:06d}.jpg",
        seq_dir / "image_sequence" / f"img_{image_number:06d}.png",
        seq_dir / f"img_{image_number:06d}.jpg",
        seq_dir / f"img_{image_number:06d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find image img_{image_number:06d}.jpg/png. Tried:\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


def load_annotation(seq_dir):
    annot_path = seq_dir / "annot_data.mat"
    if not annot_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annot_path}")
    try:
        return sio.loadmat(annot_path, squeeze_me=False, struct_as_record=False)
    except NotImplementedError:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "annot_data.mat is MATLAB v7.3. Install h5py in this environment: pip install h5py"
            ) from exc

        annot = {}
        with h5py.File(annot_path, "r") as mat_file:
            for key in mat_file.keys():
                annot[key] = np.array(mat_file[key])
        return annot


def get_gt_pose(annot, raw_frame_idx):
    if "annot3" in annot:
        gt_all = np.asarray(annot["annot3"])
    elif "univ_annot3" in annot:
        gt_all = np.asarray(annot["univ_annot3"])
    else:
        return None

    if gt_all.ndim == 4:
        if gt_all.shape[0] == 3:
            return gt_all[:, :, 0, raw_frame_idx].T
        if gt_all.shape[-1] == 3:
            return gt_all[raw_frame_idx, 0, :, :]
    if gt_all.ndim == 3:
        if gt_all.shape[0] == 3:
            return gt_all[:, :, raw_frame_idx].T
        if gt_all.shape[-1] == 3:
            return gt_all[raw_frame_idx, :, :]
    raise ValueError(f"Unexpected annotation shape: {gt_all.shape}")


def draw_3d_pose(ax, pose, title):
    pose = pose.copy()
    pose -= pose[14]

    for i, j in BONES:
        ax.plot(
            [pose[i, 0], pose[j, 0]],
            [pose[i, 1], pose[j, 1]],
            [pose[i, 2], pose[j, 2]],
            color="b",
            linewidth=2,
        )

    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    max_range = (pose.max(axis=0) - pose.min(axis=0)).max() / 2.0
    mid = (pose.max(axis=0) + pose.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.view_init(elev=100, azim=90)
    ax.invert_xaxis()


def resolve_frame(annot, frame, frame_mode):
    valid = np.asarray(annot["valid_frame"]).reshape(-1).astype(bool)
    valid_indices = np.where(valid)[0]

    if frame_mode == "raw_image":
        image_number = frame
        raw_frame_idx = frame - 1
    else:
        if frame < 0 or frame >= len(valid_indices):
            raise ValueError(f"Valid-frame index must be between 0 and {len(valid_indices) - 1}")
        raw_frame_idx = int(valid_indices[frame])
        image_number = raw_frame_idx + 1

    if raw_frame_idx < 0 or raw_frame_idx >= len(valid):
        raise ValueError(f"Raw image number must be between 1 and {len(valid)}")

    return raw_frame_idx, image_number, bool(valid[raw_frame_idx])


def visualize_frame(
    dataset_root,
    seq="TS5",
    frame=12,
    frame_mode="raw_image",
    show_pose=True,
    output="",
):
    seq = seq.upper()
    seq_dir = find_sequence_dir(dataset_root, seq)
    annot = load_annotation(seq_dir)
    raw_frame_idx, image_number, is_valid = resolve_frame(annot, frame, frame_mode)
    image_path = find_image(seq_dir, image_number)

    print(f"Sequence folder: {seq_dir}")
    print(f"Image path: {image_path}")
    print(f"Input frame: {frame} ({frame_mode})")
    print(f"Raw frame index: {raw_frame_idx}")
    print(f"Image number: {image_number}")
    print(f"valid_frame: {is_valid}")

    image = mpimg.imread(image_path)
    if show_pose:
        pose = get_gt_pose(annot, raw_frame_idx)
    else:
        pose = None

    if pose is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"{seq} image {image_number:06d}")
    else:
        fig = plt.figure(figsize=(14, 7))
        ax_img = fig.add_subplot(1, 2, 1)
        ax_pose = fig.add_subplot(1, 2, 2, projection="3d")

        ax_img.imshow(image)
        ax_img.axis("off")
        ax_img.set_title(f"{seq} image {image_number:06d}")
        draw_3d_pose(ax_pose, pose, "Dataset GT 3D pose")

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved: {output}")
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="View an MPI-INF-3DHP test-set image and optional GT 3D pose.")
    parser.add_argument("--dataset-root", required=True, help="Folder containing TS1/ts1 ... TS6/ts6.")
    parser.add_argument("-sq", "--seq", default="TS5", help="Sequence, e.g. TS5.")
    parser.add_argument("-fr", "--frame", type=int, default=12, help="Frame number or valid-frame index.")
    parser.add_argument(
        "--frame-mode",
        choices=["raw_image", "valid"],
        default="raw_image",
        help="raw_image means -fr 12 opens img_000012.jpg. valid means -fr 12 opens the 13th valid frame.",
    )
    parser.add_argument("--no-pose", action="store_true", help="Show only the original image.")
    parser.add_argument("--output", default="", help="Optional output PNG path.")
    return parser.parse_args()


def main():
    args = parse_args()
    visualize_frame(
        dataset_root=args.dataset_root,
        seq=args.seq,
        frame=args.frame,
        frame_mode=args.frame_mode,
        show_pose=not args.no_pose,
        output=args.output,
    )
    plt.show()


if __name__ == "__main__":
    main()
