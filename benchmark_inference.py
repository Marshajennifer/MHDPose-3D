'''
  python benchmark_inference.py
  runs with:
  tta_enabled: False
  Only this enables TTA:
  python benchmark_inference.py --bench-enable-tta
'''

import argparse
import os
import sys
import time

import torch

from common.arguments import parse_args


def parse_benchmark_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-d", "--dataset", default="h36m", choices=["h36m", "3dhp"], help="Dataset/model variant")
    parser.add_argument("-c", "--checkpoint", default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--evaluate", default="", help="Checkpoint file to evaluate")
    parser.add_argument("-f", "--number-of-frames", type=int, default=243, help="Temporal receptive field")
    parser.add_argument("-cs", type=int, default=512, help="Channel size")
    parser.add_argument("-dep", type=int, default=8, help="Model depth")
    parser.add_argument("-num_proposals", type=int, default=10, help="Number of generated hypotheses")
    parser.add_argument("-sampling_timesteps", type=int, default=5, help="Number of DDIM sampling steps")
    parser.add_argument("-gpu", default="0", help="GPU id")
    parser.add_argument("--bench-batch-size", type=int, default=1, help="Synthetic batch size for inference benchmarking")
    parser.add_argument("--bench-warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--bench-iters", type=int, default=30, help="Number of timed iterations")
    parser.add_argument("--bench-enable-tta", action="store_true", help="Enable test-time augmentation for benchmarking")
    parser.add_argument("--bench-seed", type=int, default=1234, help="Random seed for synthetic inputs")
    parser.add_argument("--bench-flop-proposals", type=int, default=1, help="Hypotheses used for denoiser FLOP/MAC estimate")
    bench_args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()
    args.test_time_augmentation = bench_args.bench_enable_tta
    args.dataset = bench_args.dataset
    args.checkpoint = bench_args.checkpoint
    args.evaluate = bench_args.evaluate
    args.number_of_frames = bench_args.number_of_frames
    args.cs = bench_args.cs
    args.dep = bench_args.dep
    args.num_proposals = bench_args.num_proposals
    args.sampling_timesteps = bench_args.sampling_timesteps
    args.gpu = bench_args.gpu
    args.bench_batch_size = bench_args.bench_batch_size
    args.bench_warmup = bench_args.bench_warmup
    args.bench_iters = bench_args.bench_iters
    args.bench_seed = bench_args.bench_seed
    args.bench_flop_proposals = bench_args.bench_flop_proposals
    return args


def build_h36m_model(args):
    from common.h36m_dataset import Human36mDataset
    from common.diffusionpose import DiffMamba

    dataset = Human36mDataset("data/data_3d_h36m.npz")
    skel = dataset.skeleton()
    joints_left = list(skel.joints_left())
    joints_right = list(skel.joints_right())
    parents_np = skel.parents()

    model = DiffMamba(
        args,
        joints_left,
        joints_right,
        is_train=False,
        num_proposals=args.num_proposals,
        sampling_timesteps=args.sampling_timesteps,
    )
    model.pose_estimator.attach_kinrope(
        parents_np, rope_eps=0.01, joints_left=joints_left, joints_right=joints_right
    )
    return model


def build_3dhp_model(args):
    from common.diffusionpose_3dhp import DiffMamba

    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]
    model = DiffMamba(
        args,
        joints_left,
        joints_right,
        is_train=False,
        num_proposals=args.num_proposals,
        sampling_timesteps=args.sampling_timesteps,
    )
    return model


def load_checkpoint_if_available(model, args, device):
    if not args.evaluate:
        return model

    chk_path = os.path.join(args.checkpoint, args.evaluate)
    checkpoint = torch.load(chk_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_pos"]
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)
    return model


def make_synthetic_inputs(args, device):
    torch.manual_seed(args.bench_seed)
    batch = args.bench_batch_size
    frames = args.number_of_frames
    joints = 17
    inputs_2d = torch.randn(batch, frames, joints, 2, device=device)
    inputs_3d = torch.randn(batch, frames, joints, 3, device=device)
    inputs_3d[:, :, 0] = 0

    inputs_2d_flip = None
    if args.test_time_augmentation:
        inputs_2d_flip = inputs_2d.clone()
        inputs_2d_flip[:, :, :, 0] *= -1
    return inputs_2d, inputs_3d, inputs_2d_flip


def selective_scan_flops(batch, length, d_inner, d_state):
    flops = d_inner * length * d_state
    flops += d_inner * length * d_state * 2
    flops += (d_inner * d_state + d_inner * d_state) * length
    flops += d_inner * length
    flops += d_inner * length
    return flops * batch


def estimate_scan_flops(model, inputs):
    from mamba_ssm import Mamba

    total_scan = 0
    hooks = []

    def make_hook():
        def hook(module, inp, out):
            nonlocal total_scan
            x = inp[0]
            if x.dim() != 3:
                return
            batch, length, _ = x.shape
            d_inner = module.d_inner
            d_state = module.d_state
            flops = selective_scan_flops(batch, length, d_inner, d_state)
            total_scan += flops
        return hook

    for module in model.modules():
        if isinstance(module, Mamba):
            hooks.append(module.register_forward_hook(make_hook()))

    with torch.no_grad():
        model(*inputs)

    for hook in hooks:
        hook.remove()

    return {
        "scan_flops": total_scan,
        "scan_macs": total_scan / 2,
    }


def estimate_denoiser_ops(model, args, device):
    pose_estimator = model.pose_estimator
    pose_estimator.eval()

    batch = args.bench_batch_size
    frames = args.number_of_frames
    joints = 17
    proposals = args.bench_flop_proposals

    torch.manual_seed(args.bench_seed)
    inputs_2d = torch.randn(batch, frames, joints, 2, device=device)
    noisy_3d = torch.randn(batch, proposals, frames, joints, 3, device=device)
    timestep = torch.zeros(batch, device=device)
    inputs = (inputs_2d, noisy_3d, timestep)

    from thop import profile
    with torch.no_grad():
        thop_macs, _ = profile(pose_estimator, inputs=inputs, verbose=False)

    scan = estimate_scan_flops(pose_estimator, inputs)
    combined_macs = thop_macs + scan["scan_macs"]
    combined_flops = thop_macs * 2 + scan["scan_flops"]
    tta_factor = 2 if args.test_time_augmentation else 1

    return {
        "flop_proposals": proposals,
        "denoiser_macs": combined_macs,
        "denoiser_flops": combined_flops,
        "denoiser_macs_per_frame": combined_macs / frames,
        "train_macs_per_frame_g": combined_macs / frames / 1e9,
        "infer_macs_per_frame_g": (
            combined_macs
            / frames
            / 1e9
            * args.num_proposals
            * args.sampling_timesteps
            * tta_factor
        ),
        "combined_macs": combined_macs,
        "combined_flops": combined_flops,
        "combined_macs_per_frame": combined_macs / frames,
    }


def benchmark_model(model, args, device):
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    ops = estimate_denoiser_ops(model, args, device)
    inputs_2d, inputs_3d, inputs_2d_flip = make_synthetic_inputs(args, device)

    if device.type != "cuda":
        raise RuntimeError("This benchmark is intended for CUDA devices.")

    with torch.no_grad():
        for _ in range(args.bench_warmup):
            _ = model(inputs_2d, inputs_3d, input_2d_flip=inputs_2d_flip)
        torch.cuda.synchronize(device)

        times_ms = []
        peak_mem_bytes = 0
        for _ in range(args.bench_iters):
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            _ = model(inputs_2d, inputs_3d, input_2d_flip=inputs_2d_flip)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            times_ms.append(elapsed_ms)
            peak_mem_bytes = max(peak_mem_bytes, torch.cuda.max_memory_allocated(device))

    avg_ms = sum(times_ms) / len(times_ms)
    std_ms = (sum((x - avg_ms) ** 2 for x in times_ms) / len(times_ms)) ** 0.5
    throughput = args.bench_batch_size / (avg_ms / 1000.0)
    fps = throughput * args.number_of_frames

    return {
        "params_m": params / 1_000_000.0,
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "throughput_clips_s": throughput,
        "fps": fps,
        "peak_mem_gb": peak_mem_bytes / (1024 ** 3),
        **ops,
    }


def main():
    args = parse_benchmark_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    if args.dataset == "h36m":
        model = build_h36m_model(args)
    elif args.dataset == "3dhp":
        model = build_3dhp_model(args)
    else:
        raise ValueError("dataset must be 'h36m' or '3dhp'")

    model = load_checkpoint_if_available(model, args, device)
    model = model.to(device)

    results = benchmark_model(model, args, device)

    print("=== Inference Benchmark ===")
    print(f"dataset: {args.dataset}")
    print(f"checkpoint_dir: {args.checkpoint}")
    print(f"checkpoint_file: {args.evaluate if args.evaluate else 'none'}")
    print("temporal_mode: seq_to_seq")
    print(f"frames: {args.number_of_frames}")
    print(f"rf: {args.number_of_frames}")
    print(f"output_frames_per_forward: {args.number_of_frames}")
    print(f"num_proposals: {args.num_proposals}")
    print(f"sampling_timesteps: {args.sampling_timesteps}")
    print(f"tta_enabled: {args.test_time_augmentation}")
    print(f"batch_size: {args.bench_batch_size}")
    print(f"warmup_iters: {args.bench_warmup}")
    print(f"timed_iters: {args.bench_iters}")
    print(f"params_m: {results['params_m']:.3f}")
    print(f"denoiser_flop_proposals: {results['flop_proposals']}")
    print(f"denoiser_flops_g: {results['denoiser_flops'] / 1e9:.3f}")
    print(f"denoiser_macs_g: {results['denoiser_macs'] / 1e9:.3f}")
    print(f"denoiser_macs_per_frame_m: {results['denoiser_macs_per_frame'] / 1e6:.3f}")
    print(f"train_macs_per_frame_g: {results['train_macs_per_frame_g']:.3f}")
    print(f"infer_macs_per_frame_g: {results['infer_macs_per_frame_g']:.3f}")
    print("denoiser_ops_note: thop-supported ops + analytical Mamba selective scan")
    print(f"latency_ms: {results['avg_ms']:.3f}")
    print(f"throughput: {results['throughput_clips_s']:.3f} clips/s")
    print(f"gpu_mem_gb: {results['peak_mem_gb']:.3f}")
    print(f"avg_latency_ms: {results['avg_ms']:.3f}")
    print(f"std_latency_ms: {results['std_ms']:.3f}")
    print(f"throughput_clips_per_s: {results['throughput_clips_s']:.3f}")
    print(f"fps: {results['fps']:.3f}")
    print(f"peak_memory_gb: {results['peak_mem_gb']:.3f}")


if __name__ == "__main__":
    main()
