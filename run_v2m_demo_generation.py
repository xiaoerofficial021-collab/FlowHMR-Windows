"""FlowHMR inference from raw videos: video -> SMPL-H motion (npz, + fbx if fbxsdkpy is installed).

    python run_v2m_demo_generation.py assets/example/example.mp4
    python run_v2m_demo_generation.py a.mp4 b.mov my_videos/ --ckpt models/flowhmr_base/flowhmr_base.ckpt

The default pipeline is: 30 fps transcode -> YOLOX person detection & tracking ->
VGGT-Omega camera -> SAM-3D-Body tokens -> FlowHMR generation. --static_camera skips
VGGT for a fixed camera using approximate intrinsics. Results land in
output/<video name>/<ckpt name>.npz; the extracted features are kept next to it and reused
when the same video is run again (e.g. with another checkpoint). --overwrite recomputes them.
"""
import argparse
import glob
import os
import shutil
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from flowhmr.model_files import core_model_paths, missing_model_paths

OUTPUT_ROOT = os.path.join(REPO_ROOT, "output")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def collect_videos(inputs):
    videos = []
    for path in inputs:
        if os.path.isdir(path):
            videos += sorted(p for p in glob.glob(os.path.join(path, "*"))
                             if p.lower().endswith(VIDEO_EXTS))
        elif os.path.isfile(path):
            videos.append(path)
        else:
            sys.exit(f"not found: {path}")
    if not videos:
        sys.exit("no input videos found")
    return [os.path.abspath(v) for v in videos]


def check_files(ckpt, config):
    required = core_model_paths(ckpt, config)
    missing = missing_model_paths(required)
    if missing:
        sys.exit("missing files (see models/README.md):\n  " + "\n  ".join(missing))
    return str(required[0]), str(required[1])


def run_video(engine, video, args, tag):
    stem = os.path.splitext(os.path.basename(video))[0]
    out_dir = os.path.join(OUTPUT_ROOT, stem)
    if args.overwrite:
        for name in (f"{stem}_30fps.mp4", f"{stem}_bbox.npz", f"{stem}_vggt_camera.npz",
                     f"{stem}_static_camera.npz", f"{stem}_sam3d_feat.pt", f"{stem}_sam3d_feat_static.pt"):
            path = os.path.join(out_dir, name)
            if os.path.exists(path):
                os.remove(path)

    engine.process_video(video, OUTPUT_ROOT, seed=args.seed, cfg_scale=args.cfg_scale,
                         vggt_frame_interval=args.vggt_interval, static_camera=args.static_camera,
                         sam_chunk_size=args.sam_chunk_size,
                         sam_hand_stride=args.sam_hand_stride)

    # process_video writes <stem>_seed<k>_origin.* (raw) and, with post-processing,
    # <stem>_seed<k>.*; keep the final one as <tag>.* and drop the other.
    gen = os.path.join(out_dir, f"{stem}_seed{args.seed}")
    final = gen if engine.postprocess is not None else gen + "_origin"
    for ext in (".npz", ".fbx"):
        if os.path.exists(final + ext):
            shutil.move(final + ext, os.path.join(out_dir, tag + ext))
        for leftover in (gen + ext, gen + "_origin" + ext):
            if os.path.exists(leftover):
                os.remove(leftover)
    result = os.path.join(out_dir, tag + ".npz")
    if not os.path.exists(result):
        raise FileNotFoundError(f"generation output missing: {result}")
    return result


def parse_args():
    p = argparse.ArgumentParser(description="FlowHMR: raw video -> SMPL-H motion")
    p.add_argument("videos", nargs="+", help="video files and/or directories of videos")
    p.add_argument("--ckpt", help="FlowHMR checkpoint (default: models/flowhmr_latest/flowhmr_latest.ckpt)")
    p.add_argument("--model_cfg", default=None,
                   help="model config yaml (default: config.yml next to the ckpt)")
    p.add_argument("--device", default="auto",
                   help="auto (default), cuda, cuda:N, rocm, mps, apple or cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=None,
                   help="flow-matching ODE steps (default: from config, 20)")
    p.add_argument("--postprocess", action="store_true",
                   help="apply RGBMocap contact planning and anatomical IK (see docs/install.md) "
                        "(off: raw model output)")
    p.add_argument("--vggt_interval", type=int, default=1,
                   help="run VGGT-Omega on every N-th frame (camera interpolated in between); "
                        "use 10-30 for long videos or limited GPU memory; ignored with --static_camera")
    p.add_argument("--static_camera", action="store_true",
                   help="skip VGGT for a fixed camera; use a constant pose and default intrinsics")
    p.add_argument("--sam_chunk_size", type=int, default=16,
                   help="frames per SAM body batch; hand batch is twice this size (default: 16)")
    p.add_argument("--sam_hand_stride", type=int, default=1, choices=(1, 2, 4),
                   help="decode SAM hand decoders on one seeded frame per N frames and "
                        "linearly interpolate the rest (1 = every frame, default)")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute cached detection / camera / SAM features")
    args = p.parse_args()
    if args.sam_chunk_size < 1:
        p.error("--sam_chunk_size must be a positive integer")
    return args


def main():
    args = parse_args()
    from flowhmr.device_support import detect_device

    backend = detect_device(args.device)
    args.device = backend.device
    print(f"backend: {backend.label} ({backend.device}, {backend.runtime or 'host'})")
    videos = collect_videos(args.videos)
    ckpt, model_cfg = check_files(args.ckpt, args.model_cfg)
    tag = os.path.splitext(os.path.basename(ckpt))[0]
    if args.seed != 0:
        tag += f"_seed{args.seed}"

    if args.device.startswith("cuda:"):
        torch.cuda.set_device(args.device)  # YOLOX builds its grids on the current device
    os.chdir(REPO_ROOT)  # body-model / asset paths are repo-relative

    from flowhmr.pipeline.v2m_inference import V2MInferenceEngine

    engine = V2MInferenceEngine.build(ckpt=ckpt, model_cfg_path=model_cfg,
                                      device=args.device, validation_steps=args.steps,
                                      postprocess=args.postprocess)

    results = []
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video}")
        t0 = time.time()
        try:
            out = run_video(engine, video, args, tag)
            results.append(("ok", video, out))
            print(f"  [ok] {time.time() - t0:.1f}s -> {out}")
        except Exception as e:  # keep going with the remaining videos
            import traceback
            traceback.print_exc()
            results.append(("failed", video, str(e)))
            print(f"  [failed] {time.time() - t0:.1f}s: {e}")

    n_ok = sum(r[0] == "ok" for r in results)
    print(f"\n=== Done | ok={n_ok} failed={len(results) - n_ok} ===")
    for status, video, info in results:
        print(f"  {status:6s} {video} -> {info}")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
