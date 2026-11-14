"""FlowHMR web demo: upload a video, get SMPL-H motion, inspect it in a 3D viewer.

Launch (from the repo root):
    python app.py --ckpt models/flowhmr_latest/flowhmr_latest.ckpt --port 8080
then open http://<host>:8080

Pipeline per job (jobs are queued):
    transcode 30fps -> YOLOX detection + tracking -> VGGT-Omega camera
    -> SAM-3D-Body tokens -> FlowHMR flow-matching generation
"""
import argparse
import base64
import binascii
import copy
import hashlib
import hmac
import json
import math
import mimetypes
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import defaultdict
from contextlib import closing, contextmanager
from functools import wraps
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory
from werkzeug.serving import make_server

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(WEB_DIR))
sys.path.insert(0, REPO_ROOT)

from flowhmr.utils.runtime_capabilities import (
    EmptyPersonRangeError, InferenceRAMError, SAMMemoryError, SharedCameraSamplingError, SharedGPUMemoryError,
    SOLE_FIT_REFERENCE_STRENGTH, cached_detection_matches, native_correction_available, requested_frame_range,
)
from flowhmr.utils.runtime_hamer import hamer_available, HAMER_ISSUES, HAMER_STRIDES, MISSING_HAMER
from flowhmr.utils.runtime_preview_inspection import PreviewInspection
from flowhmr.utils.runtime_preview_thumbnail import ensure_thumbnail
from flowhmr.core.bodymodels.body_variants import available_variants, model_path, write_preview_vertices
from flowhmr.model_files import (
    HAMER_FILES, MODEL_ROOT, SAM3D_FILES, VGGT, SMPLH_FILES, core_model_paths, ensure_model_folders,
    flowhmr_paths, missing_model_paths, resolve_model, smplh_reference,
)
from flowhmr import model_integrity
from flowhmr.character_files import CharacterFiles, KEY_ENV as CHARACTER_KEY_ENV
from flowhmr.character_library import CharacterLibrary
from flowhmr.model_sources import PUBLIC_ASSETS
from flowhmr.model_import import ModelImporter, ModelImportError
from flowhmr.model_inventory import model_inventory
from flowhmr.model_downloads import ACTIVE_DOWNLOAD, DownloadError, ModelDownloads, missing_downloads
from flowhmr.runtime_environments import RUNTIME, cuda_ordinal
from flowhmr import runtime_rocm, runtime_xpu
from flowhmr.mcp.service import MCPService
from flowhmr.mcp.contract import (Parameters as MCPParameters, ExportOptions as MCPExportOptions,
                                  SERVICE_KEY_HEADER, TOOL_NAMES, parameter_defaults, single_result_person_id)
from flowhmr.mcp.characters import BONES, PosePatch, empty_pose, import_character
from flowhmr.mcp.activity import Activity
from flowhmr.mcp.events import UIEvents

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
STAGES = [
    ("detection", "Detection", "YOLOX person detection & tracking"),
    ("camera", "Camera", "VGGT-Omega camera estimation"),
    ("feature_extract", "Body features", "SAM-3D-Body feature extraction"),
    ("generation", "Generation", "FlowHMR flow-matching generation"),
]
HAMER_STAGE = ("hamer", "Hand refinement", "HaMeR 手指精修")
ALL_STAGES = (*STAGES, HAMER_STAGE)
STAGE_INDEX = {name: i for i, (name, _, _) in enumerate(ALL_STAGES)}
SUB = "input"  # every job stores its upload as input.<ext>, so intermediates live in <job>/input/

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
MCP_SERVICE = MCPService(REPO_ROOT)
MCP_ACTIVITY = Activity()
UI_EVENTS = UIEvents()
MCP_SERVICE.on_change = lambda: UI_EVENTS.publish('mcp_service')


@app.after_request
def notify_ui_changes(response):
    if request.path == '/api/mcp':
        response.headers['Cache-Control'] = 'no-store'
    if request.method in ('POST', 'PUT', 'PATCH', 'DELETE') and response.status_code < 300:
        path = request.path
        if path == '/api/upload' or path.startswith('/api/jobs'):
            UI_EVENTS.publish('jobs')
        elif path.startswith('/api/character-library') or path.startswith('/api/mcp/characters') or (
                path.startswith('/api/mcp/jobs/') and path.endswith('/character')):
            UI_EVENTS.publish('characters')
        elif path == '/api/mcp':
            UI_EVENTS.publish('mcp_service')
    return response


@app.get('/api/ui/events')
def api_ui_events():
    if not local_model_request() or request.headers.get('Sec-Fetch-Site') == 'cross-site':
        abort(403)
    try:
        cursor = int(request.headers.get('Last-Event-ID', UI_EVENTS.sequence))
        if cursor < 0:
            raise ValueError()
    except ValueError:
        abort(400)

    def stream():
        nonlocal cursor
        yield 'event: connected\ndata: {}\n\n'
        while True:
            events = UI_EVENTS.wait(cursor)
            if not events:
                yield ': heartbeat\n\n'
            for event in events:
                cursor = event['id']
                yield f"id: {cursor}\nevent: update\ndata: {json.dumps(event)}\n\n"
    return Response(stream(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.post('/api/mcp/events')
def api_mcp_events():
    require_mcp_service()
    value = request.get_json()
    if not isinstance(value, dict) or value.get('kind') != 'exports':
        abort(400)
    UI_EVENTS.publish('exports')
    return jsonify(ok=True)


@app.get('/api/mcp/viewer-snapshot')
def api_mcp_viewer_snapshot():
    require_mcp_service()
    snapshot = UI_EVENTS.request_snapshot()
    if snapshot is None:
        return api_error('viewer_unavailable', 'No connected UI responded to the snapshot request.', 503)
    return jsonify(snapshot)


@app.post('/api/mcp/viewer-command')
def api_mcp_viewer_command():
    require_mcp_service()
    value = request.get_json()
    if (not isinstance(value, dict) or set(value) != {'job_id', 'action', 'frame'}
            or not isinstance(value['job_id'], str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value['job_id'])
            or value['action'] not in ('seek', 'play', 'pause')
            or (value['action'] == 'seek' and (type(value['frame']) is not int or value['frame'] < 0))
            or (value['action'] != 'seek' and value['frame'] is not None)):
        return api_error('invalid_playback_command', 'Invalid playback command.', 400)
    result = UI_EVENTS.request_command(value)
    if result is None:
        return api_error('outcome_unknown', 'No UI acknowledged the command; inspect the viewer before retrying.', 504)
    if 'error' in result:
        return api_error('viewer_command_rejected', result['error'], 409)
    return jsonify(result)


@app.post('/api/ui/viewer-command/<token>/<operation>')
def api_ui_viewer_command(token, operation):
    check_character_library_request()
    if not re.fullmatch(r'[0-9a-f]{32}', token):
        abort(400)
    if operation == 'claim':
        accepted = UI_EVENTS.claim_command(token)
    elif operation == 'reply':
        value = request.get_json()
        if not isinstance(value, dict):
            abort(400)
        accepted = UI_EVENTS.reply_command(token, value)
    else:
        abort(404)
    if not accepted:
        return api_error('viewer_command_expired', 'Command is already claimed or no longer pending.', 410)
    return jsonify(ok=True)


@app.post('/api/ui/viewer-snapshot/<token>')
def api_ui_viewer_snapshot_reply(token):
    if not local_model_request() or request.headers.get('Sec-Fetch-Site') == 'cross-site':
        abort(403)
    if request.content_length and request.content_length > 2 * 1024 * 1024:
        abort(413)
    snapshot = request.get_json()
    if not isinstance(snapshot, dict) or not re.fullmatch(r'[0-9a-f]{32}', token):
        abort(400)
    if not UI_EVENTS.reply_snapshot(token, snapshot):
        return api_error('snapshot_expired', 'Snapshot request is no longer pending.', 410)
    return jsonify(ok=True)


@app.route('/api/mcp/activity', methods=['GET', 'POST'])
def api_mcp_activity():
    if request.method == 'GET':
        require_desktop_session()
        directory = Path(REPO_ROOT) / '.cache/mcp/exports'
        files = sorted(directory.glob('*.json'), key=lambda path: path.stat().st_mtime, reverse=True)[:20]
        exports = [json.loads(path.read_text('utf-8')) for path in files]
        for row in exports:
            row.pop('temporary_path', None)
        return jsonify(operations=MCP_ACTIVITY.snapshot(), exports=exports)
    require_mcp_service()
    value = request.get_json()
    if (not isinstance(value, dict) or value.get('tool') not in TOOL_NAMES
            or value.get('state') not in ('running', 'done', 'failed')):
        abort(400)
    try:
        row = MCP_ACTIVITY.record(value)
        UI_EVENTS.publish('activity')
        if row['state'] == 'done' and row['tool'] in (
                'flowhmr_import_character', 'flowhmr_select_character', 'flowhmr_update_character_pose'):
            UI_EVENTS.publish('characters')
        return jsonify(row)
    except ValueError as error:
        return api_error('mcp_activity_invalid', str(error), 400)

# On Windows, Python's mimetypes resolves extensions through the registry, so a
# machine whose HKCR\.js "Content Type" is text/plain makes every .js file we
# serve come back as text/plain. Browsers enforce a JavaScript MIME type on
# module scripts and silently refuse to run the page, so pin the types we ship
# instead of trusting whatever the host reports.
for _extension, _mime_type in (
    (".js", "text/javascript"),
    (".mjs", "text/javascript"),
    (".css", "text/css"),
    (".json", "application/json"),
    (".map", "application/json"),
    (".wasm", "application/wasm"),
    (".svg", "image/svg+xml"),
    (".glb", "model/gltf-binary"),
    (".gltf", "model/gltf+json"),
):
    mimetypes.add_type(_mime_type, _extension)

MODEL_DOWNLOADS = ModelDownloads(app.logger)
MODEL_IMPORTER = ModelImporter()
MODEL_VERIFY_LOCK = threading.Lock()
RUNTIME.logger = app.logger
RUNTIME_SWITCH_LOCK = threading.Lock()

ENGINE = None
INSPECTION = None
ARGS = None
JOBS = {}
MEMBER_DIRS = {}
JOBS_LOCK = threading.Lock()
FBX_EXPORT_LOCK = threading.Lock()
BODY_PREVIEW_LOCK = threading.Lock()
JOB_QUEUE = queue.Queue()
_POSTPROCESS = None
SELECTIONS = {}           # job_id -> {"event": Event, "people": list[int] | None}
CANCEL_REQUESTS = set()
SELECT_TIMEOUT_S = 600    # a job waiting for a person pick falls back to the default after this
ACTIVE_STATUSES = {"queued", "running", "waiting_user", "cancel_requested", "uploading"}


class JobCancelled(Exception):
    pass


class UserMessage(ValueError):
    def __init__(self, message, code, **params):
        super().__init__(message)
        self.code = code
        self.params = params


# --------------------------------------------------------------------------- engine
def checked_model_paths(args, body_reference=None):
    paths = core_model_paths(args.ckpt, args.model_cfg, body_reference=body_reference)
    missing = missing_model_paths(paths)
    if missing:
        listing = "\n".join(missing)
        raise UserMessage("推理模型或配置未就绪。请在右上角「模型」查看：\n" + listing,
                          "model_files_not_ready", files=listing)
    if RUNTIME.cuda_variant() == "cu121":
        _check_cu121_weights(paths)
    return paths


def _check_cu121_weights(paths):
    # Official cu121 uses PyTorch 2.5.1, whose weights_only loader cannot
    # safely open unknown files (CVE-2025-32434).
    unverified = []
    for path in paths:
        if path.suffix.lower() not in (".pt", ".pth", ".ckpt"):
            continue
        asset = model_integrity.pinned_asset(model_integrity.relative_name(path))
        try:
            valid = asset is not None and asset.sha256 is not None and model_integrity.digest(path) == asset.sha256
        except OSError:
            valid = False
        if not valid:
            unverified.append(asset.relative if asset else path.name)
    if unverified:
        listing = "\n".join(unverified)
        raise UserMessage("CUDA 12.1 只能使用已通过校验的官方权重。请在「模型」重新安装对应文件：\n" + listing,
                          "runtime_cu121_model_unverified", files=listing)


def resolve_inference_device(requested):
    import torch

    device = torch.device(requested)
    if device.type == "xpu":
        try:
            index = runtime_xpu.xpu_ordinal(torch, ordinal=device.index)
            torch.xpu.set_device(index)
        except (AttributeError, RuntimeError) as error:
            raise UserMessage("Intel XPU 需要 Windows 11 和受支持的 Intel Arc 显卡。请检查驱动。",
                              "runtime_xpu_no_device") from error
        return torch.device("xpu", index)
    if device.type == "cuda":
        if sys.platform == "win32" and torch.version.hip:
            profile = runtime_rocm.PROFILES.get(RUNTIME.active)
            try:
                index = runtime_rocm.gpu_ordinal(torch, profile.arch if profile else None, ordinal=device.index)
            except RuntimeError as error:
                if profile:
                    raise UserMessage(f"当前 ROCm 环境需要可用的 {profile.arch} 显卡。",
                                      "runtime_rocm_gpu_required", arch=profile.arch) from error
                raise UserMessage("没有可用于当前 ROCm 环境的显卡。请检查驱动和显卡设置。",
                                  "runtime_rocm_no_device") from error
            device = torch.device("cuda", index)
        elif variant := RUNTIME.cuda_variant():
            try:
                device = torch.device("cuda", cuda_ordinal(torch, variant, device.index))
            except RuntimeError as error:
                raise UserMessage("此版本不适用于可用的 NVIDIA 显卡。", "runtime_cuda_arch_mismatch") from error
        torch.cuda.set_device(device.index if device.index is not None else 0)
        runtime_rocm.configure_windows_hip(torch, device=device, attention=True)
    return device


def load_engine(args, body_reference, paths, *, sam_backend="gpu"):
    """Validate and load the models from the job worker, never from startup or HTTP."""
    ckpt, model_cfg = paths[:2]
    device = resolve_inference_device(args.device)
    if sam_backend == "hybrid":
        import torch
        from flowhmr.utils.sam_cpu_backbone import cpu_backbone_supported

        if device.type != "cuda" or not runtime_rocm.uses_shared_memory(torch, device) or not cpu_backbone_supported():
            raise UserMessage("混合 SAM 需要 Windows AMD 核显和支持 AVX-512 BF16 的 CPU。", "sam_hybrid_unavailable")
    from flowhmr.pipeline.v2m_inference import V2MInferenceEngine

    return V2MInferenceEngine.build(
        ckpt=str(ckpt), model_cfg_path=str(model_cfg), device=str(device),
        smpl_model_path=str(paths[-2]), body_reference=body_reference,
        sam_cpu_backbone=sam_backend == "hybrid",
    )


def job_body_reference(job):
    reference = job.get("body_reference")
    return reference if isinstance(reference, str) and reference in SMPLH_FILES else "neutral"


def missing_body_reference(reference):
    return UserMessage("请安装此任务生成时使用的 SMPL-H 体型。",
                       "body_reference_missing", body=reference)


def get_engine(job_id, requested_reference=None, *, sam_backend="gpu"):
    global ENGINE, _POSTPROCESS
    from flowhmr.core.bodymodels.smplh_data import validated_smplh_reference

    _raise_if_cancelled(job_id)
    try:
        reference, _ = validated_smplh_reference(requested_reference)
    except (OSError, ValueError) as error:
        if requested_reference:
            raise missing_body_reference(requested_reference) from error
        raise UserMessage("尚未安装可用的 SMPL-H 体型。请导入官方 MANO v1.2 的男／女体型，或 Extended SMPL+H 的中性体型。",
                          "model_body_required") from error
    paths = checked_model_paths(ARGS, reference)
    if ENGINE is None or ENGINE.body_reference != reference or ENGINE.sam_backend != sam_backend:
        old = ENGINE
        ENGINE = _POSTPROCESS = None
        if old is not None:
            stop = getattr(old.sam_extractor, "stop", None)
            if stop is not None:
                stop()
            del old
            cuda = getattr(sys.modules.get("torch"), "cuda", None)
            if getattr(cuda, "is_initialized", lambda: False)():
                cuda.empty_cache()
        ENGINE = load_engine(ARGS, reference, paths, sam_backend=sam_backend)
    with JOBS_LOCK:
        if job_id in CANCEL_REQUESTS:
            raise JobCancelled()
        JOBS[job_id].update(loading_models=False, message="Starting", body_reference=reference)
        _persist_job_locked(JOBS[job_id])
    return ENGINE


def get_postprocess(floor, preset, *, sole_fit_strength=0.0):
    """Post-processing pipeline, built on first use (only needed when a job asks for it)."""
    global _POSTPROCESS
    if _POSTPROCESS is None:
        from flowhmr.utils.runtime_v2m_postprocess_simple import PostprocessPipeline
        _POSTPROCESS = PostprocessPipeline.default(body_model=ENGINE.body_model,
                                                   smpl_mesh=ENGINE.smpl_mesh)
    _POSTPROCESS.configure(floor, preset=preset, sole_fit_strength=sole_fit_strength)
    return _POSTPROCESS


# --------------------------------------------------------------------------- parameters
# SAM hand-decoder sampling intervals exposed in the Settings card (1 = every frame).
SAM_HAND_STRIDES = (1, 2, 4)


# Defaults for the page's Settings card; everything is editable per run in the UI.
DEFAULT_PARAMS = {
    "seed": 0, "cfg_scale": 1.0, "steps": 20, "max_frames": None, "vggt_interval": 0,
    "sam_chunk_size": None, "sam_hand_stride": 1, "start_frame": 0, "end_frame": None,
    "sam_backend": "gpu",
    "postprocess": True, "manual_select": True, "auto_frames": True, "static_camera": True,
    "correction_floor_m": None, "correction_preset": "strong",
    "sole_fit": True, "sole_fit_strength": SOLE_FIT_REFERENCE_STRENGTH,
    "smoothing_frames": 0.12, "hamer": False, "hamer_stride": 1,  # 0.04 on the 0–1 slider
}


def default_params():
    return {**DEFAULT_PARAMS, "postprocess": DEFAULT_PARAMS["postprocess"] and native_correction_available()}


def _to_bool(value, default):
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "on", "yes")


def _read_sole_fit_strength(settings):
    raw = settings.get("sole_fit_strength")
    if raw is None:
        return SOLE_FIT_REFERENCE_STRENGTH if _to_bool(settings.get("sole_fit"), False) else 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        raise UserMessage("脚掌贴地强度须为 0 到 1 的数字。", "param_sole_fit") from None
    if isinstance(raw, bool) or not 0.0 <= value <= 1.0:
        raise UserMessage("脚掌贴地强度须为 0 到 1 的数字。", "param_sole_fit")
    return value


def parse_params(src, base=None):
    """Validate per-job parameters from a form / JSON body; missing keys fall back to `base`."""
    defaults = default_params()
    fit_settings = next(settings for settings in (src, base or {}, defaults)
                        if "sole_fit_strength" in settings or "sole_fit" in settings)
    sole_fit_strength = _read_sole_fit_strength(fit_settings)
    base = {**defaults, **(base or {})}

    def number(key, cast, lo, hi):
        raw = src.get(key)
        if raw is None or raw == "":
            return base[key]
        try:
            value = cast(raw)
        except (TypeError, ValueError, OverflowError):
            raise UserMessage(f"invalid {key}: {raw!r}", "param_numeric", field=key) from None
        if not lo <= value <= hi:
            raise UserMessage(f"{key} must be between {lo} and {hi}", "param_bounds", field=key, min=lo, max=hi)
        return value

    def integer(key, minimum, *, optional=False):
        raw = src.get(key, base[key])
        if optional and raw in (None, "", "auto", "null"):
            return None
        value = int(raw.strip()) if isinstance(raw, str) and raw.strip().isdecimal() else raw
        if type(value) is not int or value < minimum:
            raise UserMessage(f"{key} must be an integer >= {minimum}", "param_integer", field=key, min=minimum)
        return value

    start = integer("start_frame", 0)
    end = integer("end_frame", 0, optional=True)
    if end is not None and end < start:
        raise UserMessage("结束帧不能早于起始帧。", "param_frame_order")
    hamer_stride = integer("hamer_stride", 1)
    if hamer_stride not in HAMER_STRIDES:
        raise UserMessage("HaMeR 采样间隔须为 1、2、3 或 4 帧。", "param_hamer_stride")
    sam_hand_stride = integer("sam_hand_stride", 1)
    if sam_hand_stride not in SAM_HAND_STRIDES:
        raise UserMessage("SAM 手部采样间隔须为逐帧、每 2 帧或每 4 帧。", "param_sam_hand_stride")
    sam_backend = src.get("sam_backend", base["sam_backend"])
    if sam_backend not in ("gpu", "hybrid"):
        raise UserMessage("请选择有效的 SAM 运行模式。", "param_sam_backend")
    if sam_backend == "hybrid" and (str(ARGS.device).split(":", 1)[0] != "cuda"
                                    or not runtime_rocm.sam_hybrid_candidate(RUNTIME.active)):
        raise UserMessage("混合 SAM 需要 Windows AMD 核显和支持 AVX-512 BF16 的 CPU。", "sam_hybrid_unavailable")
    try:
        smoothing = float(src.get("smoothing_frames", base["smoothing_frames"]))
    except (TypeError, ValueError, OverflowError):
        raise UserMessage("平滑强度须为大于或等于 0 的数字。", "param_smoothing") from None
    if not math.isfinite(smoothing) or smoothing < 0:
        raise UserMessage("平滑强度须为大于或等于 0 的有限值。", "param_smoothing")

    floor = src.get("correction_floor_m", base["correction_floor_m"])
    if floor in (None, "", "null", "auto"):
        floor = None
    else:
        try:
            floor = float(floor)
        except (ValueError, TypeError, OverflowError):
            raise UserMessage("修正地面高度须为有效数字（米）。", "param_floor") from None
        if not math.isfinite(floor):
            raise UserMessage("修正地面高度须为有限值。", "param_floor")

    preset = src.get("correction_preset", base["correction_preset"])
    if preset not in ("normal", "strong"):
        raise UserMessage("脚步锁定须选择标准或加强。", "param_lock")

    vggt_interval = number("vggt_interval", int, 0, 120)
    return {
        "seed": number("seed", int, 0, 2 ** 31 - 1),
        "cfg_scale": number("cfg_scale", float, 0.0, 10.0),
        "steps": number("steps", int, 1, 100),
        "max_frames": integer("max_frames", 1, optional=True),  # legacy prefix limit
        "auto_frames": _to_bool(src.get("auto_frames"), base["auto_frames"]),
        "start_frame": start, "end_frame": end,
        "smoothing_frames": smoothing,
        "hamer": _to_bool(src.get("hamer"), base["hamer"]),
        "hamer_stride": hamer_stride,
        "vggt_interval": vggt_interval,
        "sam_chunk_size": integer("sam_chunk_size", 1, optional=True),
        "sam_hand_stride": sam_hand_stride,
        "sam_backend": sam_backend,
        "postprocess": _to_bool(src.get("postprocess"), base["postprocess"]),
        "correction_floor_m": floor,
        "correction_preset": preset,
        "sole_fit": sole_fit_strength > 0,
        "sole_fit_strength": sole_fit_strength,
        "manual_select": _to_bool(src.get("manual_select"), base.get("manual_select", True)),
        "static_camera": _to_bool(src.get("static_camera"), base["static_camera"] and vggt_interval == 0),
    }


# --------------------------------------------------------------------------- jobs
def job_detection_range(params):
    legacy_limit = None if params.get("auto_frames", True) else params.get("max_frames")
    return requested_frame_range(params.get("start_frame", 0), params.get("end_frame"), legacy_limit)


def job_dir(job_id):
    return MEMBER_DIRS.get(job_id, os.path.join(ARGS.work_dir, job_id))


def camera_summary(directory, params):
    if params.get("static_camera"):
        return {"mode": "static"}
    path = os.path.join(directory, SUB, f"{SUB}_vggt_camera.npz")
    try:
        with np.load(path, allow_pickle=False) as data:
            return {"mode": "manual" if params.get("vggt_interval") else "auto",
                    "interval": int(data["camera_sampling_interval"]),
                    "samples": int(data["camera_sample_indices"].size),
                    "frames": len(data["K"]),
                    "profile": str(data["camera_profile"]) if "camera_profile" in data else None}
    except (OSError, ValueError, KeyError, BadZipFile):
        return None


def _write_job(job, directory):
    path = os.path.join(directory, "job.json")
    temp = path + ".tmp"
    saved = dict(job)
    if job.get("status") != "waiting_user":
        saved["candidates"] = None
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(saved, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _persist_job_locked(job):
    _write_job(job, job_dir(job["id"]))
    if job.get('ext') and os.path.isfile(os.path.join(job_dir(job['id']), f"input{job['ext']}")):
        UI_EVENTS.publish('jobs', {'job_id': job.get('person_batch') or job['id']})


def _legacy_job(job_id, directory):
    """Recover pre-persistence runs from their input and completed result files."""
    inputs = [name for name in os.listdir(directory)
              if name.startswith("input.") and os.path.isfile(os.path.join(directory, name))]
    if not inputs:
        return None
    input_name = inputs[0]
    created = os.path.getmtime(os.path.join(directory, input_name))
    try:
        created = time.mktime(time.strptime(job_id[:15], "%Y%m%d_%H%M%S"))
    except ValueError:
        pass
    meta_path = os.path.join(directory, "mesh", "meta.json")
    motion_path = os.path.join(directory, "motion.npz")
    done = os.path.isfile(meta_path) and os.path.isfile(motion_path)
    meta = {}
    if done:
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            done = False
    status = "done" if done else "failed"
    stages = [{"name": name, "label": label, "detail": detail,
               "status": "completed" if done else ("failed" if i == 0 else "pending"),
               "progress": 100 if done else 0,
               "start": created if done else None, "end": created if done else None}
              for i, (name, label, detail) in enumerate(STAGES)]
    return {
        "id": job_id, "filename": f"Recovered run {job_id}", "params": default_params(),
        "recovered": True,
        "ext": os.path.splitext(input_name)[1], "status": status,
        "message": "Done" if done else "Recovered from an earlier session; detailed progress is unavailable.",
        "error": None if done else "This earlier run did not have saved job details.",
        "error_code": None if done else "job_details_unavailable",
        "progress": 100 if done else 0, "created": created,
        "started": None, "finished": created if done else None,
        "num_frames": meta.get("num_frames") if done else None, "fps": meta.get("fps", 30),
        "has_fbx": os.path.isfile(os.path.join(directory, "motion.fbx")),
        "candidates": None, "default_person": None, "selected_person": None,
        "selection_auto": False, "stages": stages,
    }


def restore_jobs():
    """Restore persisted jobs and migrate older run folders into the history."""
    def job_folders(directory):
        with os.scandir(directory) as entries:
            return [e for e in entries if e.is_dir(follow_symlinks=False)
                    and re.fullmatch(r"\d{8}_\d{6}_[a-f0-9]{6}", e.name)]

    roots = job_folders(ARGS.work_dir)
    folders = list(roots)
    for root in roots:
        nested = os.path.join(root.path, "members")
        if os.path.isdir(nested) and not os.path.islink(nested):
            for entry in job_folders(nested):
                MEMBER_DIRS[entry.name] = entry.path
                folders.append(entry)
    for entry in folders:
        path = os.path.join(entry.path, "job.json")
        try:
            with open(path, encoding="utf-8") as f:
                job = json.load(f)
            if not isinstance(job, dict) or not isinstance(job.get("params"), dict):
                raise ValueError("invalid job metadata")
            job["id"] = entry.name
        except FileNotFoundError:
            job = _legacy_job(entry.name, entry.path)
            if job is None:
                continue
        except (OSError, ValueError):
            job = _legacy_job(entry.name, entry.path)
            if job is None:
                continue

        if job.get("status") in ("queued", "running", "waiting_user", "cancel_requested", "uploading"):
            cancelled = job["status"] == "cancel_requested"
            interrupted_at = time.time()
            job.update(status="cancelled" if cancelled else "failed",
                       message="Run stopped when FlowHMR restarted." if cancelled
                       else "Run interrupted when FlowHMR restarted.",
                       error=None if cancelled else "The server restarted before this run finished.",
                       error_code=None if cancelled else "job_interrupted", error_params={},
                       finished=interrupted_at, loading_models=False)
            for stage in job.get("stages", []):
                if stage.get("status") in ("running", "waiting_user"):
                    paused_at = stage.pop("paused_at", None)
                    if paused_at is not None:
                        stage["paused_seconds"] = stage.get("paused_seconds", 0.0) + max(
                            0.0, interrupted_at - paused_at,
                        )
                    stage.update(status="cancelled" if cancelled else "failed", end=interrupted_at)
        saved_params = job.get("params", {})
        try:
            strength = _read_sole_fit_strength(saved_params)
        except ValueError:
            strength = 0.0
        # Older jobs that omit these settings retain the defaults they ran with.
        job["params"] = {**DEFAULT_PARAMS, "static_camera": False, "postprocess": False,
                         "correction_preset": "normal", "smoothing_frames": 0.0, **saved_params,
                         "sole_fit": strength > 0, "sole_fit_strength": strength}
        job.setdefault("stages", [])
        report = job.get("hamer_report")
        if report and not any(stage["name"] == "hamer" for stage in job["stages"]):
            name, label, detail = HAMER_STAGE
            job["stages"].append({"name": name, "label": label, "detail": detail, "status": "completed",
                                  "progress": 100, "start": None, "end": None, "elapsed_seconds": report.get("seconds")})
        if job["status"] == "done" and "camera" not in job:
            job["camera"] = camera_summary(entry.path, job["params"])
        if job.get("person_batch") == job["id"]:
            manifest = Path(entry.path) / "scene" / "scene.json"
            try:
                scene = json.loads(manifest.read_text(encoding="utf-8"))
                if not (manifest.parent / "motion.npz").is_file() or not scene["people"]:
                    raise ValueError("Incomplete scene")
                job.update(scene=scene, scene_state="ready", scene_error=None)
            except (OSError, ValueError, KeyError, TypeError):
                job.pop("scene", None)
                if job.get("scene_state") in ACTIVE_STATUSES or job.get("scene_state") == "ready":
                    job["scene_state"] = "pending"
        with JOBS_LOCK:
            JOBS[job["id"]] = job
            _persist_job_locked(job)


def _build_job(filename, params, ext):
    job_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    job = {
        "id": job_id, "filename": filename, "params": params, "ext": ext, "recovered": False,
        "status": "queued", "message": "Waiting in queue", "error": None, "progress": 0,
        "source": "desktop", "selection_policy": "legacy",
        "created": time.time(), "started": None, "finished": None,
        "num_frames": None, "fps": 30, "has_fbx": False,
        "candidates": None, "default_person": None, "selected_person": None, "selection_auto": False,
        "stages": [{"name": n, "label": l, "detail": d, "status": "pending", "progress": 0,
                    "start": None, "end": None} for n, l, d in (ALL_STAGES if params.get("hamer") else STAGES)],
    }
    if params.get("static_camera", False):
        job["stages"][STAGE_INDEX["camera"]].update(
            label="Static camera", detail="Camera estimation skipped; fixed camera pose",
        )
    return job


def new_job(filename, params, ext, **metadata):
    job = _build_job(filename, params, ext)
    job.update(metadata)
    os.makedirs(job_dir(job["id"]), exist_ok=True)
    with JOBS_LOCK:
        JOBS[job["id"]] = job
        _persist_job_locked(job)
    return job


def update_job(job_id, **kw):
    with JOBS_LOCK:
        JOBS[job_id].update(kw)
        _persist_job_locked(JOBS[job_id])


def input_video(job_id):
    return os.path.join(job_dir(job_id), f"input{JOBS[job_id]['ext']}")


def enqueue(job):
    JOB_QUEUE.put(job["id"])
    UI_EVENTS.publish('jobs', {'job_id': job.get('person_batch') or job['id']})


def worker_loop():
    global ENGINE, _POSTPROCESS
    while True:
        job_id = JOB_QUEUE.get()
        if isinstance(job_id, tuple):
            try:
                _assemble_scene(job_id[1], automatic=False)
            finally:
                JOB_QUEUE.task_done()
            continue
        try:
            run_job(job_id)
        except JobCancelled:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    for stage in job["stages"]:
                        if stage["status"] == "running":
                            stage["status"] = "cancelled"
                    job.update(status="cancelled", message="Stopped by user", loading_models=False, finished=time.time())
                    _persist_job_locked(job)
        except Exception as e:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None:
                    continue
                cancelled = job_id in CANCEL_REQUESTS
                for s in job["stages"]:
                    if s["status"] == "running":
                        s["status"] = "cancelled" if cancelled else "failed"
                detail = f"{type(e).__name__}: {e}"
                if isinstance(e, UserMessage):
                    code, params = e.code, e.params
                elif isinstance(e, EmptyPersonRangeError):
                    code, params = "detection_empty_range", {}
                elif isinstance(e, SharedCameraSamplingError):
                    code, params = "rocm_camera_sampling", {}
                elif isinstance(e, SAMMemoryError):
                    code, params = f"sam_low_{e.resource}", {}
                elif isinstance(e, SharedGPUMemoryError):
                    code, params = "rocm_shared_ram_low", e.params
                elif isinstance(e, InferenceRAMError):
                    code, params = "inference_ram_low", e.params
                else:
                    code, params = "job_failed", {"detail": detail}
                job.update(status="cancelled" if cancelled else "failed",
                           error=None if cancelled else detail, error_code=None if cancelled else code,
                           error_params={} if cancelled else params,
                           message="Stopped by user" if cancelled else "Failed", loading_models=False, finished=time.time())
                _persist_job_locked(job)
            if not cancelled:
                traceback.print_exc()
        finally:
            with JOBS_LOCK:
                CANCEL_REQUESTS.discard(job_id)
                batch = JOBS.get(job_id, {}).get("person_batch")
            try:
                if ENGINE is not None:
                    try:
                        ENGINE.release_stage()
                    except Exception:
                        app.logger.exception("Unable to park the active inference model; discarding the engine")
                        ENGINE = _POSTPROCESS = None
                cuda = getattr(sys.modules.get("torch"), "cuda", None)
                if getattr(cuda, "is_initialized", lambda: False)():
                    cuda.empty_cache()
                if batch:
                    _assemble_scene(batch, automatic=True)
            except RuntimeError:
                traceback.print_exc()
            finally:
                JOB_QUEUE.task_done()


def _raise_if_cancelled(job_id):
    with JOBS_LOCK:
        if job_id in CANCEL_REQUESTS:
            raise JobCancelled()


def _link_video(source, target):
    # Source and transcoded videos are immutable. Only link videos, never per-person caches.
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _copy_person_thumbnail(directory, destination, person):
    name = f"{person}.jpg"
    source = os.path.join(directory, "persons", name)
    if os.path.isfile(source):
        os.makedirs(os.path.join(destination, "persons"), exist_ok=True)
        shutil.copy2(source, os.path.join(destination, "persons", name))


def _selected_track(candidate, *, automatic=False):
    return {"selected_person": candidate["index"], "selection_auto": automatic,
            "person_track": {key: candidate[key] for key in ("track_id", "num_frames", "start", "end")}}


def _confirm_people(job_id, result, candidates, people, *, automatic):
    """Keep the first track in this job and queue a separate job for each other track."""
    first, *remaining = people
    changes = dict(status="running", candidates=None, **_selected_track(candidates[first], automatic=automatic),
                   message="No selection - using default person" if automatic
                   else f"已选择 {len(people)} 人" if remaining else f"已选择人物 {first + 1}")
    if not remaining:
        with JOBS_LOCK:
            if job_id in CANCEL_REQUESTS:
                raise JobCancelled()
            JOBS[job_id].update(changes)
            _persist_job_locked(JOBS[job_id])
        return

    from flowhmr.utils.runtime_video_processing import VideoProcessor

    with JOBS_LOCK:
        parent = JOBS[job_id]
        filename, params, ext = parent["filename"], parent["params"], parent["ext"]
        reference = job_body_reference(parent)
    directory = job_dir(job_id)
    video = f"{SUB}{ext}"
    canonical = os.path.join(SUB, f"{SUB}_30fps.mp4")
    changes["person_batch"] = job_id
    children = []
    # Prepare complete jobs outside the live queue so a polling client cannot see half a job.
    with tempfile.TemporaryDirectory(prefix=".person-batch-", dir=directory) as staging:
        for i in remaining:
            _raise_if_cancelled(job_id)
            candidate = candidates[i]
            child = _build_job(filename, params.copy(), ext)
            child.update(_selected_track(candidate), person_batch=job_id,
                         source=parent.get('source', 'desktop'),
                         selection_policy=parent.get('selection_policy', 'legacy'),
                         video_size=[result.width, result.height], source_frames=result.total_frames,
                         body_reference=reference)
            child_stage = os.path.join(staging, child["id"])
            sub_dir = os.path.join(child_stage, SUB)
            os.makedirs(sub_dir)
            _link_video(os.path.join(directory, video), os.path.join(child_stage, video))
            _link_video(os.path.join(directory, canonical), os.path.join(child_stage, canonical))
            _copy_person_thumbnail(directory, child_stage, i)
            VideoProcessor.save_detected_track(os.path.join(sub_dir, f"{SUB}_bbox.npz"), result,
                                                candidate["track_index"], job_detection_range(params))
            _write_job(child, child_stage)
            children.append((child, child_stage))

        with JOBS_LOCK:
            if job_id in CANCEL_REQUESTS:
                raise JobCancelled()
            moved = []
            try:
                members_dir = os.path.join(directory, "members")
                os.makedirs(members_dir, exist_ok=True)
                for child, source in children:
                    dest = Path(members_dir, child["id"]).resolve()
                    if Path(ARGS.work_dir).resolve() not in dest.parents or os.path.lexists(dest):
                        raise FileExistsError(f"Cannot publish person job {child['id']}")
                    os.rename(source, dest)
                    moved.append((source, dest))
                updated = {**JOBS[job_id], **changes}
                _persist_job_locked(updated)
            except OSError:
                for source, dest in reversed(moved):
                    os.rename(dest, source)
                raise
            JOBS[job_id] = updated
            for (child, _), (_, dest) in zip(children, moved):
                MEMBER_DIRS[child["id"]] = os.fspath(dest)
                JOBS[child["id"]] = child
                enqueue(child)


def _batch_members_locked(batch):
    primary = JOBS.get(batch)
    if primary is None or primary.get("person_batch") != batch:
        return []
    return [primary, *(j for j in JOBS.values() if j["id"] != batch and j.get("person_batch") == batch)]


def _scene_status_locked(parent, members):
    """Present a batch as one task while keeping the individual artifacts addressable."""
    active = [j for j in members if j["status"] in ACTIVE_STATUSES]
    completed = [j for j in members if j["status"] == "done"]
    view = {**parent, "selected_person": None, "candidates": None,
            "person_count": len(members), "person_done": len(completed),
            "long_motion_frames": max(j.get("long_motion_frames") or 0 for j in members) or None,
            "has_fbx": False, "error": None, "error_code": None, "error_params": {}}
    if active:
        order = ("waiting_user", "running", "cancel_requested", "queued", "uploading")
        current = min(active, key=lambda j: (order.index(j["status"]), j["created"]))
        view.update({k: current.get(k) for k in ("status", "message", "stages", "loading_models", "sam_batch",
                                                 "hamer_report", "camera", "correction")})
        if current.get("selected_person") is not None:
            view["current_person"] = current["selected_person"] + 1
        view["progress"] = int(sum(j["progress"] for j in members) * .95 / len(members))
        view["finished"] = None
    elif parent.get("scene") and parent.get("scene_state") == "ready":
        scene = parent["scene"]
        view.update(status="done", progress=100, has_fbx=fbx_export_available(job_body_reference(parent)),
                    num_frames=scene["num_frames"], track_range=[scene["track_start"], scene["track_end"]],
                    error=None, finished=parent.get("scene_finished") or max(j["finished"] or 0 for j in members))
    elif parent.get("scene_state") in ("queued", "running", "cancel_requested"):
        status = parent["scene_state"]
        view.update(status=status, progress=95, error=None, finished=None,
                    message=parent.get("scene_message", "正在对齐多人站位…"),
                    stages=[dict(name="scene", label="场景对齐", status="running" if status == "running" else "pending",
                                 progress=50 if status == "running" else 0, start=parent.get("scene_started"), end=None)])
    elif completed:
        error = parent.get("scene_error")
        view.update(status="needs_alignment", progress=95, error=error, scene_needs_build=True,
                    error_code="scene_failed" if error else None,
                    error_params={"detail": parent.get("scene_error_detail") or error} if error else {},
                    message=error or "动作已生成，可按原片对齐多人场景。")
    else:
        failures = [j for j in members if j["status"] == "failed"]
        view.update(status="failed" if failures else "cancelled")
        if failures:
            view.update({key: failures[0].get(key) for key in ("error", "error_code", "error_params")})
    return view


def _assemble_scene(batch, *, automatic):
    with JOBS_LOCK:
        members = _batch_members_locked(batch)
        if len(members) < 2 or any(j["status"] in ACTIVE_STATUSES for j in members):
            return
        parent = members[0]
        state = parent.get("scene_state")
        if (automatic and state not in (None, "pending")) or (not automatic and state != "queued"):
            return
        done = [(j.copy(), job_dir(j["id"])) for j in members if j["status"] == "done"]
        if not done:
            return
        parent.update(scene_state="running", scene_message="正在对齐多人站位…", scene_started=time.time(), scene_error=None)
        _persist_job_locked(parent)

    def check():
        with JOBS_LOCK:
            if JOBS[batch].get("scene_state") == "cancel_requested":
                raise JobCancelled()

    def notify(message):
        check()
        update_job(batch, scene_message=message)

    @contextmanager
    def load_scene_sam():
        from flowhmr.utils.runtime_sam_features import build_sam3d_extractor

        notify("正在加载站位估计模型…")
        with closing(build_sam3d_extractor(device, xpu_backbone_device=xpu_backbone)) as extractor:
            yield extractor

    try:
        import torch
        from flowhmr.utils.runtime_group_scene import build_group_scene

        if RUNTIME.cuda_variant() == "cu121":
            _check_cu121_weights(tuple(map(resolve_model, (*SAM3D_FILES, VGGT))))
        device = ENGINE.device if ENGINE is not None else resolve_inference_device(ARGS.device)
        xpu_backbone = device if device.type == "xpu" else None
        if xpu_backbone is not None:
            device = torch.device("cpu")  # VGGT and scene geometry run on the host.
        acquire_sam = ENGINE.scene_sam if ENGINE is not None else load_scene_sam
        scene = build_group_scene(Path(job_dir(batch)) / "scene", done, INSPECTION, acquire_sam, device,
                                  check=check, notify=notify)
        update_job(batch, scene=scene, scene_state="ready", scene_error=None, scene_finished=time.time())
        print(f"  [scene] {batch}: {len(scene['people'])} aligned people, source frames {scene['track_start'] + 1}–{scene['track_end'] + 1}", flush=True)
    except JobCancelled:
        update_job(batch, scene_state="cancelled")
    except Exception as exc:
        update_job(batch, scene_state="failed", scene_error=f"多人场景对齐失败：{exc}", scene_error_detail=str(exc))
        traceback.print_exc()


def run_job(job_id):
    _raise_if_cancelled(job_id)
    with JOBS_LOCK:
        job = JOBS[job_id]
        params = job["params"]
        sam_backend = params.get("sam_backend", "gpu")
        requested_reference = job.get("body_reference")
        if not isinstance(requested_reference, str) or requested_reference not in SMPLH_FILES:
            requested_reference = None
        if params["hamer"] and not hamer_available(ARGS.device):
            if str(ARGS.device).split(":", 1)[0] in ("cpu", "xpu"):
                raise UserMessage(HAMER_ISSUES["hamer_gpu_required"], "hamer_gpu_required")
            raise UserMessage(MISSING_HAMER, "hamer_not_ready")
        loading_models = (ENGINE is None or ENGINE.body_reference != (requested_reference or smplh_reference())
                          or ENGINE.sam_backend != sam_backend)
        job.update(status="running", loading_models=loading_models, started=time.time(),
                   message="首次运行，正在加载推理模型…" if loading_models else "Starting")
        _persist_job_locked(job)

    if params["hamer"] and RUNTIME.cuda_variant() == "cu121":
        _check_cu121_weights(tuple(map(resolve_model, HAMER_FILES)))

    def on_progress(stage, progress, desc):
        idx = STAGE_INDEX.get(getattr(stage, "value", str(stage)))
        if idx is None:
            return
        now = time.time()
        with JOBS_LOCK:
            if job_id in CANCEL_REQUESTS:
                raise JobCancelled()
            job = JOBS[job_id]
            if idx >= len(job["stages"]):
                return
            before = (job["progress"], job["stages"][idx]["progress"], job["stages"][idx]["status"])
            for s in job["stages"][:idx]:  # earlier stages are finished once a later one reports
                if s["status"] != "completed":
                    s.update(status="completed", progress=100, end=now, start=s["start"] or now)
            s = job["stages"][idx]
            s["start"] = s["start"] or now
            s["progress"] = int(max(s["progress"], min(progress, 100)))
            if s["progress"] >= 100:
                s.update(status="completed", end=s["end"] or now)
            else:
                s["status"] = "running"
            job["progress"] = int(sum(x["progress"] for x in job["stages"]) / len(job["stages"]))
            job["message"] = desc
            after = (job["progress"], s["progress"], s["status"])
            if before != after:
                _persist_job_locked(job)

    def select_person(result, candidates, default_idx):
        """Pause for one or more choices; continue this single-person pipeline on the first."""
        slot = {"event": threading.Event(), "people": None}
        try:
            save_person_thumbnails(job_id, candidates)
        except Exception:  # thumbnails are only a UI aid
            traceback.print_exc()
        with JOBS_LOCK:
            if job_id in CANCEL_REQUESTS:
                raise JobCancelled()
            SELECTIONS[job_id] = slot
            job = JOBS[job_id]
            job["stages"][STAGE_INDEX["detection"]].update(
                status="waiting_user", paused_at=time.time(),
            )
            job.update(status="waiting_user", candidates=candidates, default_person=default_idx,
                       video_size=[result.width, result.height],
                       message=f"检测到 {len(candidates)} 人，请勾选要生成的人物")
            _persist_job_locked(job)
        try:
            slot["event"].wait(SELECT_TIMEOUT_S)
        finally:
            with JOBS_LOCK:
                people = slot["people"]
                stage = JOBS[job_id]["stages"][STAGE_INDEX["detection"]]
                stage["paused_seconds"] = stage.get("paused_seconds", 0.0) + max(
                    0.0, time.time() - stage.pop("paused_at"),
                )
                stage["status"] = "running"
                SELECTIONS.pop(job_id, None)
        _raise_if_cancelled(job_id)
        if people is None and JOBS[job_id].get('selection_policy') == 'explicit':
            raise UserMessage('等待人物选择已超时，请重跑任务并重新选择。', 'selection_timeout')
        chosen = people or [default_idx]
        _confirm_people(job_id, result, candidates, chosen, automatic=people is None)
        return chosen[0]

    engine = get_engine(job_id, requested_reference, sam_backend=sam_backend)
    _raise_if_cancelled(job_id)
    engine.pipeline.validation_steps = params["steps"]
    engine.postprocess = None
    if params["postprocess"]:
        engine.postprocess = get_postprocess(
            params["correction_floor_m"], params["correction_preset"], sole_fit_strength=params["sole_fit_strength"],
        )
    sub_dir = engine.process_video(
        input_video(job_id), job_dir(job_id), seed=params["seed"], cfg_scale=params["cfg_scale"],
        max_frames=None if params["auto_frames"] else params["max_frames"],
        start_frame=params["start_frame"], end_frame=params["end_frame"],
        smoothing_frames=params["smoothing_frames"],
        hamer=params["hamer"],
        hamer_stride=params["hamer_stride"],
        vggt_frame_interval=params["vggt_interval"],
        static_camera=params.get("static_camera", False),
        sam_chunk_size=params.get("sam_chunk_size", DEFAULT_PARAMS["sam_chunk_size"]),
        sam_hand_stride=params["sam_hand_stride"],
        person_selector=select_person if params.get("manual_select", True) else None,
        progress_callback=on_progress,
        cancel_check=lambda: _raise_if_cancelled(job_id),
        details_callback=lambda details: update_job(job_id, **details),
    )
    _raise_if_cancelled(job_id)
    final_suffix = "" if params["postprocess"] or params["hamer"] else "_origin"
    result = os.path.join(sub_dir, f"{SUB}_seed{params['seed']}{final_suffix}")
    if not os.path.exists(result + ".npz"):
        raise FileNotFoundError(f"generation output missing: {result}.npz")

    update_job(job_id, message="Building 3D mesh")
    out_root = job_dir(job_id)
    _raise_if_cancelled(job_id)
    meta = export_mesh_sequence(result + ".npz", os.path.join(out_root, "mesh"),
                                cancel_check=lambda: _raise_if_cancelled(job_id))
    _raise_if_cancelled(job_id)
    start = 0
    bbox_npz = os.path.join(sub_dir, f"{SUB}_bbox.npz")
    if os.path.exists(bbox_npz):
        with np.load(bbox_npz, allow_pickle=False) as bbox:
            start = int(bbox["start_end"][0])
    meta.update({"track_start": start, "track_end": start + meta["num_frames"] - 1,
                 "body_reference": engine.body_reference})
    with open(os.path.join(out_root, "mesh", "meta.json"), "w") as f:
        json.dump(meta, f)

    shutil.copy2(result + ".npz", os.path.join(out_root, "motion.npz"))
    has_fbx = os.path.exists(result + ".fbx")
    if has_fbx:
        shutil.copy2(result + ".fbx", os.path.join(out_root, "motion.fbx"))

    camera = camera_summary(out_root, params)
    correction = None
    if engine.postprocess is not None:
        report = engine.postprocess.last_report
        correction = {"floor_m": float(report["floor"]), "automatic": params["correction_floor_m"] is None,
                      "preset": report["preset"],
                      "contacts": int(report["contacts"]), "seconds": float(report["elapsed_seconds"]),
                      "warning": report.get("warning", "")}
        if "sole_fit" in report:
            correction["sole_fit"] = report["sole_fit"]
    now = time.time()
    with JOBS_LOCK:
        if job_id in CANCEL_REQUESTS:
            raise JobCancelled()
        job = JOBS[job_id]
        for s in job["stages"]:
            if s["status"] != "completed":
                s.update(status="completed", progress=100, end=now, start=s["start"] or now)
        job.update(status="done", progress=100, message="Done", finished=now,
                   num_frames=meta["num_frames"], has_fbx=has_fbx, camera=camera, correction=correction)
        _persist_job_locked(job)


# --------------------------------------------------------------------------- person thumbnails
def save_person_thumbnails(job_id, candidates):
    """One crop per candidate (at the frame where its box is largest), for the selection list."""
    import cv2
    from flowhmr.utils.runtime_image_io import write_image
    video = os.path.join(job_dir(job_id), SUB, f"{SUB}_30fps.mp4")
    out_dir = os.path.join(job_dir(job_id), "persons")
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    try:
        for c in candidates:
            boxes = np.asarray(c["boxes"], dtype=np.float64)
            k = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(c["frames"][k]))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = boxes[k]
            px, py = 0.15 * (x2 - x1), 0.08 * (y2 - y1)
            x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
            x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            scale = 192.0 / crop.shape[0]
            crop = cv2.resize(crop, (max(1, int(round(crop.shape[1] * scale))), 192),
                              interpolation=cv2.INTER_AREA)
            write_image(os.path.join(out_dir, f"{c['index']}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    finally:
        cap.release()


# --------------------------------------------------------------------------- mesh export
def export_mesh_sequence(motion_npz, out_dir, cancel_check=lambda: None):
    """Run SMPL-H forward on the generated motion and write a float16 vertex stream.

    No ground alignment: Y=0 is the floor and the model's root height is shown
    as-is (open post-processing in the page if feet should contact the floor).
    """
    import torch

    os.makedirs(out_dir, exist_ok=True)
    data = np.load(motion_npz, allow_pickle=True)
    poses = torch.from_numpy(np.asarray(data["poses"], dtype=np.float32))
    trans = torch.from_numpy(np.asarray(data["trans"], dtype=np.float32))
    betas = torch.from_numpy(np.asarray(data["betas"], dtype=np.float32)).reshape(1, -1)
    T = poses.shape[0]
    poses = poses.reshape(T, -1, 3)

    mesh_model = ENGINE.smpl_mesh
    dev = mesh_model.v_template.device
    betas = betas[:, :mesh_model.shapedirs.shape[-1]].to(dev)

    faces = np.asarray(mesh_model.faces, dtype=np.uint32)
    minimum = np.full(3, np.inf, np.float32)
    maximum = np.full(3, -np.inf, np.float32)
    temporary = os.path.join(out_dir, "vertices.bin.part")
    try:
        with torch.no_grad(), open(temporary, "wb") as stream:
            for s in range(0, T, 64):
                cancel_check()
                e = min(T, s + 64)
                out = mesh_model({"poses": poses[s:e].to(dev), "shapes": betas,
                                  "trans": trans[s:e].to(dev)})
                verts = out["vertices"].float().cpu().numpy()
                flat = verts.reshape(-1, 3)
                np.minimum(minimum, flat.min(0), out=minimum)
                np.maximum(maximum, flat.max(0), out=maximum)
                verts.astype(np.float16).tofile(stream)
        cancel_check()
        os.replace(temporary, os.path.join(out_dir, "vertices.bin"))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    faces.tofile(os.path.join(out_dir, "faces.bin"))

    root = trans.numpy().copy()
    return {
        "num_frames": int(T), "num_vertices": int(mesh_model.v_template.shape[-2]), "num_faces": int(faces.shape[0]),
        "fps": 30, "vertex_dtype": "float16", "face_dtype": "uint32", "up_axis": "y",
        "bbox_min": minimum.tolist(), "bbox_max": maximum.tolist(),
        "root_trajectory": root[:, [0, 2]].round(4).tolist(),
    }


# --------------------------------------------------------------------------- routes
def get_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            abort(404)
        return copy.deepcopy(job)


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    # A 304 response carries no Content-Type, so a client that once cached these
    # files under a wrong MIME type keeps reusing that type and never recovers,
    # even after the server is fixed. Always answer with the full response so the
    # MIME type we send is authoritative.
    return send_from_directory(os.path.join(WEB_DIR, "static"), path, conditional=False)


def local_model_request():
    try:
        hostname = urlsplit("//" + request.host).hostname
    except ValueError:
        return False
    if request.remote_addr not in ("127.0.0.1", "::1") or hostname not in ("localhost", "127.0.0.1", "::1"):
        return False
    origin = request.headers.get("Origin")
    return origin is None or origin == request.host_url.rstrip("/")


def api_error(error_code, message, status, **params):
    return jsonify(error=message, error_code=error_code, error_params=params), status


@app.get('/api/mcp/parameters')
def api_mcp_parameters():
    require_mcp_service()
    return jsonify(parameters=MCPParameters.model_json_schema(), defaults=parameter_defaults(default_params()),
                   export_options=MCPExportOptions.model_json_schema(),
                   frame_timeline='Zero-based normalized 30 FPS video; end_frame is inclusive.',
                   parameter_application='Import uses defaults; rerun inherits omitted parameters and creates a new job.')


def submission_metadata(values, params):
    if values.get('source') != 'mcp':
        return {'source': 'desktop', 'selection_policy': 'legacy'}
    require_mcp_service(max_bytes=None)
    policy = values.get('selection_policy', 'explicit')
    if policy not in ('explicit', 'automatic'):
        raise UserMessage('Invalid selection policy.', 'param_selection_policy')
    params['manual_select'] = policy == 'explicit'
    return {'source': 'mcp', 'selection_policy': policy}


@app.route('/api/mcp', methods=['GET', 'POST'])
def api_mcp():
    require_desktop_session()
    try:
        if request.method == 'GET':
            return jsonify(MCP_SERVICE.snapshot())
        body = request.get_json()
        if not isinstance(body, dict):
            raise ValueError('Expected a JSON object.')
        action = body.get('action')
        if set(body) - {'action', 'port', 'auth_enabled', 'auto_start'}:
            raise ValueError('Unknown MCP configuration field.')
        if action == 'start':
            return jsonify(MCP_SERVICE.start(body.get('port'), f'http://127.0.0.1:{ARGS.port}'))
        if action == 'stop':
            MCP_SERVICE.stop()
            return jsonify(MCP_SERVICE.snapshot())
        if action == 'reset_token':
            return jsonify(MCP_SERVICE.reset_token())
        if action == 'configure_auth':
            return jsonify(MCP_SERVICE.configure_auth(body.get('auth_enabled')))
        if action == 'configure_auto_start':
            return jsonify(MCP_SERVICE.configure_auto_start(body.get('auto_start')))
        if action == 'test':
            return jsonify(MCP_SERVICE.test_connection())
        raise ValueError('Unknown MCP action.')
    except Exception as error:
        app.logger.exception('MCP management failed')
        return api_error('mcp_failed', str(error), 400)


CHARACTER_FILES = CharacterFiles()
CHARACTER_LIBRARY = CharacterLibrary(MODEL_ROOT / "characters")


def mcp_character_actor(job_id, person_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            abort(404)
        members = _batch_members_locked(job_id)
        if len(members) > 1:
            if type(person_id) is not int:
                raise ValueError('Multi-person scenes require person_id from flowhmr_get_job')
            matches = [member for member in members if member.get('selected_person') == person_id - 1]
            if not matches:
                raise ValueError('Unknown scene person_id')
            actor = matches[0]
            slot = str(person_id)
        else:
            actor, slot = job, 'single'
            if person_id is not None and (type(person_id) is not int or person_id != single_result_person_id(job)):
                raise ValueError('Unknown result person_id')
        if actor['status'] != 'done':
            raise ValueError('Character settings require a completed actor result')
        return actor['id'], slot


@app.get('/api/mcp/jobs/<job_id>/keyframes')
def api_mcp_motion_keyframes(job_id):
    require_mcp_service()
    try:
        person = request.args.get('person', type=int)
        start = request.args.get('start_frame', default=0, type=int)
        count = request.args.get('count', default=1, type=int)
        if (('person' in request.args and person is None) or start is None or count is None
                or start < 0 or not 1 <= count <= 120):
            raise ValueError('Invalid person, start_frame or count')
        actor, _ = mcp_character_actor(job_id, person)
        from flowhmr.mcp.joint_names import SMPLH_JOINT_NAMES_52
        with np.load(os.path.join(job_dir(actor), 'motion.npz'), allow_pickle=False) as motion:
            poses = np.asarray(motion['poses'])
            trans = np.asarray(motion['trans'])
            if poses.ndim == 2 and poses.shape[1] == 156:
                poses = poses.reshape(-1, 52, 3)
            if poses.ndim != 3 or poses.shape[1:] != (52, 3) or trans.shape != (len(poses), 3):
                raise ValueError('Unsupported source motion shape')
            if start >= len(poses) or start + count > len(poses):
                raise ValueError('Requested frame range exceeds motion length')
            fps = float(np.asarray(motion['mocap_framerate']).item())
            frames = [{'frame': i, 'time_seconds': i / fps, 'root_translation': trans[i].tolist(),
                       'rotations_axis_angle': poses[i].tolist()} for i in range(start, start + count)]
        return jsonify(job_id=job_id, actor_id=actor, person_id=person, frame_count=len(poses), fps=fps,
                       skeleton='SMPL-H', joints=[{'index': i, 'name': name} for i, name in enumerate(SMPLH_JOINT_NAMES_52)],
                       rotation_units='radians, axis-angle; local joint rotations', translation_units='metres',
                       keyframe_type='dense source motion samples', frames=frames)
    except (ValueError, KeyError, OSError) as error:
        return api_error('motion_keyframes_invalid', str(error), 400)


@app.post('/api/mcp/characters/import')
def api_mcp_import_character():
    require_mcp_service()
    value = request.get_json()
    if not isinstance(value, dict) or set(value) - {'file', 'name'}:
        abort(400)
    try:
        return jsonify(import_character(CHARACTER_LIBRARY, CHARACTER_FILES, value.get('file'), value.get('name')))
    except (OSError, ValueError, RuntimeError) as error:
        return api_error('character_import_failed', str(error), 400)


@app.route('/api/mcp/jobs/<job_id>/character', methods=['GET', 'POST', 'PATCH'])
def api_mcp_character(job_id):
    require_mcp_service()
    try:
        value = request.get_json() if request.method != 'GET' else {}
        if not isinstance(value, dict):
            raise ValueError('Expected an object')
        person = request.args.get('person', type=int) if request.method == 'GET' else value.get('person_id')
        if request.method == 'GET' and 'person' in request.args and person is None:
            raise ValueError('person must be an integer')
        actor, slot = mcp_character_actor(job_id, person)
        catalog = CHARACTER_LIBRARY.snapshot([actor])
        if request.method == 'POST':
            if set(value) - {'person_id', 'character_id'} or 'character_id' not in value:
                raise ValueError('Expected character_id and optional person_id')
            identifier = value['character_id']
            models = {item['id']: item for item in catalog['models']}
            if identifier is not None and identifier not in models:
                raise ValueError('Character is not in the library')
            saved = catalog['actors'].get(actor, {}).get('poses', {}).get(identifier)
            pose = saved if saved is not None else catalog['presets'].get(identifier)
            if pose is None:
                pose = empty_pose(models[identifier]['kind']) if identifier else None
            CHARACTER_LIBRARY.set_actor(actor, slot, identifier, pose, select=True)
        elif request.method == 'PATCH':
            if set(value) - {'person_id', 'character_id', 'patch'}:
                raise ValueError('Unknown pose setting')
            patch = PosePatch.model_validate(value.get('patch'))
            if patch.ik_mode == 'ik':
                kind = next((item['kind'] for item in catalog['models'] if item['id'] == value.get('character_id')), None)
                if kind != 'pmx':
                    raise ValueError('Native IK mode requires a compatible PMX character')
            CHARACTER_LIBRARY.update_pose(actor, slot, value.get('character_id'), patch.patch)
        catalog = CHARACTER_LIBRARY.snapshot([actor])
        record = catalog['actors'].get(actor)
        inherited = record is None
        if inherited:
            default = catalog['defaults'].get(slot, {})
            record = {'id': default.get('id'), 'poses': {default.get('id'): default.get('offsets')}}
        identifier = record.get('id')
        return jsonify(actor_id=actor, person_id=person, character_id=identifier,
            assignment_source='default' if inherited and identifier else 'actor' if not inherited else 'none',
            pose=record.get('poses', {}).get(identifier), bones=BONES,
            patch_schema=PosePatch.model_json_schema(),
            scope='Preview offsets only; original motion and generic MCP exports remain unchanged.',
            rotation_units='degrees; XYZ in the retarget rig bind axes', translation_units='metres',
            finger_controls='22-36 are shared hand controls, not raw target bone indices',
            mirror='Editor preference only; provide both body joint rotations to edit both sides.')
    except (OSError, ValueError, RuntimeError) as error:
        return api_error('character_settings_failed', str(error), 400)


def check_character_library_request(max_bytes=16384):
    """Bound the JSON body, but let callers opt out (thumbnail uploads pass
    None and enforce their own, larger limit)."""
    if not local_model_request() or request.headers.get("Sec-Fetch-Site") == "cross-site":
        abort(403)
    if max_bytes is not None and request.content_length and request.content_length > max_bytes:
        abort(413)


def require_mcp_service(max_bytes=16384):
    if not local_model_request() or request.headers.get('Sec-Fetch-Site') == 'cross-site':
        abort(403)
    supplied = request.headers.get(SERVICE_KEY_HEADER, '')
    if not supplied.isascii() or not hmac.compare_digest(supplied, MCP_SERVICE.service_key):
        abort(403)
    if max_bytes is not None and request.content_length and request.content_length > max_bytes:
        abort(413)


@app.route("/api/character-library", methods=["GET", "POST"])
def api_character_library():
    check_character_library_request()
    try:
        if request.method in ("GET", "HEAD"):
            return jsonify(CHARACTER_LIBRARY.snapshot(request.args.getlist("actor")))
        value = request.get_json(silent=True)
        ticket = value.get("ticket") if isinstance(value, dict) else None
        if not isinstance(ticket, str) or not 1 <= len(ticket) <= 64:
            abort(400)
        return jsonify(CHARACTER_LIBRARY.save(CHARACTER_FILES, ticket, value.get("name")))
    except (OSError, ValueError, RuntimeError):
        if request.method == "POST":
            return api_error("character_library_save", "无法将模型和贴图存入角色库。", 400)
        return api_error("character_library_read", "无法读取本地角色库。", 500)


@app.post("/api/character-library/<identifier>/thumbnail")
def api_set_character_thumbnail(identifier):
    # Token uploads are far larger than a JSON call, so the generic body bound is
    # lifted here and a 1 MiB ceiling applied instead.
    check_character_library_request(max_bytes=None)
    if (request.content_length or 0) > 1024 * 1024:
        abort(413)
    try:
        CHARACTER_LIBRARY.write_thumbnail(identifier, request.get_data(cache=False))
    except FileNotFoundError:
        return api_error("character_library_missing", "此角色的文件无法读取，请重新导入。", 404)
    except (OSError, ValueError):
        return api_error("character_library_thumbnail", "无法保存角色缩略图。", 400)
    return jsonify(ok=True)


@app.post("/api/character-library/<identifier>/open")
def api_open_character_library_entry(identifier):
    check_character_library_request()
    try:
        return jsonify(CHARACTER_LIBRARY.open(identifier, CHARACTER_FILES))
    except (OSError, ValueError, RuntimeError):
        return api_error("character_library_missing", "此角色的文件无法读取，请重新导入。", 404)


@app.delete("/api/character-library/<identifier>")
def api_remove_character_library_entry(identifier):
    check_character_library_request()
    try:
        cleaned = CHARACTER_LIBRARY.remove(identifier, CHARACTER_FILES)
    except FileNotFoundError:
        return api_error("character_library_missing", "此角色的文件无法读取，请重新导入。", 404)
    except (OSError, ValueError, RuntimeError):
        app.logger.exception("Could not remove the saved character")
        return api_error("character_library_delete", "无法删除角色，请检查文件是否被占用。", 500)
    return jsonify(catalog=CHARACTER_LIBRARY.snapshot(request.args.getlist("actor")), cleanup_pending=not cleaned)


@app.put("/api/character-library/actors/<actor_id>")
def api_set_actor_character(actor_id):
    check_character_library_request()
    job = get_job(actor_id)
    if job["status"] != "done":
        abort(409)
    value = request.get_json(silent=True)
    if not isinstance(value, dict) or "id" not in value or type(value.get("select")) is not bool:
        abort(400)
    number = value.get("person")
    if number is not None and (type(number) is not int or not 1 <= number <= 2**31 - 1
                               or number - 1 != job.get("selected_person")):
        abort(400)
    identifier = value["id"]
    if identifier is None and not value["select"]:
        abort(400)
    try:
        CHARACTER_LIBRARY.set_actor(actor_id, str(number) if number is not None else "single",
                                    identifier, value.get("offsets"), select=value["select"])
    except FileNotFoundError:
        return api_error("character_library_missing", "此角色的文件无法读取，请重新导入。", 404)
    except (OSError, ValueError, RuntimeError):
        return api_error("character_library_settings", "无法保存角色设置。", 500)
    return jsonify(ok=True)


def require_desktop_session():
    key = os.environ.get(CHARACTER_KEY_ENV, "")
    supplied = request.headers.get("X-FlowHMR-Desktop-Key", "")
    if (not local_model_request() or request.headers.get('Sec-Fetch-Site') == 'cross-site'
            or not key or not key.isascii() or not supplied.isascii()
            or not hmac.compare_digest(key, supplied)):
        abort(403)


@app.get("/api/desktop/session")
def api_desktop_session():
    require_desktop_session()
    return "", 204, {"Cache-Control": "no-store"}


@app.post("/api/desktop/character-file")
def api_desktop_character_file():
    require_desktop_session()
    if request.content_length and request.content_length > 16384:
        abort(413)
    value = request.get_json(silent=True)
    try:
        return jsonify(CHARACTER_FILES.register(value.get("file") if isinstance(value, dict) else None))
    except (ValueError, OSError, RuntimeError):
        return api_error("character_file", "无法打开所选角色文件。", 400)


@app.get("/api/character-resource/<ticket>")
def api_character_resource(ticket):
    if not local_model_request() or request.headers.get("Sec-Fetch-Site") == "cross-site":
        abort(403)
    texture = request.args.get("texture")
    path = CHARACTER_FILES.resource(ticket, texture)
    if path is None:
        abort(404)
    kind = mimetypes.guess_type(path.name)[0] if texture is not None else None
    response = send_file(path, mimetype=kind or "application/octet-stream", conditional=True)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = "default-src 'none'; sandbox"
    return response


def jobs_active():
    with JOBS_LOCK:
        return any(job["status"] in ACTIVE_STATUSES or job.get("scene_state") in ACTIVE_STATUSES
                   for job in JOBS.values())


def model_download_active():
    return MODEL_DOWNLOADS.snapshot()["status"] in ACTIVE_DOWNLOAD


@app.route("/api/info")
def api_info():
    config = getattr(ARGS, "model_cfg", None)
    inventory = model_inventory(ARGS.device, ARGS.ckpt, config)
    rocm_profile = runtime_rocm.PROFILES.get(RUNTIME.active)
    if local_model_request():
        plan = missing_downloads(ARGS.device, ARGS.ckpt, config, include_hamer=True)
        inventory["download_sizes"] = {group: sum(asset.size for asset in assets) for group, assets in plan.items()}
        inventory["hamer_download_bytes"] = inventory["download_sizes"].get("hamer", 0)
        inventory["downloadable"] = list(plan)
        inventory["download_bytes"] = sum(inventory["download_sizes"].values())
        inventory["download"] = MODEL_DOWNLOADS.snapshot()
        inventory["import"] = MODEL_IMPORTER.status()
    return jsonify({
        "model": flowhmr_paths(ARGS.ckpt, config)[0].name, "device": ARGS.device,
        "models_loaded": ENGINE is not None,
        "model_files": inventory,
        "runtime": RUNTIME.summary(),
        "steps": DEFAULT_PARAMS["steps"],
        "fps": 30, "defaults": default_params(),
        "stages": [{"name": n, "label": l, "detail": d} for n, l, d in ALL_STAGES],
        "max_upload_mb": app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
        "postprocess_available": native_correction_available(),
        "hamer_available": any(item["id"] == "hamer" and item["status"] == "ready" for item in inventory["items"]),
        "body_variants": available_variants(),
        "body_reference": smplh_reference(),
        "sam_manual_default": getattr(ARGS, "sam_chunk_default", 16),
        "camera_on_cpu_for_amd_igpu": bool(rocm_profile and rocm_profile.integrated),
        "sam_hybrid_available": str(ARGS.device).split(":", 1)[0] == "cuda" and runtime_rocm.sam_hybrid_candidate(RUNTIME.active),
    })


@app.get("/api/runtime")
def api_runtime():
    if not local_model_request():
        return api_error("runtime_local_only", "请在本机 FlowHMR 窗口管理运行环境。", 403)
    return jsonify(RUNTIME.details())


@app.post("/api/runtime/download")
def api_runtime_download():
    if not local_model_request():
        return api_error("runtime_local_only", "请在本机 FlowHMR 窗口管理运行环境。", 403)
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("version"), str):
        return api_error("runtime_version_required", "请选择运行环境版本。", 400)
    with RUNTIME_SWITCH_LOCK:
        if RUNTIME.summary()["pending_restart"]:
            return api_error("runtime_restart_required", "请先重启运行环境，完成切换。", 409)
        try:
            return jsonify(RUNTIME.start(data["version"]))
        except DownloadError as error:
            return api_error(error.code, str(error), 400, **error.params)


@app.post("/api/runtime/download/stop")
def api_runtime_download_stop():
    if not local_model_request():
        return api_error("runtime_local_only", "请在本机 FlowHMR 窗口管理运行环境。", 403)
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("id"), str):
        return api_error("download_task_invalid", "无效的下载任务。", 400)
    return jsonify(RUNTIME.stop(data["id"]))


@app.post("/api/runtime/select")
def api_runtime_select():
    if not local_model_request():
        return api_error("runtime_local_only", "请在本机 FlowHMR 窗口管理运行环境。", 403)
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("version"), str):
        return api_error("runtime_version_required", "请选择运行环境版本。", 400)
    with RUNTIME_SWITCH_LOCK:
        if (jobs_active() or model_download_active() or RUNTIME.snapshot()["status"] in ACTIVE_DOWNLOAD
                or MODEL_IMPORTER.active or MODEL_VERIFY_LOCK.locked()):
            return api_error("runtime_busy", "有生成、模型操作或下载任务在运行，完成后再切换。", 409)
        try:
            return jsonify(RUNTIME.select(data["version"]))
        except DownloadError as error:
            return api_error(error.code, str(error), 400, **error.params)
        except OSError:
            app.logger.exception("Could not save runtime selection")
            return api_error("runtime_save_failed", "无法保存运行环境选择，请检查目录写入权限。", 500)


def requires_runtime(view):
    @wraps(view)
    def guarded(*args, **kwargs):
        with RUNTIME_SWITCH_LOCK:
            if MODEL_IMPORTER.active:
                return api_error("model_import_busy", "正在导入模型，请等导入完成。", 409)
            state = RUNTIME.summary()
            if not state["ready"]:
                message = ("请先重启运行环境，完成切换。" if state["pending_restart"] else
                           "请在右上角「运行环境」下载并启用推理环境。")
                return jsonify(error=message, code="runtime_required",
                               error_code="runtime_restart_required" if state["pending_restart"] else "runtime_required"), 409
            return view(*args, **kwargs)
    return guarded


@app.route("/api/models/download", methods=["GET", "POST"])
def api_model_download():
    if not local_model_request():
        return api_error("models_local_only", "请在本机 FlowHMR 窗口管理模型。", 403)
    if request.method == "GET":
        return jsonify(MODEL_DOWNLOADS.snapshot())
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or set(body) - {"group"}:
        return api_error("download_invalid", "无效的下载请求。", 400)
    group = body.get("group") if body else None
    if body and (not isinstance(group, str) or group not in (*PUBLIC_ASSETS, "all", "hamer")):
        return api_error("download_invalid", "无效的下载请求。", 400)
    with RUNTIME_SWITCH_LOCK:
        if RUNTIME.summary()["pending_restart"]:
            return api_error("runtime_restart_required", "请先重启运行环境，完成切换。", 409)
        if MODEL_IMPORTER.active or MODEL_VERIFY_LOCK.locked():
            return api_error("model_operation_busy", "正在导入或校验模型，请稍后再下载。", 409)
        config = getattr(ARGS, "model_cfg", None)
        plan = missing_downloads(ARGS.device, ARGS.ckpt, config, include_hamer=group in ("all", "hamer"))
        if group is not None and group != "all":
            plan = {group: plan[group]} if group in plan else {}
        return jsonify(MODEL_DOWNLOADS.start(plan, scope=group))


@app.route("/api/models/download/stop", methods=["POST"])
def api_stop_model_download():
    if not local_model_request():
        return api_error("models_local_only", "请在本机 FlowHMR 窗口管理模型。", 403)
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("id"), str):
        return api_error("download_task_invalid", "无效的下载任务。", 400)
    return jsonify(MODEL_DOWNLOADS.stop(data["id"]))


@app.route("/api/models/verify", methods=["POST"])
def api_verify_models():
    """Check the models this installation actually uses."""
    if not local_model_request():
        return api_error("models_local_only", "请在本机 FlowHMR 窗口管理模型。", 403)
    with RUNTIME_SWITCH_LOCK:
        if MODEL_IMPORTER.active or model_download_active() or not MODEL_VERIFY_LOCK.acquire(blocking=False):
            return api_error("model_operation_busy", "正在下载、导入或校验模型，请稍后再校验。", 409)
    try:
        required = core_model_paths(ARGS.ckpt, getattr(ARGS, "model_cfg", None))
        return jsonify(model_integrity.summary(deep=True, required_paths=required))
    finally:
        MODEL_VERIFY_LOCK.release()


@app.route("/api/models/import", methods=["GET", "POST"])
def api_model_import():
    if not local_model_request():
        return api_error("models_local_only", "请在本机 FlowHMR 窗口管理模型。", 403)
    if request.method == "GET":
        return jsonify(MODEL_IMPORTER.status())
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or set(body) - {"path", "kind"} or not isinstance(body.get("path"), str):
        return api_error("invalid_source", "请选择支持的模型压缩包或已解压的文件夹。", 400)
    kind = body.get("kind", "folder")
    if kind not in ("folder", "archive"):
        return api_error("invalid_source", "请选择支持的模型压缩包或已解压的文件夹。", 400)
    with RUNTIME_SWITCH_LOCK:
        if RUNTIME.summary()["pending_restart"]:
            return api_error("runtime_restart_required", "请先重启运行环境，完成切换。", 409)
        if jobs_active():
            return api_error("model_import_jobs_active", "请先等待当前生成结束或停止任务，再导入模型。", 409)
        if MODEL_IMPORTER.active or model_download_active() or MODEL_VERIFY_LOCK.locked():
            return api_error("model_operation_busy", "正在下载、导入或校验模型，请稍后再导入。", 409)
        try:
            return jsonify(MODEL_IMPORTER.start(body["path"], kind=kind)), 202
        except ModelImportError as error:
            return api_error(error.code, "无法打开选中的模型文件或文件夹。", 400)


@app.route("/api/upload", methods=["POST"])
@requires_runtime
def api_upload():
    f = request.files.get("video")
    if f is None or not f.filename:
        return api_error("video_required", "请选择一个视频文件。", 400)
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in VIDEO_EXTS:
        formats = ", ".join(sorted(VIDEO_EXTS))
        return api_error("video_format", f"unsupported format {ext}; use {formats}", 400, ext=ext, formats=formats)
    try:
        params = parse_params(request.form)
        metadata = submission_metadata(request.form, params)
    except UserMessage as e:
        return api_error(e.code, str(e), 400, **e.params)
    job = new_job(f.filename, params, ext, **metadata)
    f.save(input_video(job["id"]))
    enqueue(job)
    return jsonify(job_id=job['id'], status='queued', params=params, selection_policy=job['selection_policy'])


@app.route("/api/jobs/<job_id>/rerun", methods=["POST"])
@requires_runtime
def api_rerun(job_id):
    """New job on the same video; reuse caches compatible with its camera and range."""
    with JOBS_LOCK:
        src = JOBS.get(job_id)
        if src is None:
            abort(404)
        members = _batch_members_locked(job_id)
        is_scene = len(members) > 1
        src = copy.deepcopy(_scene_status_locked(src, members) if is_scene else src)
    if src["status"] in ACTIVE_STATUSES:
        return api_error("job_active", "任务仍在运行。", 409)
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return api_error("param_invalid", "生成参数无效。", 400)
    if is_scene:
        body = {**body, "manual_select": True, "reselect_person": True}
    try:
        params = parse_params(body, base=src["params"])
        metadata = submission_metadata(body, params)
    except UserMessage as e:
        return api_error(e.code, str(e), 400, **e.params)
    job = new_job(src["filename"], params, src["ext"], **metadata)
    _link_video(input_video(job_id), input_video(job["id"]))
    old_thumbnail = os.path.join(job_dir(job_id), "thumbnail.jpg")
    if os.path.isfile(old_thumbnail):
        shutil.copy2(old_thumbnail, os.path.join(job_dir(job["id"]), "thumbnail.jpg"))
    src_sub = os.path.join(job_dir(job_id), SUB)
    if src["status"] in ("done", "failed", "cancelled") and os.path.isdir(src_sub):
        dst_sub = os.path.join(job_dir(job["id"]), SUB)
        os.makedirs(dst_sub, exist_ok=True)
        skip = (f"{SUB}_seed",)  # generated motion is recomputed
        reselect = _to_bool(body.get("reselect_person"), False)
        camera_changed = params["static_camera"] != src["params"].get("static_camera", False)
        scan_range = job_detection_range(params)
        bbox_name = f"{SUB}_bbox.npz"
        detection_changed = (reselect or job_detection_range(src["params"]) != scan_range or
                             not cached_detection_matches(os.path.join(src_sub, bbox_name), scan_range))
        if detection_changed:
            skip += (bbox_name,)
        if detection_changed or camera_changed:
            # SAM conditioning also depends on camera intrinsics, not just frame count.
            skip += (f"{SUB}_sam3d_feat.pt", f"{SUB}_vggt_camera.npz",
                     f"{SUB}_sam3d_feat_static.pt", f"{SUB}_static_camera.npz")
        elif not params["static_camera"] and params["vggt_interval"] != src["params"]["vggt_interval"]:
            skip += (f"{SUB}_sam3d_feat.pt", f"{SUB}_vggt_camera.npz")
        if any(params[key] != src["params"].get(key, DEFAULT_PARAMS[key])
               for key in ("sam_chunk_size", "sam_hand_stride", "sam_backend")):
            skip += (f"{SUB}_sam3d_feat.pt", f"{SUB}_sam3d_feat_static.pt")
        previous_sam = (src.get("sam_batch") or {}).get("backend", "gpu")
        device_type = str(ARGS.device).split(":", 1)[0]
        next_sam = device_type if device_type in ("cpu", "xpu") else params["sam_backend"]
        if previous_sam != next_sam:
            skip += (f"{SUB}_sam3d_feat.pt", f"{SUB}_sam3d_feat_static.pt")
        for name in os.listdir(src_sub):
            path = os.path.join(src_sub, name)
            if not name.startswith(skip) and os.path.isfile(path):
                if name == f"{SUB}_30fps.mp4":
                    _link_video(path, os.path.join(dst_sub, name))
                else:
                    shutil.copy2(path, dst_sub)
        person = src.get("selected_person")
        if person is not None and not reselect and os.path.isfile(os.path.join(dst_sub, f"{SUB}_bbox.npz")):
            _copy_person_thumbnail(job_dir(job_id), job_dir(job["id"]), person)
            update_job(job["id"], selected_person=person, person_track=src.get("person_track"),
                       selection_auto=src.get("selection_auto", False))
    enqueue(job)
    return jsonify(job_id=job['id'], status='queued', params=params, selection_policy=job['selection_policy'])


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def api_cancel_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            abort(404)
        members = _batch_members_locked(job_id)
        if not members:
            members = [job]
        active = [j for j in members if j["status"] in ("queued", "running", "waiting_user")]
        if not active and job.get("scene_state") not in ("queued", "running"):
            return api_error("job_inactive", "任务已经结束。", 409)
        if len(members) > 1:
            job["scene_state"] = "cancel_requested" if job.get("scene_state") == "running" else "cancelled"
            _persist_job_locked(job)
        for member in active:
            mid = member["id"]
            CANCEL_REQUESTS.add(mid)
            member.update(status="cancel_requested", message="Stop requested; stopping at the next safe step")
            _persist_job_locked(member)
            slot = SELECTIONS.get(mid)
            if slot is not None:
                slot["event"].set()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/jobs/<job_id>/select", methods=["POST"])
def api_select(job_id):
    """Resume with {"people": [candidate indices]}; older clients may send one person."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return api_error("people_required", "请提交所选人物。", 400)
    people = body.get("people", [body.get("person")])
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            abort(404)
        slot = SELECTIONS.get(job_id)
        if slot is None or slot["event"].is_set() or job["status"] != "waiting_user":
            return api_error("person_choice_closed", "此任务已不在等待选人。", 409)
        n = len(job["candidates"] or [])
        if not isinstance(people, list) or not people or any(type(i) is not int or not 0 <= i < n for i in people):
            return api_error("people_required", "请至少勾选一个已检测到的人物。", 400)
        people = list(dict.fromkeys(people))
        slot["people"] = people
        slot["event"].set()
    return jsonify(ok=True, person=people[0], people=people)


@app.route("/api/jobs/<job_id>/scene", methods=["POST"])
@requires_runtime
def api_prepare_scene(job_id):
    with JOBS_LOCK:
        members = _batch_members_locked(job_id)
        if len(members) < 2:
            return api_error("scene_missing", "此任务没有多人场景。", 404)
        parent = members[0]
        if not _scene_status_locked(parent, members).get("scene_needs_build"):
            return api_error("scene_active", "场景已存在或正在处理。", 409)
        parent.update(scene_state="queued", scene_message="等待对齐多人站位…", scene_error=None)
        _persist_job_locked(parent)
        JOB_QUEUE.put(("scene", job_id))
    return jsonify(ok=True)


@app.route("/api/jobs/<job_id>")
def api_job(job_id):
    with JOBS_LOCK:
        row = JOBS.get(job_id)
        if row is None:
            abort(404)
        members = _batch_members_locked(job_id)
        job = copy.deepcopy(_scene_status_locked(row, members) if len(members) > 1 else row)
    if job["status"] == "done":
        scene = job.get("person_batch") == job_id
        motion = os.path.join(job_dir(job_id), "scene" if scene else "", "motion.npz")
        job["has_vmd"] = os.path.isfile(motion)
        job["has_vrma"] = job["has_vmd"]
        job["has_fbx"] = job["has_vmd"] and fbx_export_available(job_body_reference(job))
    canonical = os.path.join(job_dir(job_id), SUB, f"{SUB}_30fps.mp4")
    job["has_preview_video"] = os.path.isfile(canonical)
    if not job.get("source_frames") and job["status"] in ("done", "cancelled", "failed"):
        if job["has_preview_video"]:
            import cv2

            capture = cv2.VideoCapture(canonical)
            try:
                frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                count = int(frames) if math.isfinite(frames) else 0
                if count > 0:
                    job["source_frames"] = count
                    with JOBS_LOCK:
                        if job_id in JOBS and not JOBS[job_id].get("source_frames"):
                            JOBS[job_id]["source_frames"] = count
                            _persist_job_locked(JOBS[job_id])
            except (OSError, ValueError, cv2.error):
                pass
            finally:
                capture.release()
    with JOBS_LOCK:
        job["queue_position"] = sum(1 for j in JOBS.values()
                                    if j["status"] == "queued" and j["created"] < job["created"])
    job["now"] = time.time()
    job["models_loaded"] = ENGINE is not None
    return jsonify(job)


def history_key(row):
    return float(row["created"]), row["id"]


def encode_history_cursor(key):
    return base64.urlsafe_b64encode(json.dumps(key, separators=(",", ":")).encode("utf-8")).decode("ascii").rstrip("=")


def decode_history_cursor(token):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", token):
        raise ValueError("Invalid history cursor")
    payload = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True)
    created, job_id = json.loads(payload)
    if (type(created) not in (int, float) or not math.isfinite(created)
            or not isinstance(job_id, str) or not 0 < len(job_id) <= 64):
        raise ValueError("Invalid history cursor")
    return created, job_id


@app.route("/api/jobs")
def api_jobs():
    try:
        limit = int(request.args.get("limit", "20"))
        before_token, through_token = request.args.get("before"), request.args.get("through")
        if not 1 <= limit <= 100 or (before_token is not None and through_token is not None):
            raise ValueError("Invalid page")
        before = decode_history_cursor(before_token) if before_token is not None else None
        through = decode_history_cursor(through_token) if through_token is not None else None
    except (ValueError, TypeError, binascii.Error, OverflowError):
        return api_error("history_invalid_page", "历史记录分页无效，请刷新后重试。", 400)

    keys = ("id", "filename", "status", "progress", "created", "finished", "num_frames", "params", "recovered", "source", "selection_policy",
            "selected_person", "person_count", "person_done", "message", "loading_models", "scene_state", "source_frames", "stages",
            "sam_batch", "camera")
    with JOBS_LOCK:
        groups = defaultdict(list)
        for row in JOBS.values():
            if row.get("person_batch"):
                groups[row["person_batch"]].append(row)
        grouped = {mid for batch, members in groups.items() if batch in JOBS and len(members) > 1
                   for mid in (row["id"] for row in members if row["id"] != batch)}
        ordered = sorted(JOBS.values(), key=history_key, reverse=True)
        ordered = [_scene_status_locked(j, groups[j["id"]]) if len(groups.get(j["id"], ())) > 1 else j
                   for j in ordered if j["id"] not in grouped]
        eligible = [row for row in ordered if before is None or history_key(row) < before]
        page = eligible[:limit]
        boundary = history_key(page[-1]) if page else None
        if through is not None and (boundary is None or through < boundary):
            boundary = through
            page = [row for row in eligible if history_key(row) >= through]
        jobs = page
        if before is None:
            # Active jobs remain reachable even when newer runs fill the page.
            # They don't advance the cursor past undisplayed completed jobs.
            jobs = [row for row in eligible if (boundary is not None and history_key(row) >= boundary)
                    or row["status"] in ACTIVE_STATUSES]
        cursor = encode_history_cursor(boundary) if boundary is not None else None
        deletable_count = sum(j["status"] not in ACTIVE_STATUSES for j in ordered)
        return jsonify({"now": time.time(),
                        "jobs": [{k: copy.deepcopy(j.get(k)) for k in keys} for j in jobs],
                        "total": len(ordered), "cursor": cursor,
                        "next_cursor": cursor if len(eligible) > len(page) else None,
                        "deletable_count": deletable_count})


@app.route("/api/jobs", methods=["DELETE"])
def api_delete_jobs():
    """Delete selected jobs, or all finished jobs when no ids are supplied."""
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return api_error("history_invalid_delete", "无效的历史删除请求。", 400)
    requested = body.get("ids")
    if requested is not None and (not isinstance(requested, list) or
                                  any(not isinstance(job_id, str) for job_id in requested)):
        return api_error("history_invalid_delete", "无效的历史删除请求。", 400)

    deleted, skipped = [], []
    with JOBS_LOCK:
        job_ids = list(JOBS) if requested is None else list(dict.fromkeys(requested))
        visited = set()
        for requested_id in job_ids:
            row = JOBS.get(requested_id)
            job_id = (row.get("person_batch") or requested_id) if row else requested_id
            if job_id not in JOBS and row:
                job_id = requested_id
            if job_id in visited:
                continue
            visited.add(job_id)
            job = JOBS.get(job_id)
            if job is None:
                if requested is not None:
                    skipped.append({"id": job_id, "reason": "not found"})
                continue
            members = _batch_members_locked(job_id) or [job]
            if any(j["status"] in ACTIVE_STATUSES for j in members) or job.get("scene_state") in ACTIVE_STATUSES:
                skipped.append({"id": job_id, "reason": "job is still active"})
                continue
            try:
                work = Path(ARGS.work_dir).resolve()
                root = Path(job_dir(job_id)).resolve()
                # Older jobs have sibling folders; newer group members live inside this root.
                paths = [Path(job_dir(member["id"])).resolve() for member in members[1:]]
                if any(work not in path.parents for path in [root, *paths]):
                    raise OSError("Job path is outside the FlowHMR results directory")
                for path in [*(p for p in paths if root not in p.parents), root]:
                    if path.is_dir():
                        shutil.rmtree(path)
            except FileNotFoundError:
                pass
            except OSError as e:
                skipped.append({"id": job_id, "reason": f"could not remove files: {e}"})
                continue
            for member in members:
                mid = member["id"]
                visited.add(mid)
                JOBS.pop(mid, None)
                MEMBER_DIRS.pop(mid, None)
                SELECTIONS.pop(mid, None)
                deleted.append(mid)
    try:
        CHARACTER_LIBRARY.forget_actors(deleted)
    except OSError:
        app.logger.warning("Could not remove deleted jobs from character preferences", exc_info=True)
    return jsonify({"deleted": deleted, "skipped": skipped})


@app.route("/api/jobs/<job_id>/person/<int:index>.jpg")
def api_person_thumb(job_id, index):
    get_job(job_id)
    path = os.path.join(job_dir(job_id), "persons", f"{index}.jpg")
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="image/jpeg", conditional=True)


@app.route("/api/jobs/<job_id>/track")
def api_track(job_id):
    """Tracked person's boxes (xyxy, pixels of the 30 fps video), available after detection."""
    get_job(job_id)
    path = os.path.join(job_dir(job_id), SUB, f"{SUB}_bbox.npz")
    if not os.path.exists(path):
        abort(404)
    data = np.load(path)
    start, end = (int(x) for x in data["start_end"])
    return jsonify({"fps": 30, "start": start, "end": end,
                    "boxes": np.round(data["bbox"].astype(np.float64), 1).tolist()})


_RESULT_FILES = {
    "meta": ("mesh/meta.json", "application/json"),
    "vertices": ("mesh/vertices.bin", "application/octet-stream"),
    "faces": ("mesh/faces.bin", "application/octet-stream"),
    "npz": ("motion.npz", "application/octet-stream"),
}


def fbx_export_available(reference):
    if find_spec("fbx") is None:
        return False
    try:
        model_path(reference)
    except ValueError:
        return False
    return True


def export_stem(job):
    name = os.path.splitext(job["filename"])[0]
    person = job.get("selected_person")
    return f"{name}_person{person + 1}" if person is not None else name


def file_stamp(path):
    stat = os.stat(path)
    return f"{stat.st_mtime_ns:x}_{stat.st_size:x}"


def variant_preview(directory, body, reference, reference_path):
    variant = model_path(body)
    source = os.path.join(directory, "motion.npz")
    mesh = os.path.join(directory, "mesh")
    with open(os.path.join(mesh, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    size = int(meta["num_frames"]) * int(meta["num_vertices"]) * 6
    name = f"vertices_v2_{reference}_{body}_{file_stamp(source)}_{file_stamp(reference_path)}_{file_stamp(variant)}.bin"
    output = os.path.join(mesh, name)
    with BODY_PREVIEW_LOCK:
        if os.path.isfile(output) and os.path.getsize(output) == size:
            return output
        part = output + ".part"
        try:
            with open(part, "wb") as f:
                write_preview_vertices(source, variant, f, reference_path=reference_path)
            if os.path.getsize(part) != size:
                raise ValueError("体型预览的帧数与动作不一致。")
            os.replace(part, output)
        finally:
            if os.path.exists(part):
                os.unlink(part)
    return output


FBX_TEMPLATE_PROFILES = {"metahuman": "metahuman", "mixamo_iclone": "mixamo"}
FBX_PROFILES = {"flowhmr", "mixamo", "maya_adv", *FBX_TEMPLATE_PROFILES}
FBX_TEMPLATE_MB = 128
FBX_TEMPLATE_LIMIT = FBX_TEMPLATE_MB * (1 << 20)


def motion_export_source(job_id):
    job = get_job(job_id)
    directory = job_dir(job_id)
    is_scene = job.get("person_batch") == job_id
    source = os.path.join(directory, "scene" if is_scene else "", "motion.npz")
    if ((not job.get("scene") if is_scene else job["status"] != "done")
            or not os.path.isfile(source)):
        abort(404)
    return job, source, job["scene"]["people"] if is_scene else None


def select_export_people(available):
    if available is None:
        return None, ""
    numbers = {str(p["person"]) for p in available}
    requested = set(request.args.getlist("person")) or numbers
    if not requested <= numbers:
        raise UserMessage("请选择此场景中的人物。", "scene_people")
    chosen = [p for p in available if str(p["person"]) in requested]
    return chosen, hashlib.sha256(",".join(sorted(requested)).encode()).hexdigest()[:12]


def export_selection_label(available, chosen, signature):
    if len(chosen) == len(available):
        return "scene"
    if len(chosen) == 1:
        return f"person{chosen[0]['person']}"
    return f"scene_{len(chosen)}people_{signature[:6]}"


def copy_fbx_template(file, path, kind):
    digest = hashlib.sha256()
    size = 0
    with open(path, "wb") as output:
        for block in iter(lambda: file.stream.read(1 << 20), b""):
            size += len(block)
            if size > FBX_TEMPLATE_LIMIT:
                raise UserMessage("FBX 模板过大。", f"{kind}_template_size", max=FBX_TEMPLATE_MB)
            digest.update(block)
            output.write(block)
    if not size:
        raise UserMessage("请选择有效的 FBX 骨架模板。", f"{kind}_template_required")
    return digest.hexdigest()


@app.route("/api/jobs/<job_id>/fbx", methods=["GET", "POST"])
def api_fbx(job_id):
    job, source, available = motion_export_source(job_id)
    if find_spec("fbx") is None:
        abort(404)
    profile = request.args.get("profile", "flowhmr")
    template_kind = "mixamo" if profile == "mixamo" and request.method == "POST" else FBX_TEMPLATE_PROFILES.get(profile)
    if profile not in FBX_PROFILES or (request.method == "POST" and not template_kind):
        return api_error("fbx_profile", "请选择有效的 FBX 骨架预设。", 400)
    reference = job_body_reference(job)
    try:
        reference_path = model_path(reference)
    except ValueError:
        error = missing_body_reference(reference)
        return api_error(error.code, str(error), 409, **error.params)
    mode = request.args.get("mesh", "1")
    if mode not in ("0", "1"):
        return api_error("fbx_mesh_mode", "请选择仅骨架或包含蒙皮网格。", 400)
    first_pose = request.args.get("first_pose", "none")
    if first_pose not in ("none", "t", "a"):
        return api_error("fbx_reference_pose", "参考姿势请选择不添加、T Pose 或 A Pose。", 400)
    try:
        chosen, signature = select_export_people(available)
    except UserMessage as error:
        return api_error(error.code, str(error), 400, **error.params)
    if profile == "maya_adv":
        if chosen is not None and len(chosen) != 1:
            return api_error("maya_adv_export_options", "AdvancedSkeleton 每个 FBX 只导出一人的动作。", 400)
        mode, first_pose = "0", "t"
    template = None
    if template_kind:
        required = f"{template_kind}_template_required"
        if request.method != "POST":
            return api_error(required, "请选择有效的 FBX 骨架模板。", 400)
        if not local_model_request():
            return api_error(f"{template_kind}_local_only", "请在本机 FlowHMR 中选择 FBX 模板。", 403)
        if mode != "0" or first_pose != "none" or (chosen is not None and len(chosen) != 1):
            return api_error(f"{template_kind}_export_options", "按模板每次只导出一人的动画骨架。", 400)
        if request.content_length and request.content_length > FBX_TEMPLATE_LIMIT + (1 << 20):
            return api_error(f"{template_kind}_template_size", "FBX 模板过大。", 413, max=FBX_TEMPLATE_MB)
        templates = request.files.getlist("template")
        if len(templates) != 1 or not (templates[0].filename or "").lower().endswith(".fbx"):
            return api_error(required, "请选择有效的 FBX 骨架模板。", 400)
        template = templates[0]
    with_skin = mode == "1"
    suffix = "skin" if with_skin else "skeleton"
    body = request.args.get("body", reference) if with_skin else reference
    try:
        variant = model_path(body) if body != reference else None
    except ValueError as exc:
        return api_error("body_variant", str(exc), 400, variant=body)
    tag = f"_{reference}_{file_stamp(reference_path)}"
    if variant is not None:
        tag += f"_{body}_{file_stamp(variant)}"
    # The final NPZ is authoritative. Re-export old jobs with the corrected
    # rig instead of sending a legacy FBX saved before the exporter fix.
    exports = os.path.join(os.path.dirname(source), "exports")
    revision = 5 if first_pose != "none" else 4
    prefix = f"fbx_scene_v5_{signature}" if chosen is not None else f"fbx_mmr_v{revision}"
    if profile != "flowhmr":
        revision = 3 if profile in ("metahuman", "mixamo", "maya_adv") else 2
        prefix = f"fbx_{profile}_v{revision}_{signature or 'single'}"
    pose_tag = f"_{first_pose}_pose" if first_pose != "none" else ""
    cache_name = f"{prefix}_{file_stamp(source)}_{suffix}{tag}{pose_tag}"

    def write(template_path=None, template_hash=""):
        name = cache_name if profile == "flowhmr" else (
            f"{prefix}_{hashlib.sha256(f'{cache_name}:{template_hash}'.encode()).hexdigest()[:32]}"
        )
        path = os.path.join(exports, f"{name}.fbx")
        if not os.path.isfile(path):
            from flowhmr.core.export.smplh2fbx import SMPLH2FBX

            with SMPLH2FBX(smplh_model_path=reference_path) as converter:
                options = dict(export_mesh=with_skin, variant_path=variant, first_pose=first_pose,
                               profile=profile, template_path=template_path)
                if chosen is None:
                    converter.convert_npz_to_fbx(source, path, **options)
                else:
                    converter.convert_scene_to_fbx(source, chosen, path, **options)
        return path

    try:
        with FBX_EXPORT_LOCK:
            if template is None:
                path = write()
            else:
                os.makedirs(exports, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=exports, prefix=".template-") as scratch:
                    target = os.path.join(scratch, "body.fbx")
                    template_hash = copy_fbx_template(template, target, template_kind)
                    path = write(target, template_hash)
    except UserMessage as error:
        return api_error(error.code, str(error), 413 if error.code.endswith("_template_size") else 400, **error.params)
    except ValueError as error:
        code = f"{template_kind}_template_invalid" if template_kind else "motion_export_failed"
        return api_error(code, "无法导出动作。", 422, detail=str(error)[:300])
    except (BadZipFile, ImportError, KeyError, OSError, RuntimeError):
        app.logger.exception("FBX export failed for job %s", job_id)
        code = f"{template_kind}_template_invalid" if template_kind else "motion_export_failed"
        return api_error(code, "无法导出动作。", 422, detail="")
    if with_skin and body != "neutral":
        suffix = f"{body}_{suffix}"
    suffix += pose_tag
    if chosen is not None:
        stem = os.path.splitext(job["filename"])[0]
        group = export_selection_label(available, chosen, signature)
        name = f"{stem}_{profile}_{group}_{suffix}.fbx"
    else:
        name = f"{export_stem(job)}_{profile}_{suffix}.fbx"
    return send_file(path, mimetype="application/octet-stream", as_attachment=True, download_name=name, conditional=True)


def write_humanoid_motion(source, path, chosen, format_name, export_person):
    if chosen is None:
        export_person(source, path, None)
        return
    with np.load(source, allow_pickle=False) as data:
        fps = data["mocap_framerate"]

        def write(person, output):
            number, first, last = person["person"], person["start"], person["end"]
            motion = {key: data[f"{key}_{number}"] for key in ("poses", "trans", "betas")}
            if motion["trans"].shape != (last - first + 1, 3):
                raise ValueError(f"{format_name.upper()} source range does not match the scene motion")
            export_person({**motion, "mocap_framerate": fps}, output, person)

        if len(chosen) == 1:
            write(chosen[0], path)
            return
        with tempfile.TemporaryDirectory(dir=os.path.dirname(path), prefix=f".{format_name}-") as scratch:
            archive = os.path.join(scratch, "people.zip")
            with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=4) as packed:
                for person in chosen:
                    name = f"person_{person['person']:02d}.{format_name}"
                    output = os.path.join(scratch, name)
                    write(person, output)
                    packed.write(output, name)
            os.replace(archive, path)


def send_humanoid_motion(path, job, available, chosen, signature, format_name, extension):
    if chosen is None:
        name = f"{export_stem(job)}_{format_name}.{extension}"
    else:
        stem = os.path.splitext(job["filename"])[0]
        group = export_selection_label(available, chosen, signature)
        name = f"{stem}_{format_name}_{group}.{extension}"
    mime = "application/zip" if extension == "zip" else "application/octet-stream"
    return send_file(path, mimetype=mime, as_attachment=True, download_name=name, conditional=True)


@app.get("/api/jobs/<job_id>/vmd")
def api_vmd(job_id):
    job, source, available = motion_export_source(job_id)
    try:
        chosen, signature = select_export_people(available)
    except UserMessage as error:
        return api_error(error.code, str(error), 400, **error.params)
    multiple = chosen is not None and len(chosen) > 1
    extension = "zip" if multiple else "vmd"
    folder = os.path.join(os.path.dirname(source), "exports")
    try:
        reference_path = model_path(job_body_reference(job))
    except ValueError:
        reference_path = None
    reference_stamp = file_stamp(reference_path) if reference_path is not None else "standard"
    path = os.path.join(folder, f"vmd_v2_{file_stamp(source)}_{signature or 'single'}_{reference_stamp}.{extension}")
    try:
        with FBX_EXPORT_LOCK:
            if not os.path.isfile(path):
                from flowhmr.core.export.vmd import export_vmd

                os.makedirs(folder, exist_ok=True)
                end = max(p["end"] for p in chosen) if chosen is not None else None

                def write(motion, output, person):
                    export_vmd(motion, output, start_frame=person["start"] if person else 0,
                               scene_end=end, reference_path=reference_path)

                write_humanoid_motion(source, path, chosen, "vmd", write)
    except (OSError, ValueError, KeyError, BadZipFile) as error:
        app.logger.warning("VMD export failed for job %s: %s", job_id, error)
        return api_error("motion_export_failed", "无法导出动作。", 422, detail=str(error)[:300])
    return send_humanoid_motion(path, job, available, chosen, signature, "vmd", extension)


@app.get("/api/jobs/<job_id>/vrma")
def api_vrma(job_id):
    job, source, available = motion_export_source(job_id)
    try:
        chosen, signature = select_export_people(available)
    except UserMessage as error:
        return api_error(error.code, str(error), 400, **error.params)
    source_reference = job_body_reference(job)
    references = {}
    for person in chosen if chosen is not None else (None,):
        number = person["person"] if person else None
        reference = person.get("body_reference") if person else None
        if not isinstance(reference, str) or reference not in SMPLH_FILES:
            reference = source_reference
        try:
            references[number] = model_path(reference)
        except ValueError:
            error = missing_body_reference(reference)
            return api_error(error.code, str(error), 409, **error.params)
    extension = "zip" if chosen is not None and len(chosen) > 1 else "vrma"
    folder = os.path.join(os.path.dirname(source), "exports")
    try:
        floor = float(job["scene"].get("ground_m", 0.) if chosen is not None
                      else (job.get("correction") or {}).get("floor_m", 0.))
        if not np.isfinite(floor):
            raise ValueError("Invalid motion ground height")
        stamp = (file_stamp(source), signature, floor,
                 [(number, str(model), file_stamp(model)) for number, model in references.items()])
        key = hashlib.sha256(json.dumps(stamp, separators=(",", ":")).encode()).hexdigest()[:24]
        path = os.path.join(folder, f"vrma_v1_{key}.{extension}")
        with FBX_EXPORT_LOCK:
            if not os.path.isfile(path):
                from flowhmr.core.export.vrma import export_vrma

                os.makedirs(folder, exist_ok=True)

                def write(motion, output, person):
                    number = person["person"] if person else None
                    export_vrma(motion, output, reference_path=references[number],
                                start_frame=person["start"] if person else 0, floor=floor)

                write_humanoid_motion(source, path, chosen, "vrma", write)
    except (OSError, ValueError, KeyError, TypeError, BadZipFile) as error:
        app.logger.warning("VRMA export failed for job %s: %s", job_id, error)
        return api_error("motion_export_failed", "无法导出动作。", 422, detail=str(error)[:300])
    return send_humanoid_motion(path, job, available, chosen, signature, "vrma", extension)


@app.route("/api/jobs/<job_id>/thumbnail.jpg")
def api_thumbnail(job_id):
    job = get_job(job_id)
    directory = job_dir(job_id)
    person = job.get("selected_person") if job.get("person_batch") != job_id else None
    if person is not None:
        path = os.path.join(directory, "persons", f"{person}.jpg")
        if os.path.isfile(path):
            return send_file(path, mimetype="image/jpeg", conditional=True, max_age=3600)
    thumbnail = os.path.join(directory, "thumbnail.jpg")
    if not os.path.isfile(thumbnail):
        canonical = os.path.join(directory, SUB, f"{SUB}_30fps.mp4")
        video = canonical if os.path.isfile(canonical) else os.path.join(directory, f"input{job['ext']}")
        try:
            if not os.path.isfile(video) or not ensure_thumbnail(video, thumbnail):
                abort(404)
        except FileNotFoundError:
            abort(404)
    return send_file(thumbnail, mimetype="image/jpeg", conditional=True, max_age=3600)


@app.route("/api/jobs/<job_id>/skeleton")
def api_skeleton(job_id):
    job = get_job(job_id)
    if job["status"] != "done":
        abort(404)
    reference = job_body_reference(job)
    try:
        model_path(reference)
    except ValueError:
        error = missing_body_reference(reference)
        return api_error(error.code, str(error), 409, **error.params)
    try:
        path, info = INSPECTION.load(job_dir(job_id), body_reference=reference)
    except FileNotFoundError:
        abort(404)
    correction = job.get("correction")
    if correction:
        info = {**info, "ground": {"height_m": correction["floor_m"], "source": "correction"}}
    response = send_file(path, mimetype="application/octet-stream", conditional=True)
    response.headers["X-FlowHMR-Inspection"] = json.dumps(info, separators=(",", ":"))
    return response


@app.get("/api/jobs/<job_id>/pmx-motion")
def api_pmx_motion(job_id):
    job = get_job(job_id)
    directory = job_dir(job_id)
    if job["status"] != "done" or not os.path.isfile(os.path.join(directory, "motion.npz")):
        abort(404)
    reference = job_body_reference(job)
    try:
        reference_path = model_path(reference)
    except ValueError:
        error = missing_body_reference(reference)
        return api_error(error.code, str(error), 409, **error.params)
    from flowhmr.core.export.vrma import VRM_NECK_PITCH
    from flowhmr.utils.runtime_pmx_motion import pmx_motion

    try:
        joint_path, info = INSPECTION.load(directory, body_reference=reference)
        floor = (job.get("correction") or {}).get("floor_m", info["ground"]["height_m"])
        stream, meta = pmx_motion(directory, reference_path, joint_path, info, floor)
    except (OSError, ValueError, KeyError, TypeError, BadZipFile) as exc:
        app.logger.warning("Could not read the PMX source motion for %s: %s", job_id, exc)
        return api_error("pmx_motion", "无法读取此任务的角色预览动作。", 422)
    response = app.response_class(stream, mimetype="application/octet-stream")
    response.headers["X-FlowHMR-PMX-Motion"] = json.dumps(
        {**meta, "vrmNeckPitch": VRM_NECK_PITCH}, separators=(",", ":"), allow_nan=False)
    return response


@app.route("/api/jobs/<job_id>/<kind>")
def api_result(job_id, kind):
    job = get_job(job_id)
    if kind == "video":
        # Chromium may request the remainder of a partially buffered file by byte
        # range. Never switch the file behind a URL the player is still using.
        version = request.args.get("source")
        if version not in (None, "original", "preview"):
            abort(400)
        canonical = os.path.join(job_dir(job_id), SUB, f"{SUB}_30fps.mp4")
        has_preview = os.path.isfile(canonical)
        if version == "preview" and not has_preview:
            abort(404)
        path = canonical if has_preview and version != "original" else input_video(job_id)
        return send_file(path, conditional=True)
    if kind not in _RESULT_FILES:
        abort(404)
    if kind == "npz" and job.get("person_batch") == job_id:
        if not job.get("scene"):
            abort(404)
        return send_file(os.path.join(job_dir(job_id), "scene", "motion.npz"),
                         mimetype="application/octet-stream", as_attachment=True,
                         download_name=f"{os.path.splitext(job['filename'])[0]}_flowhmr_scene.npz", conditional=True)
    rel, mime = _RESULT_FILES[kind]
    path = os.path.join(job_dir(job_id), rel)
    reference = job_body_reference(job)
    body = request.args.get("body", "source")
    if kind == "vertices" and body not in (reference, "source"):
        if job["status"] != "done":
            abort(404)
        try:
            reference_path = model_path(reference)
        except ValueError:
            error = missing_body_reference(reference)
            return api_error(error.code, str(error), 409, **error.params)
        try:
            path = variant_preview(job_dir(job_id), body, reference, reference_path)
        except FileNotFoundError:
            abort(404)
        except (ValueError, KeyError) as exc:
            return api_error("body_variant", str(exc), 400, variant=body)
    if not os.path.exists(path):
        abort(404)
    as_attachment = kind == "npz"
    name = f"{export_stem(job)}_flowhmr.{kind}" if as_attachment else None
    return send_file(path, mimetype=mime, as_attachment=as_attachment,
                     download_name=name, conditional=True)


# --------------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(prog="app.py", description="FlowHMR web demo")
    p.add_argument("--ckpt", help="FlowHMR checkpoint (default: models/flowhmr_latest/flowhmr_latest.ckpt)")
    p.add_argument("--model_cfg", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--device", default=RUNTIME.default_device())
    p.add_argument("--sam-chunk-default", type=int, default=16, help="Initial value in the manual batch field")
    p.add_argument("--work_dir", default=os.path.join(REPO_ROOT, "output", "web_jobs"))
    args = p.parse_args()
    if args.sam_chunk_default < 1:
        p.error("--sam-chunk-default must be a positive integer")
    return args


def main():
    global ARGS, INSPECTION
    ARGS = parse_args()
    if ARGS.ckpt:
        ARGS.ckpt = os.path.abspath(ARGS.ckpt)
    if ARGS.model_cfg:
        ARGS.model_cfg = os.path.abspath(ARGS.model_cfg)
    ARGS.work_dir = os.path.abspath(ARGS.work_dir)
    os.makedirs(ARGS.work_dir, exist_ok=True)
    ensure_model_folders()
    restore_jobs()
    os.chdir(REPO_ROOT)  # body-model / asset paths are repo-relative
    INSPECTION = PreviewInspection()
    threading.Thread(target=worker_loop, daemon=True).start()
    print(f"\n  FlowHMR web demo ready:  http://{ARGS.host}:{ARGS.port}\n")
    # Bind the backend first; failed binds must not launch an orphan MCP service.
    server = make_server(ARGS.host, ARGS.port, app, threaded=True)

    def start_mcp():
        try:
            MCP_SERVICE.start_if_configured(f'http://127.0.0.1:{server.server_port}')
        except Exception as error:
            MCP_SERVICE.startup_error = str(error)
            app.logger.exception('MCP auto-start failed')
        finally:
            UI_EVENTS.publish('mcp_service')

    threading.Thread(target=start_mcp, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        MCP_SERVICE.stop()
        server.server_close()


if __name__ == "__main__":
    main()
