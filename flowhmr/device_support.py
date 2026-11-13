"""Pick and describe the local inference backend.

torch exposes NVIDIA and AMD/ROCm GPUs under the same ``cuda`` namespace, so
``device.type`` alone cannot tell them apart; ``torch.version.hip`` can. Apple
Silicon shows up as ``mps``. The resolved :class:`Backend` carries the device
name plus the capability bits callers actually need, instead of repeating
``device.type == "cuda"`` checks that silently mean different things.
"""

import platform
import subprocess
from dataclasses import dataclass

import torch

CUDA_KINDS = ("nvidia", "rocm")
DEVICE_CHOICES = ("auto", "nvidia", "rocm", "cuda", "mps", "apple", "xpu", "cpu")


@dataclass(frozen=True)
class Backend:
    """A resolved device plus what it can actually run."""

    kind: str
    device: str
    label: str
    runtime: str
    supports_fp64: bool
    supports_bf16: bool
    torchvision_nms: bool
    gpu_preprocess: bool

    @property
    def is_cuda_like(self) -> bool:
        """True when torch addresses this device through the CUDA API."""
        return self.kind in CUDA_KINDS

    def summary(self) -> dict:
        return {
            "kind": self.kind,
            "device": self.device,
            "label": self.label,
            "runtime": self.runtime,
        }


def mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _apple_label() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or platform.machine()
    except (OSError, subprocess.SubprocessError):
        return platform.machine()


def _cuda_runtime() -> str:
    hip = getattr(torch.version, "hip", None)
    if hip:
        return f"rocm{hip}"
    cuda = getattr(torch.version, "cuda", None)
    return f"cuda{cuda}" if cuda else "cuda"


def _cuda_kind() -> str:
    return "rocm" if getattr(torch.version, "hip", None) else "nvidia"


def _cuda_bf16(kind: str, index: int) -> bool:
    if kind == "rocm":
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    return torch.cuda.get_device_capability(index)[0] >= 8


def _cuda_backend(index: int | None) -> Backend:
    kind = _cuda_kind()
    resolved = torch.cuda.current_device() if index is None else index
    props = torch.cuda.get_device_properties(resolved)
    return Backend(
        kind=kind,
        device="cuda" if index is None else f"cuda:{index}",
        label=props.name,
        runtime=_cuda_runtime(),
        supports_fp64=True,
        supports_bf16=_cuda_bf16(kind, resolved),
        torchvision_nms=True,
        gpu_preprocess=True,
    )


def _xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    return bool(xpu is not None and xpu.is_available())


def _xpu_backend(index: int | None = None) -> Backend:
    device = torch.device("xpu") if index is None else torch.device("xpu", index)
    try:
        label = torch.xpu.get_device_name(device).strip()
    except Exception:
        label = "Intel XPU"
    bf16 = getattr(torch.xpu, "is_bf16_supported", None)
    return Backend(
        kind="intel",
        device=str(device),
        label=label or "Intel XPU",
        runtime="xpu",
        supports_fp64=True,
        supports_bf16=bool(bf16(device)) if bf16 is not None else False,
        # torchvision has no XPU NMS, and the pipeline keeps detection and the
        # crop preparation on the host while only the SAM encoder uses the XPU.
        torchvision_nms=False,
        gpu_preprocess=False,
    )


def _runtime_hint() -> str | None:
    """Device the Windows runtime picker resolved to, when it is meaningful."""
    try:
        from .runtime_environments import RUNTIME, SUPPORTED
    except Exception:
        return None
    return RUNTIME.default_device() if SUPPORTED else None


def detect_device(value: str | None = None) -> Backend:
    """Resolve ``auto``/``cuda``/``cuda:N``/``rocm``/``mps``/``apple``/``xpu``/``cpu``.

    ``auto`` follows the selected Windows runtime (CUDA, ROCm, XPU or CPU) and
    otherwise prefers CUDA, then Apple MPS, then CPU; an explicit choice raises
    when the hardware is missing instead of silently falling back.
    """
    name = (value or "auto").strip().lower()
    if name in ("", "auto"):
        hint = _runtime_hint()
        if hint in ("cpu", "xpu"):
            return detect_device(hint)
        if torch.cuda.is_available():
            return _cuda_backend(None)
        if mps_available():
            return Backend("apple", "mps", _apple_label(), "mps",
                           supports_fp64=False, supports_bf16=False,
                           torchvision_nms=False, gpu_preprocess=False)
        return Backend("cpu", "cpu", platform.processor() or platform.machine(), "",
                       supports_fp64=True, supports_bf16=False,
                       torchvision_nms=False, gpu_preprocess=False)

    if name in ("nvidia", "rocm", "cuda") or name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {value!r} requested but CUDA/HIP is not available")
        if name == "rocm" and _cuda_kind() != "rocm":
            raise RuntimeError("device 'rocm' requested but torch was built for CUDA")
        if name == "nvidia" and _cuda_kind() != "nvidia":
            raise RuntimeError("device 'nvidia' requested but torch was built for ROCm")
        index = int(name.split(":", 1)[1]) if name.startswith("cuda:") else None
        return _cuda_backend(index)

    if name in ("mps", "apple"):
        if not mps_available():
            raise RuntimeError(f"device {value!r} requested but Apple MPS is not available")
        return Backend("apple", "mps", _apple_label(), "mps",
                       supports_fp64=False, supports_bf16=False,
                       torchvision_nms=False, gpu_preprocess=False)

    if name == "xpu" or name.startswith("xpu:"):
        if not _xpu_available():
            raise RuntimeError(f"device {value!r} requested but torch has no XPU device")
        index = int(name.split(":", 1)[1]) if name.startswith("xpu:") else None
        return _xpu_backend(index)

    if name == "cpu":
        return Backend("cpu", "cpu", platform.processor() or platform.machine(), "",
                       supports_fp64=True, supports_bf16=False,
                       torchvision_nms=False, gpu_preprocess=False)

    raise ValueError(f"unknown device {value!r}; expected one of {', '.join(DEVICE_CHOICES)}")


def default_device() -> str:
    """CUDA when present, otherwise Apple-Silicon MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"
