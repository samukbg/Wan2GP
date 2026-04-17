"""
Resource guard for API generation requests.

Before each generation the guard:
  1. Waits until free system RAM and GPU VRAM exceed the requirements of the
     specific model that is about to be loaded.
  2. Serialises API requests so only one generation runs at a time.

Both checks are necessary because competing services (OctaSpace, Salad, …)
run inside WSL distros and consume host RAM and CUDA VRAM concurrently.

The guard NEVER drops a request — it queues indefinitely in FIFO order and
proceeds as soon as resources are sufficient.

────────────────────────────────────────────────────────────────────────────
Observed RAM requirements (from live partial-pinning logs):
  LTX2 22B distilled
    transformer : ~17 721 MB  (~17.3 GB)
    text_encoder: ~ 7 958 MB  (~ 7.8 GB)
    extras       : ~  300 MB  (VAE, audio …)
    → pinned total ≈ 25.4 GB, ideal full-pin ≈ 41.4 GB
    → safe minimum  = 26 GB + 6 GB safety buffer = 32 GB
    VRAM: offloaded block-by-block; WSL services mustn't hold more than
          ~16 GB of a 24 GB GPU → require ≥ 8 GB free VRAM

  Flux2 Klein 9B
    transformer : ~ 9 000 MB  (~ 8.8 GB, fully pinnable)
    text_encoder: ~ 9 017 MB  (~ 8.8 GB, fully pinnable)
    VAE          : ~   160 MB
    → pinned total ≈ 18.0 GB
    → safe minimum  = 18 GB + 4 GB safety buffer = 22 GB
    VRAM: both components fit in 24 GB; require ≥ 8 GB free VRAM
────────────────────────────────────────────────────────────────────────────

All thresholds are overridable via environment variables (see below).
"""

import os
import threading
import time

import torch


# ---------------------------------------------------------------------------
# Per-model-family memory requirements
# (ram_gb, vram_gb) = minimum FREE resources needed before generation starts.
# The model has been unloaded (release_model()) before this check runs, so
# these values represent what the system must have available to load and run
# the model without OOM.
# ---------------------------------------------------------------------------

_MODEL_REQUIREMENTS: dict[str, dict] = {
    # LTX2 22B family (distilled or not):
    # 25.4 GB observed pinned + 6 GB safety buffer; 8 GB free VRAM minimum.
    "ltx2": {
        "ram_gb":  float(os.environ.get("WGP_LTX2_MIN_FREE_RAM_GB",  "32.0")),
        "vram_gb": float(os.environ.get("WGP_LTX2_MIN_FREE_VRAM_GB",  "8.0")),
    },
    # Flux2 family (Klein 4B / 9B, dev, etc.):
    # 18.0 GB observed + 4 GB safety buffer; 8 GB free VRAM minimum.
    "flux2": {
        "ram_gb":  float(os.environ.get("WGP_FLUX2_MIN_FREE_RAM_GB",  "22.0")),
        "vram_gb": float(os.environ.get("WGP_FLUX2_MIN_FREE_VRAM_GB",  "8.0")),
    },
    # Legacy Flux (1.x):
    "flux": {
        "ram_gb":  float(os.environ.get("WGP_FLUX_MIN_FREE_RAM_GB",   "20.0")),
        "vram_gb": float(os.environ.get("WGP_FLUX_MIN_FREE_VRAM_GB",   "8.0")),
    },
    # Hunyuan family (large models):
    "hunyuan": {
        "ram_gb":  float(os.environ.get("WGP_HUNYUAN_MIN_FREE_RAM_GB", "40.0")),
        "vram_gb": float(os.environ.get("WGP_HUNYUAN_MIN_FREE_VRAM_GB", "8.0")),
    },
    # Wan 14B family:
    "wan": {
        "ram_gb":  float(os.environ.get("WGP_WAN_MIN_FREE_RAM_GB",    "28.0")),
        "vram_gb": float(os.environ.get("WGP_WAN_MIN_FREE_VRAM_GB",    "8.0")),
    },
    # Conservative fallback for any unknown model:
    "default": {
        "ram_gb":  float(os.environ.get("WGP_DEFAULT_MIN_FREE_RAM_GB",  "8.0")),
        "vram_gb": float(os.environ.get("WGP_DEFAULT_MIN_FREE_VRAM_GB",  "4.0")),
    },
}

# Mapping from model_type prefix → requirements key.
# Checked in order; first match wins.
_FAMILY_PREFIX_MAP: list[tuple[str, str]] = [
    ("ltx2",     "ltx2"),
    ("pi_flux2", "flux2"),   # pi_flux2 is the PixArt-Flux2 variant
    ("flux2",    "flux2"),
    ("flux",     "flux"),
    ("hunyuan",  "hunyuan"),
    ("wan",      "wan"),
]

POLL_INTERVAL_SECS = float(os.environ.get("WGP_RESOURCE_POLL_SECS", "5.0"))

# Binary semaphore — at most one API generation at a time.
# CPython wakes threads in FIFO order so queue discipline is preserved.
_api_gen_semaphore = threading.Semaphore(1)

_queue_depth = 0
_queue_lock  = threading.Lock()

_LOG = "[ResourceGuard]"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _requirements_for(model_type: str) -> dict:
    """Return the (ram_gb, vram_gb) requirements dict for model_type."""
    if model_type:
        lower = model_type.lower()
        for prefix, key in _FAMILY_PREFIX_MAP:
            if lower.startswith(prefix):
                return _MODEL_REQUIREMENTS[key]
    return _MODEL_REQUIREMENTS["default"]


def _free_ram_gb() -> float:
    """Available system RAM in GiB; inf if psutil is unavailable."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return float("inf")


def _free_vram_gb() -> float:
    """
    Free GPU VRAM in GiB.

    torch.cuda.mem_get_info() queries the WDDM driver which accounts for
    VRAM held by ALL processes, including WSL2 workloads running OctaSpace
    or Salad containers.  Falls back to inf on non-CUDA machines.
    """
    try:
        if not torch.cuda.is_available():
            return float("inf")
        free_bytes, _ = torch.cuda.mem_get_info(0)
        return free_bytes / (1024 ** 3)
    except Exception:
        return float("inf")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def acquire_generation_slot(model_type: str = "") -> None:
    """
    Block the calling thread until ALL of the following are true:
      • Free system RAM ≥ model's required RAM
      • Free GPU VRAM  ≥ model's required VRAM
      • No other API generation is currently running

    Never times out — the request stays in the queue until resources free up.
    Always call release_generation_slot() in a finally block.
    """
    global _queue_depth

    req = _requirements_for(model_type)
    need_ram  = req["ram_gb"]
    need_vram = req["vram_gb"]

    with _queue_lock:
        _queue_depth += 1
        depth = _queue_depth

    family = model_type or "unknown"
    if depth > 1:
        print(f"{_LOG} Request for '{family}' queued (position {depth}).")

    logged_ram = logged_vram = logged_queue = False

    while True:
        ram  = _free_ram_gb()
        vram = _free_vram_gb()

        # ── RAM gate ────────────────────────────────────────────────────────
        if ram < need_ram:
            if not logged_ram:
                print(
                    f"{_LOG} Waiting for RAM [{family}]: "
                    f"{ram:.1f} GB free, need ≥{need_ram:.1f} GB"
                )
                logged_ram = True
            time.sleep(POLL_INTERVAL_SECS)
            continue
        if logged_ram:
            print(f"{_LOG} RAM available: {ram:.1f} GB free (need {need_ram:.1f} GB).")
        logged_ram = False

        # ── VRAM gate ───────────────────────────────────────────────────────
        if vram < need_vram:
            if not logged_vram:
                print(
                    f"{_LOG} Waiting for VRAM [{family}]: "
                    f"{vram:.1f} GB free, need ≥{need_vram:.1f} GB"
                )
                logged_vram = True
            time.sleep(POLL_INTERVAL_SECS)
            continue
        if logged_vram:
            print(f"{_LOG} VRAM available: {vram:.1f} GB free (need {need_vram:.1f} GB).")
        logged_vram = False

        # ── Serialisation gate ──────────────────────────────────────────────
        if not _api_gen_semaphore.acquire(blocking=False):
            if not logged_queue:
                print(
                    f"{_LOG} Another generation is running — "
                    f"'{family}' waiting in queue "
                    f"(RAM {ram:.1f} GB, VRAM {vram:.1f} GB free)"
                )
                logged_queue = True
            time.sleep(POLL_INTERVAL_SECS)
            continue
        if logged_queue:
            print(f"{_LOG} Previous generation finished — proceeding with '{family}'.")
        logged_queue = False

        # All gates cleared — we hold the semaphore.
        with _queue_lock:
            _queue_depth = max(0, _queue_depth - 1)
            remaining = _queue_depth

        print(
            f"{_LOG} Starting '{family}' generation "
            f"(RAM {ram:.1f}/{need_ram:.1f} GB, VRAM {vram:.1f}/{need_vram:.1f} GB"
            + (f" — {remaining} request(s) still queued" if remaining else "")
            + ")"
        )
        return


def release_generation_slot() -> None:
    """Release the slot acquired by acquire_generation_slot()."""
    _api_gen_semaphore.release()


def resource_status() -> dict:
    """Current resource snapshot — exposed as the /resource_status Gradio API."""
    ram  = _free_ram_gb()
    vram = _free_vram_gb()
    with _queue_lock:
        depth = _queue_depth
    return {
        "free_ram_gb":  round(ram,  2),
        "free_vram_gb": round(vram, 2),
        "queue_depth":  depth,
        "slot_available": _api_gen_semaphore._value > 0,
        "model_requirements": {
            k: v for k, v in _MODEL_REQUIREMENTS.items()
        },
    }
