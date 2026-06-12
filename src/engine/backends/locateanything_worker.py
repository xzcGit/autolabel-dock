#!/usr/bin/env python3
"""LocateAnything sidecar worker — out-of-process torch/CUDA host.

Why this exists (root cause, already diagnosed — see memory
``locateanything-runtime``): activating LA's CUDA context (torch +
bitsandbytes 4-bit) **inside the Qt GUI process** kills the X server that
drives the same single GPU (``XIO: fatal IO error 0 on :1`` — no Python
traceback, no faulthandler, no VRAM OOM; the display server dies and the app
exits passively). The standalone CLI reference script runs the same model on
the same card without crashing, and the in-process YOLO/ultralytics CUDA path
is fine — so the conflict is specifically *LA's bitsandbytes/transformers load
co-resident with Qt*. The fix is to reproduce the "standalone script process
structure": move all of LA's torch/CUDA into this **separate long-lived
subprocess**.

Process / dependency contract (HARD):
    The module top level imports ONLY stdlib. ``torch`` / ``transformers`` /
    ``bitsandbytes`` / ``PIL`` are imported **locally inside functions**, so a
    test can import this module to check the protocol/contract without pulling
    in any heavy dependency. The heavy load only happens when ``main()`` runs
    in the spawned subprocess.

IPC contract (newline-delimited JSON over the parent-owned stdin/stdout pipes;
no port, no listener — lifetime is bound to the parent process):

    Startup (worker → parent):
        {"status": "<chinese progress>"}     # zero or more, during load
        {"ready": true}                      # model loaded, ready to serve
        {"preflight_failed": "<chinese>"}    # CUDA/VRAM gate failed; worker exits
        {"fatal": "<chinese>"}               # load crashed; worker exits

    Serving (parent → worker, one JSON object per line):
        {"image": "<path>", "prompt": "<str>"}   # inference request
        {"cmd": "shutdown"}                      # graceful exit

    Serving (worker → parent, one response per request):
        {"raw_text": "<str>", "width": <int>, "height": <int>}
        {"error": "<chinese>"}                   # inference failed; worker keeps serving

Channel purity (CRITICAL): stdout carries ONLY protocol JSON lines. The model's
``trust_remote_code`` modules, transformers, bitsandbytes, and tqdm all print to
stdout by default — that would corrupt the protocol stream. So ``main()``
captures the real stdout fd first (for protocol writes), then redirects
``sys.stdout`` to ``sys.stderr`` and routes transformers logging + tqdm to
stderr. The parent redirects this worker's stderr to a log file for diagnosis.

This loading / inference contract mirrors the ground-truth reference script
``.trellis/tasks/06-08-locateanything-backend/research/locate_anything_4bit_reference.py``.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

# ── Centralized constants (kept in sync with locateanything.py) ────────────

MODEL_REPO_ID = "nvidia/LocateAnything-3B"

# VRAM gating thresholds (GiB). The worker does the *real* torch.cuda check at
# startup; the GUI side only does a cheap nvidia-smi pre-check (no torch).
MIN_TOTAL_VRAM_GB = 6.0
MIN_FREE_VRAM_GB = 5.0

# Inference cost ceilings. Detection output is short, so 512 new tokens is ample
# and roughly halves the VRAM / latency peak versus the original 1024. The
# resolution cap downsamples the longest image edge before inference to bound
# the vision-tower activation memory — coordinates are [0,1000]-normalized so
# downscaling does not affect mapping correctness (we only ever shrink large
# images, never upscale small ones).
#
# Lowered 1280 → 1024 (2026-06): on a 6-8GB Turing card SHARED with the X server,
# the MoonViT vision tower runs on SDPA (no flash_attn on sm_75), so its attention
# activation peak scales steeply with vision-token count, i.e. with image AREA. A
# near-square 146kB photo at 1280 long-edge OOMs even with free VRAM reported,
# because the transient peak — not fragmentation — exceeds the headroom. 1024 cuts
# the vision-token count to ~0.64× (area ∝ edge²), shaving the peak proportionally.
MAX_NEW_TOKENS = 512
MAX_IMAGE_LONG_EDGE = 1024

# Area (total-pixel) ceiling — the ONE that actually prevents OOM. The MoonViT
# vision tower runs without flash-attn (sm_75), so its attention peak grows with
# the PATCH COUNT squared, i.e. with image AREA² — not the longest edge. Measured
# on the 8GB Turing box from the worker log: ~1480 patches peaked ~3.2G, ~5476
# patches peaked ~6.6G and OOM'd (a near-SQUARE 1024×1011 image; a 1024×271 WIDE
# image at the same long edge has 3.7× fewer patches and is fine). The two points
# fit peak ≈ 2.9G + 1.4e-7·patches² almost exactly. Capping to ~750k px (≈3.8k
# patches) puts the peak near 5.0G — ~2G headroom under the shared 7.78G card —
# and, crucially, it is a HARD ceiling independent of aspect ratio, so there is no
# "slightly bigger / more square image still OOMs" cliff. [0,1000]-normalized
# coords mean the proportional shrink never affects mapping correctness.
MAX_IMAGE_PIXELS = 750_000

# Sampling params for generation. CRITICAL: LocateAnything's custom generate()
# keys the sampling switch off ``temperature`` (default 0 → greedy argmax) and
# IGNORES ``do_sample`` entirely (see generate_utils.py:sample_tokens). Pure
# greedy decoding sends this model into a runaway box loop — it never argmaxes
# to the terminal ``im_end`` token, so it emits boxes until ``MAX_NEW_TOKENS``
# truncates it (~85 junk boxes at 512 tokens, ≈6 tokens/box). The official
# README usage and the reference script both sample at these values, which lets
# the model terminate normally after the real objects. A fixed seed restores
# best-effort run-to-run reproducibility on top of sampling.
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_SEED = 0


# ── Protocol writer (uses the captured real stdout fd) ─────────────────────


class _ProtocolWriter:
    """Writes newline-delimited JSON to the real stdout fd, flushing each line.

    Holds the file object opened on the *duplicated* original stdout fd so that
    redirecting ``sys.stdout`` to stderr (to keep the protocol channel clean)
    does not affect protocol writes.
    """

    def __init__(self, fileobj):
        self._fp = fileobj

    def send(self, obj: dict) -> None:
        try:
            self._fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._fp.flush()
        except (OSError, ValueError):
            # Parent pipe is gone — nothing we can do; let the read loop's EOF
            # handling terminate us.
            pass


def _quiet_heavy_libraries() -> None:
    """Route transformers logging + tqdm to stderr and lower verbosity.

    Best-effort: every step is guarded so a missing/older library never aborts
    the worker. Called after stdout has been redirected to stderr, so even the
    libraries that print directly land on stderr.
    """
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Make tqdm draw to stderr if it inspects this (some integrations honor it).
    os.environ.setdefault("TQDM_DISABLE", "0")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:  # noqa: BLE001 - logging tweak must never break the worker
        pass


def _preflight(torch) -> dict:
    """Real CUDA/VRAM gate (runs in the subprocess where torch is safe).

    Returns ``{"ok": bool, "message": str, "total_gb": float, "free_gb": float}``.
    Pure CPU is intentionally NOT offered (too slow).
    """
    if not torch.cuda.is_available():
        return {
            "ok": False,
            "message": "未检测到可用的 CUDA GPU。LocateAnything 需要 GPU，纯 CPU 模式过慢，未开放。",
        }
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": f"无法查询显存信息: {exc}"}

    total_gb = total_bytes / 1024 ** 3
    free_gb = free_bytes / 1024 ** 3

    if total_gb < MIN_TOTAL_VRAM_GB:
        return {
            "ok": False,
            "message": (
                f"显卡总显存 {total_gb:.1f}GB 低于建议的 {MIN_TOTAL_VRAM_GB:.0f}GB，"
                "无法稳定运行 LocateAnything。"
            ),
        }
    if free_gb < MIN_FREE_VRAM_GB:
        return {
            "ok": False,
            "message": (
                f"空闲显存仅 {free_gb:.1f}GB，低于所需的 ~{MIN_FREE_VRAM_GB:.0f}GB。"
                "请关闭占用显存的程序后重试。"
            ),
        }
    return {
        "ok": True,
        "message": f"显存检查通过 (空闲 {free_gb:.1f}GB / 总 {total_gb:.1f}GB)。",
        "total_gb": total_gb,
        "free_gb": free_gb,
    }


def _load_model(send):
    """4-bit NF4 + fp16 model load (mirrors the reference loading contract).

    Returns ``(model, tokenizer, processor, compute_dtype)``. Heavy imports are
    local. ``send`` emits ``{"status": ...}`` progress lines for the parent.
    """
    send({"status": "正在导入依赖…"})
    import torch
    from transformers import (
        AutoModel,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    compute_dtype = torch.float16  # Turing has no hardware bf16; fp16 is native.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    send({"status": "正在加载 tokenizer / processor…"})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_REPO_ID, trust_remote_code=True)

    send({"status": "正在加载模型 (4-bit NF4)，约需 40 秒…"})
    model = AutoModel.from_pretrained(
        MODEL_REPO_ID,
        quantization_config=bnb,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        trust_remote_code=True,
    ).eval()
    return model, tokenizer, processor, compute_dtype


def _downscale_for_inference(
    image, max_long_edge=MAX_IMAGE_LONG_EDGE, max_pixels=MAX_IMAGE_PIXELS
):
    """Shrink an image to fit BOTH ceilings before inference (never upscale).

    Two independent caps, because the vision-tower attention cost scales with the
    TOTAL patch count (≈ pixel area), not the longest edge:
      * ``max_long_edge`` bounds either dimension — keeps pathologically wide/tall
        panoramas from one huge axis;
      * ``max_pixels`` bounds the AREA — this is the cap that actually prevents the
        OOM, since a near-square image at the long-edge cap packs far more patches
        (hence a far larger O(patches²) attention peak) than a wide one at the same
        long edge. It is a hard, aspect-ratio-independent ceiling, so there is no
        "slightly bigger image still OOMs" cliff.
    The more restrictive cap wins. Coordinates are [0,1000]-normalized, so any
    proportional shrink is mapping-safe.
    """
    try:
        w, h = image.size
    except Exception:  # noqa: BLE001
        return image
    scale = 1.0
    long_edge = max(w, h)
    if long_edge > max_long_edge:
        scale = min(scale, max_long_edge / float(long_edge))
    area = float(w) * float(h)
    if area > max_pixels:
        scale = min(scale, (max_pixels / area) ** 0.5)
    if scale >= 1.0:
        return image
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    try:
        from PIL import Image

        resample = getattr(Image, "Resampling", Image).LANCZOS
        return image.resize(new_size, resample)
    except Exception:  # noqa: BLE001
        return image


# ── Diagnostics (stderr → worker log file; never touches the protocol stream) ─


def _log(msg: str) -> None:
    """Write a diagnostic line to stderr (captured in the worker log file)."""
    try:
        print(f"[la-worker] {msg}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass


def _cuda_mem_brief(torch) -> str:
    """One-line VRAM snapshot: peak-allocated / reserved / free / total (GiB)."""
    try:
        if not torch.cuda.is_available():
            return "cuda unavailable"
        free_b, total_b = torch.cuda.mem_get_info()
        return (
            f"peak_alloc={torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}G "
            f"reserved={torch.cuda.memory_reserved() / 1024 ** 3:.2f}G "
            f"free={free_b / 1024 ** 3:.2f}G total={total_b / 1024 ** 3:.2f}G"
        )
    except Exception as exc:  # noqa: BLE001
        return f"<mem query failed: {exc}>"


def _log_inference_failure(exc) -> None:
    """Dump the full traceback + ``torch.cuda.memory_summary()`` to stderr.

    The protocol only carries a short Chinese message to the GUI; the real numbers
    (attempted allocation, reserved, free, largest free block) live here in the
    worker log so a persistent OOM can be diagnosed from data instead of guessed.
    """
    _log(f"inference FAILED ({exc.__class__.__name__}) — traceback + mem summary:")
    try:
        traceback.print_exc(file=sys.stderr)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        if torch.cuda.is_available():
            print(torch.cuda.memory_summary(), file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass


def _run_inference(model, tokenizer, processor, compute_dtype, image_path, prompt):
    """Run one generation. Returns ``(raw_text, width, height)``.

    Mirrors the reference script's generate() contract: stochastic sampling
    (``temperature``/``top_p`` — NOT greedy; see GEN_TEMPERATURE for why greedy
    breaks this model), seeded for reproducibility, ``generation_mode="hybrid"``,
    ``use_cache=True``, pixel_values cast to the compute dtype, output already
    decoded text (do NOT batch_decode again). CUDA cache cleared afterward to
    release the transient generation peak on a memory-tight shared GPU. Raises on
    failure — the caller converts the exception into a readable ``{"error": ...}``
    response.
    """
    import torch
    from PIL import Image, ImageOps

    with Image.open(str(image_path)) as img:
        # Apply EXIF orientation so the model sees the SAME upright image the Qt
        # canvas shows (QImageReader.setAutoTransform(True) in src/utils/image.py
        # auto-transposes on display). PIL does NOT honor EXIF by default, so
        # without this the model would return boxes in the raw (un-rotated) frame
        # while the canvas displays the rotated image → boxes appear flipped
        # (orientation 3 → upside-down) or offset (orientation 6/8 → 90° rotated).
        img = ImageOps.exif_transpose(img)
        rgb = img.convert("RGB")
        width, height = rgb.size  # upright size — matches the displayed canvas
        rgb = _downscale_for_inference(rgb)
        down_w, down_h = rgb.size

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = processor.process_vision_info(messages)
        inputs = processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to("cuda")

    pixel_values = inputs["pixel_values"]
    if compute_dtype is not None:
        pixel_values = pixel_values.to(compute_dtype)

    # Diagnostic sizing (→ worker log): vision-token count drives the OOM, so log
    # the grid + sequence length next to image dims to correlate with the VRAM
    # peak below. reset_peak_memory_stats() scopes the peak to this generate().
    try:
        grid = inputs.get("image_grid_hws", None)
        grid_repr = grid.tolist() if hasattr(grid, "tolist") else grid
        _log(
            f"infer start: orig={width}x{height} down={down_w}x{down_h} "
            f"grid_hws={grid_repr} seq_len={int(inputs['input_ids'].shape[-1])} "
            f"| {_cuda_mem_brief(torch)}"
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass

    # Seed right before generate so sampling is reproducible per (image, prompt).
    torch.manual_seed(GEN_SEED)
    try:
        with torch.no_grad():
            response = model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws", None),
                tokenizer=tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                generation_mode="hybrid",
                temperature=GEN_TEMPERATURE,
                do_sample=True,
                top_p=GEN_TOP_P,
                repetition_penalty=1.1,
                verbose=False,
            )
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # max_memory_allocated() is the peak SINCE reset above; empty_cache() in the
    # finally frees cached blocks but does not reset this stat, so this is the
    # true generate() peak — the number to compare against free VRAM.
    _log(f"infer ok: {_cuda_mem_brief(torch)}")

    # model.generate() output is already decoded text — do NOT batch_decode.
    answer = response[0] if isinstance(response, (list, tuple)) else response
    if isinstance(answer, (list, tuple)):
        answer = answer[0]
    return str(answer), width, height


def _readable_inference_error(exc) -> str:
    """Map an inference exception to a readable Chinese message."""
    msg = str(exc)
    if "out of memory" in msg.lower() or exc.__class__.__name__ == "OutOfMemoryError":
        return (
            "推理显存不足 (CUDA OOM)。请关闭其他占用显存的程序，"
            "或改用更小的图片后重试。"
        )
    return f"推理失败: {msg}"


def main() -> int:
    """Sidecar entry point: redirect stdout, preflight, load, then serve.

    Returns a process exit code.
    """
    # 1) Capture the REAL stdout for the protocol channel, then redirect the
    #    Python-level stdout to stderr so library prints never corrupt protocol.
    try:
        real_stdout_fd = os.dup(sys.stdout.fileno())
        protocol_fp = os.fdopen(real_stdout_fd, "w", encoding="utf-8")
    except (OSError, ValueError):
        # Fallback (e.g. stdout already replaced): write to original sys.stdout.
        protocol_fp = sys.stdout
    sys.stdout = sys.stderr  # all subsequent print()/library output → stderr

    writer = _ProtocolWriter(protocol_fp)
    send = writer.send

    # 2) Offline so a flaky proxy can't stall the load (weights are local).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # 3) Heavy import of torch + real CUDA/VRAM preflight (safe in subprocess).
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        send({"preflight_failed": f"未安装或无法导入 torch: {exc}"})
        return 1

    pf = _preflight(torch)
    if not pf.get("ok"):
        send({"preflight_failed": pf.get("message", "显存预检未通过")})
        return 1
    send({"status": pf.get("message", "预检通过")})

    _quiet_heavy_libraries()

    # 4) Heavy model load.
    try:
        model, tokenizer, processor, compute_dtype = _load_model(send)
    except Exception as exc:  # noqa: BLE001 - any load failure must be reported
        send({"fatal": f"模型加载失败: {exc}"})
        return 2
    send({"ready": True})

    # 5) Serve requests line-by-line until shutdown / EOF.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except (ValueError, TypeError):
            send({"error": "请求 JSON 解析失败"})
            continue
        if not isinstance(req, dict):
            send({"error": "请求格式错误"})
            continue
        if req.get("cmd") == "shutdown":
            break
        image_path = req.get("image")
        prompt = req.get("prompt", "")
        if not image_path:
            send({"error": "缺少图片路径"})
            continue
        try:
            raw_text, width, height = _run_inference(
                model, tokenizer, processor, compute_dtype, image_path, prompt,
            )
            send({"raw_text": raw_text, "width": width, "height": height})
        except Exception as exc:  # noqa: BLE001 - never let one image kill the worker
            _log_inference_failure(exc)
            send({"error": _readable_inference_error(exc)})

    return 0


if __name__ == "__main__":
    sys.exit(main())
