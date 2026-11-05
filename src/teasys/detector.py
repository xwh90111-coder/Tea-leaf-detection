"""YOLOv12n inference wrapper + annotation rendering.

The detector is loaded once per process (thread-safe) and shared by the three
detection modes of the Dash application (single image / batch / video).
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config

_MODEL = None
_MODEL_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"


def get_model():
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                from ultralytics import YOLO

                path = config.resolve_model_path()
                model = YOLO(str(path))
                try:
                    model.to(f"cuda:{config.DEVICE}" if config.DEVICE != "cpu" else "cpu")
                except Exception:
                    model.to("cpu")
                _MODEL = model
    return _MODEL


def model_info() -> dict[str, Any]:
    model = get_model()
    n_params = sum(p.numel() for p in model.model.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops

        flops = float(get_flops(model.model, config.IMG_SIZE))
    except Exception:
        flops = None
    return {
        "path": str(config.resolve_model_path()),
        "parameters": n_params,
        "gflops": round(flops, 2) if flops else None,
        "names": model.names,
    }


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def draw_detections(
    image: Image.Image, detections: list[dict[str, Any]], line_width: int | None = None
) -> Image.Image:
    """Draw boxes + Chinese class labels on a RGB PIL image."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    lw = line_width or max(2, round(min(w, h) / 320))
    font = _font(max(14, round(min(w, h) / 32)))
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = config.CLASS_COLORS.get(det["class_name"], "#2e7d32")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        label = f"{det.get('class_name_cn', det['class_name'])} {det['confidence']:.2f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 8)
        draw.rectangle([x1, ty, x1 + tw + 10, ty + th + 8], fill=color)
        draw.text((x1 + 5, ty + 2), label, fill="white", font=font)
    return img


def assess_severity(
    detections: list[dict[str, Any]], image_area: float
) -> tuple[str, float, str, str]:
    """Rule-based severity + alert level (documented in docs/method.md)."""
    if not detections:
        return "无", 0.0, "none", ""
    area = sum(d.get("area_ratio", 0) for d in detections)
    n = len(detections)
    if n <= 2 and area < 0.03:
        severity, level = "轻度", "info"
    elif n <= 5 and area < 0.12:
        severity, level = "中度", "warning"
    else:
        severity, level = "重度", "danger"
    counts: dict[str, int] = {}
    for d in detections:
        cn = config.CLASS_NAMES_CN.get(d["class_name"], d["class_name"])
        counts[cn] = counts.get(cn, 0) + 1
    desc = "、".join(f"{k}{v}处" for k, v in counts.items())
    level_cn = {"info": "黄色提示", "warning": "橙色预警", "danger": "红色预警"}[level]
    if severity == "轻度":
        msg = f"{level_cn}：检出{desc}，病害程度轻度，建议持续监测。"
    elif severity == "中度":
        msg = f"{level_cn}：检出{desc}，病害程度中度，建议7日内采取防治措施。"
    else:
        msg = f"{level_cn}：检出{desc}，病害程度重度，建议立即复核并开展应急防治。"
    return severity, area, level, msg


def _postprocess(result, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    w, h = image_size
    names = result.names
    dets: list[dict[str, Any]] = []
    if result.boxes is None:
        return dets
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy().astype(int)
    for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, clss):
        name = names.get(int(cid), str(cid)) if isinstance(names, dict) else names[int(cid)]
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(w * h)
        dets.append(
            {
                "class_id": int(cid),
                "class_name": name,
                "class_name_cn": config.CLASS_NAMES_CN.get(name, name),
                "confidence": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "area_ratio": float(area_ratio),
            }
        )
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets


def detect_image(
    path: str | Path,
    *,
    conf: float | None = None,
    iou: float | None = None,
    annotate: bool = True,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run detection on one image and optionally save an annotated copy."""
    path = Path(path)
    image = Image.open(path).convert("RGB")
    arr = np.array(image)[:, :, ::-1]  # RGB -> BGR
    model = get_model()
    with _INFER_LOCK:
        t0 = time.perf_counter()
        results = model.predict(
            source=arr,
            imgsz=config.IMG_SIZE,
            conf=conf if conf is not None else config.CONF_THRESHOLD,
            iou=iou if iou is not None else config.IOU_THRESHOLD,
            max_det=config.MAX_DET,
            device=config.DEVICE if config.DEVICE != "cpu" else "cpu",
            verbose=False,
        )
        wall_ms = (time.perf_counter() - t0) * 1000
    result = results[0]
    dets = _postprocess(result, image.size)
    severity, area, level, msg = assess_severity(dets, image.size[0] * image.size[1])
    annotated_path = None
    if annotate:
        out_dir = out_dir or config.OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = out_dir / f"{path.stem}_{uuid.uuid4().hex[:6]}_pred.jpg"
        draw_detections(image, dets).save(annotated_path, quality=92)
    return {
        "detections": dets,
        "severity": severity,
        "lesion_area_ratio": area,
        "alert_level": level,
        "alert_message": msg,
        "annotated_path": str(annotated_path) if annotated_path else None,
        "timings": {
            "preprocess_ms": float(result.speed.get("preprocess", 0)),
            "inference_ms": float(result.speed.get("inference", 0)),
            "postprocess_ms": float(result.speed.get("postprocess", 0)),
            "wall_ms": wall_ms,
        },
        "image_size": list(image.size),
    }


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
def _transcode_h264(src: Path, dst: Path) -> None:
    """Re-encode to browser-compatible H.264 (MP4V is not playable in HTML5)."""
    import shutil
    import subprocess

    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.run(
            [exe, "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-loglevel", "error", str(dst)],
            capture_output=True,
            timeout=300,
        )
        if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            src.unlink(missing_ok=True)
            return
    except Exception:
        pass
    shutil.move(str(src), str(dst))


def detect_video(
    path: str | Path,
    *,
    conf: float | None = None,
    max_seconds: int | None = None,
    record_every_s: float = 1.0,
    progress_cb=None,
) -> dict[str, Any]:
    """Process an uploaded video, write an annotated MP4 and return a summary.

    ``total_lesions`` is the peak number of simultaneous lesions observed in a
    single frame (not the sum over frames, which would grow with video length).
    """
    import cv2

    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 640)
    max_seconds = max_seconds or config.MAX_VIDEO_SECONDS
    max_frames = min(total_frames or 10 ** 9, int(fps * max_seconds))

    out_path = config.OUTPUT_DIR / f"{path.stem}_{uuid.uuid4().hex[:6]}_pred.mp4"
    raw_path = out_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), min(fps, 30.0), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("无法创建视频编码器")
    model = get_model()
    timeline: list[dict[str, Any]] = []
    class_peak: dict[str, int] = {}
    peak_dets: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    peak = 0
    max_conf = 0.0
    frames = 0
    idx = 0
    next_record = 0.0
    poster_path = None
    poster_at = None
    t_start = time.perf_counter()
    try:
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            with _INFER_LOCK:
                res = model.predict(
                    source=frame,
                    imgsz=config.IMG_SIZE,
                    conf=conf if conf is not None else config.CONF_THRESHOLD,
                    iou=config.IOU_THRESHOLD,
                    max_det=config.MAX_DET,
                    device=config.DEVICE if config.DEVICE != "cpu" else "cpu",
                    verbose=False,
                )[0]
            dets = _postprocess(res, (width, height))
            frame_rgb = Image.fromarray(frame[:, :, ::-1])
            frame = np.array(draw_detections(frame_rgb, dets))[:, :, ::-1]

            sec = idx / fps
            counts: dict[str, int] = {}
            for d in dets:
                counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
                max_conf = max(max_conf, d["confidence"])
            if len(dets) > peak:
                peak = len(dets)
                peak_dets = dets
                class_peak = counts
            if sec >= next_record:
                timeline.append({"second": round(sec, 1), "count": len(dets)})
                next_record = sec + record_every_s

            if poster_at is None:
                poster_at = max(1, int(fps * 1.0))
            if idx == poster_at:
                poster_path = config.OUTPUT_DIR / f"{out_path.stem}_poster.jpg"
                cv2.imwrite(str(poster_path), frame)
            writer.write(frame)
            idx += 1
            frames += 1
            if progress_cb and idx % 10 == 0:
                progress_cb(min(0.99, idx / max(max_frames, 1)))
    finally:
        cap.release()
        writer.release()

    _transcode_h264(raw_path, out_path)
    elapsed = time.perf_counter() - t_start
    if peak <= 2:
        severity, level = ("轻度", "info") if peak else ("无", "none")
    elif peak <= 5:
        severity, level = "中度", "warning"
    else:
        severity, level = "重度", "danger"
    msg = ""
    if peak:
        level_cn = {"info": "黄色提示", "warning": "橙色预警", "danger": "红色预警"}[level]
        desc = "、".join(
            f"{config.CLASS_NAMES_CN.get(k, k)}{v}处" for k, v in class_peak.items()
        )
        msg = f"{level_cn}：视频峰值检出{desc}，病害程度{severity}。"
    return {
        "detections": peak_dets,
        "total_lesions": peak,
        "class_counter": class_peak,
        "class_peak": class_peak,
        "max_confidence": max_conf,
        "severity": severity,
        "alert_level": level,
        "alert_message": msg,
        "annotated_path": str(out_path),
        "poster_path": str(poster_path) if poster_path else None,
        "timeline": timeline,
        "frames": frames,
        "fps": round(fps, 2),
        "elapsed_s": round(elapsed, 2),
    }
