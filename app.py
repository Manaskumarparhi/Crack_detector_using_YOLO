import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import tempfile, os, csv, io, time

MODEL_PATH = "best.pt"
model      = YOLO(MODEL_PATH)

# ────────────────────────────────────────────────────────────────────────────
# Core helpers (from measure_crack.py)
# ────────────────────────────────────────────────────────────────────────────

def compute_gsd(altitude_m, focal_length_mm, sensor_width_mm, image_width_px):
    """Ground Sampling Distance — mm per pixel."""
    if focal_length_mm <= 0 or image_width_px <= 0:
        return 1.0
    return (altitude_m * 1000 * sensor_width_mm) / (focal_length_mm * image_width_px)


def estimate_crack_length(bbox_xyxy, gsd_mm_per_px):
    x1, y1, x2, y2 = bbox_xyxy
    width_px    = abs(x2 - x1)
    height_px   = abs(y2 - y1)
    diagonal_px = float(np.sqrt(width_px**2 + height_px**2))
    return {
        "width_px"   : round(width_px,    2),
        "height_px"  : round(height_px,   2),
        "diagonal_px": round(diagonal_px, 2),
        "length_mm"  : round(diagonal_px * gsd_mm_per_px,        2),
        "length_cm"  : round(diagonal_px * gsd_mm_per_px / 10,   4),
        "length_m"   : round(diagonal_px * gsd_mm_per_px / 1000, 6),
    }


def annotate_frame(frame_bgr, conf, gsd):
    """Run model on one BGR frame, return annotated BGR frame + list of crack dicts."""
    results  = model.predict(frame_bgr, conf=conf, verbose=False)
    boxes    = results[0].boxes
    img      = frame_bgr.copy()
    font     = cv2.FONT_HERSHEY_SIMPLEX
    cracks   = []

    for i, box in enumerate(boxes or []):
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        confidence      = float(box.conf[0])
        m               = estimate_crack_length([x1, y1, x2, y2], gsd)

        color = (0, 0, 220)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.line(img, (x1, y1), (x2, y2), (0, 140, 255), 1)

        label = f"#{i+1}  {m['length_cm']:.2f}cm  ({confidence:.0%})"
        (lw, lh), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), font, 0.5, (255, 255, 255), 1)

        cracks.append({
            "crack_id"   : i + 1,
            "confidence" : f"{confidence:.2%}",
            "box_x1"     : x1, "box_y1": y1, "box_x2": x2, "box_y2": y2,
            **m
        })

    return img, cracks


def format_crack_details(cracks, gsd, filename=""):
    """Build a readable multi-line string from crack list."""
    if not cracks:
        return "No cracks detected."
    lines = []
    if filename:
        lines.append(f"File: {filename}")
    lines.append(f"GSD: {gsd:.5f} mm/px\n")
    for c in cracks:
        lines.append(
            f"Crack {c['crack_id']}\n"
            f"  Confidence   : {c['confidence']}\n"
            f"  Bounding box : ({c['box_x1']},{c['box_y1']}) → ({c['box_x2']},{c['box_y2']})\n"
            f"  Width  (px)  : {c['width_px']}\n"
            f"  Height (px)  : {c['height_px']}\n"
            f"  Diagonal(px) : {c['diagonal_px']}\n"
            f"  Length (mm)  : {c['length_mm']}\n"
            f"  Length (cm)  : {c['length_cm']}\n"
            f"  Length (m)   : {c['length_m']}\n"
        )
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Tab 1 — Single image
# ────────────────────────────────────────────────────────────────────────────

def detect_single_image(pil_img, conf, altitude, focal, sensor):
    if pil_img is None:
        return None, "Upload an image first.", ""

    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    H, W  = frame.shape[:2]
    gsd   = compute_gsd(altitude, focal, sensor, W)

    annotated, cracks = annotate_frame(frame, conf, gsd)
    out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    summary = f"Cracks detected: {len(cracks)}   |   Image: {W}×{H}px   |   GSD: {gsd:.5f} mm/px"
    details = format_crack_details(cracks, gsd)
    return Image.fromarray(out_rgb), summary, details


# ────────────────────────────────────────────────────────────────────────────
# Tab 2 — Folder of images
# ────────────────────────────────────────────────────────────────────────────

def detect_folder(files, conf, altitude, focal, sensor, progress=gr.Progress()):
    if not files:
        return None, "Upload at least one image.", "", None

    out_dir = Path(tempfile.mkdtemp()) / "annotated"
    out_dir.mkdir()

    csv_rows      = []
    all_details   = []
    preview_img   = None
    total_cracks  = 0

    for idx, file_obj in enumerate(progress.tqdm(files, desc="Processing images")):
        fpath = Path(file_obj.name)
        if fpath.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            continue

        frame = cv2.imread(str(fpath))
        if frame is None:
            continue
        H, W = frame.shape[:2]
        gsd  = compute_gsd(altitude, focal, sensor, W)

        annotated, cracks = annotate_frame(frame, conf, gsd)
        total_cracks += len(cracks)

        # Save annotated image
        save_path = out_dir / fpath.name
        cv2.imwrite(str(save_path), annotated)

        # Keep first annotated as preview
        if preview_img is None:
            preview_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        # Collect detail text
        all_details.append(format_crack_details(cracks, gsd, filename=fpath.name))
        all_details.append("─" * 50)

        # CSV rows
        if cracks:
            for c in cracks:
                csv_rows.append({
                    "filename"   : fpath.name,
                    "gsd_mm_px"  : gsd,
                    **c
                })
        else:
            csv_rows.append({"filename": fpath.name, "gsd_mm_px": gsd,
                             "crack_id": 0, "confidence": "—",
                             "box_x1": "—", "box_y1": "—", "box_x2": "—", "box_y2": "—",
                             "width_px": 0, "height_px": 0, "diagonal_px": 0,
                             "length_mm": 0, "length_cm": 0, "length_m": 0})

    # Write CSV
    csv_path = str(out_dir / "results.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    n_imgs   = len(files)
    summary  = f"Processed {n_imgs} image(s)   |   Total cracks: {total_cracks}   |   Avg per image: {total_cracks/max(n_imgs,1):.2f}"
    details  = "\n".join(all_details)
    return preview_img, summary, details, csv_path


# ────────────────────────────────────────────────────────────────────────────
# Tab 3 — Video file
# ────────────────────────────────────────────────────────────────────────────

def detect_video(video_path, conf, altitude, focal, sensor, progress=gr.Progress()):
    if video_path is None:
        return None, "Upload a video first.", ""

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gsd   = compute_gsd(altitude, focal, sensor, W)

    out_path = str(Path(tempfile.mkdtemp()) / "result.mp4")
    writer   = cv2.VideoWriter(out_path,
                               cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    frame_idx    = 0
    total_cracks = 0
    max_in_frame = 0
    max_frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, cracks = annotate_frame(frame, conf, gsd)
        writer.write(annotated)
        n = len(cracks)
        total_cracks += n
        if n > max_in_frame:
            max_in_frame = n
            max_frame_id = frame_idx
        frame_idx += 1
        if total > 0:
            progress(frame_idx / total, desc=f"Frame {frame_idx}/{total}")

    cap.release()
    writer.release()

    summary = (
        f"Frames processed : {frame_idx}\n"
        f"Total detections : {total_cracks}\n"
        f"Avg per frame    : {total_cracks/max(frame_idx,1):.2f}\n"
        f"Busiest frame    : #{max_frame_id} ({max_in_frame} cracks)\n"
        f"GSD              : {gsd:.5f} mm/px"
    )
    return out_path, summary, ""


# ────────────────────────────────────────────────────────────────────────────
# Tab 4 — Stream URL / webcam snapshot
# ────────────────────────────────────────────────────────────────────────────

def detect_stream_snapshot(url, conf, altitude, focal, sensor):
    src = url.strip() if url.strip() else 0
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    if not cap.isOpened():
        return None, "Cannot open stream. Check URL/connection.", ""
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, "Connected but could not read frame.", ""

    H, W = frame.shape[:2]
    gsd  = compute_gsd(altitude, focal, sensor, W)
    annotated, cracks = annotate_frame(frame, conf, gsd)
    out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    summary = f"Cracks: {len(cracks)}   |   {W}×{H}px   |   GSD: {gsd:.5f} mm/px"
    details = format_crack_details(cracks, gsd)
    return Image.fromarray(out_rgb), summary, details


def detect_live_frame(frame_rgb, conf, altitude, focal, sensor):
    if frame_rgb is None:
        return None, "", ""
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    H, W  = frame.shape[:2]
    gsd   = compute_gsd(altitude, focal, sensor, W)
    annotated, cracks = annotate_frame(frame, conf, gsd)
    out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    summary = f"Cracks: {len(cracks)}   |   GSD: {gsd:.5f} mm/px"
    details = format_crack_details(cracks, gsd)
    return out_rgb, summary, details


# ────────────────────────────────────────────────────────────────────────────
# Shared drone settings block (reused across tabs)
# ────────────────────────────────────────────────────────────────────────────

def drone_settings():
    with gr.Accordion("Drone / camera settings (for real-world length)", open=False):
        gr.Markdown(
            "These values are used to compute GSD (Ground Sampling Distance) "
            "and convert pixel measurements to mm/cm/m. "
            "Leave defaults if you only need pixel-level results."
        )
        with gr.Row():
            altitude = gr.Number(value=10.0,  label="Altitude (m)",         minimum=0.1)
            focal    = gr.Number(value=4.5,   label="Focal length (mm)",    minimum=0.1)
            sensor   = gr.Number(value=6.17,  label="Sensor width (mm)",    minimum=0.1)
    return altitude, focal, sensor


# ────────────────────────────────────────────────────────────────────────────
# Build UI
# ────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Crack Detector") as app:

    gr.Markdown("# Crack Detection System")
    gr.Markdown(
        "Detect cracks in drone images/video and measure their real-world length. "
        "Supports single image, folder batch, video files, and live streams."
    )

    conf_slider = gr.Slider(
        minimum=0.10, maximum=0.90, value=0.25, step=0.05,
        label="Confidence threshold"
    )

    # ── Tab 1: Single image ───────────────────────────────────────────────
    with gr.Tab("Single image"):
        altitude1, focal1, sensor1 = drone_settings()
        with gr.Row():
            img_in  = gr.Image(type="pil", label="Upload image")
            img_out = gr.Image(type="pil", label="Annotated result")
        img_summary = gr.Textbox(label="Summary",           interactive=False)
        img_details = gr.Textbox(label="Full measurements", interactive=False,
                                 lines=20)
        gr.Button("Detect & measure", variant="primary").click(
            fn      = detect_single_image,
            inputs  = [img_in, conf_slider, altitude1, focal1, sensor1],
            outputs = [img_out, img_summary, img_details]
        )

    # ── Tab 2: Folder of images ───────────────────────────────────────────
    with gr.Tab("Folder of images"):
        altitude2, focal2, sensor2 = drone_settings()
        folder_in   = gr.File(label="Upload images (select multiple)",
                              file_count="multiple",
                              file_types=["image"])
        gr.Button("Detect all images", variant="primary").click(
            fn      = detect_folder,
            inputs  = [folder_in, conf_slider, altitude2, focal2, sensor2],
            outputs = [
                gr.Image(type="pil",  label="Preview (first image)"),
                gr.Textbox(label="Batch summary",        interactive=False),
                gr.Textbox(label="Per-image details",    interactive=False, lines=25),
                gr.File(  label="Download results CSV")
            ]
        )

    # ── Tab 3: Video file ─────────────────────────────────────────────────
    with gr.Tab("Video file"):
        altitude3, focal3, sensor3 = drone_settings()
        with gr.Row():
            vid_in  = gr.Video(label="Upload video")
            vid_out = gr.Video(label="Annotated output")
        vid_summary = gr.Textbox(label="Video statistics", interactive=False, lines=6)
        gr.Button("Detect cracks in video", variant="primary").click(
            fn      = detect_video,
            inputs  = [vid_in, conf_slider, altitude3, focal3, sensor3],
            outputs = [vid_out, vid_summary, gr.Textbox(visible=False)]
        )

    # ── Tab 4: Live webcam ────────────────────────────────────────────────
    with gr.Tab("Live webcam"):
        altitude4, focal4, sensor4 = drone_settings()
        with gr.Row():
            live_in  = gr.Image(sources=["webcam"], streaming=True,
                                label="Webcam input")
            live_out = gr.Image(label="Detected output")
        live_summary = gr.Textbox(label="Summary",  interactive=False)
        live_details = gr.Textbox(label="Measurements", interactive=False, lines=15)
        live_in.stream(
            fn      = detect_live_frame,
            inputs  = [live_in, conf_slider, altitude4, focal4, sensor4],
            outputs = [live_out, live_summary, live_details]
        )

    # ── Tab 5: RTSP / HTTP stream snapshot ───────────────────────────────
    with gr.Tab("Stream URL"):
        altitude5, focal5, sensor5 = drone_settings()
        gr.Markdown(
            "Enter an RTSP or HTTP stream URL. "
            "Click the button to grab the latest frame and run detection."
        )
        url_in       = gr.Textbox(label="Stream URL",
                                  placeholder="rtsp://192.168.10.1/live  or  0 for webcam")
        with gr.Row():
            url_out  = gr.Image(type="pil", label="Detected frame")
        url_summary  = gr.Textbox(label="Summary",      interactive=False)
        url_details  = gr.Textbox(label="Measurements", interactive=False, lines=15)
        gr.Button("Grab frame & detect", variant="primary").click(
            fn      = detect_stream_snapshot,
            inputs  = [url_in, conf_slider, altitude5, focal5, sensor5],
            outputs = [url_out, url_summary, url_details]
        )


if __name__ == "__main__":
    app.launch(
        server_name = "127.0.0.1",
        server_port = 7860,
        share       = False,
    )
