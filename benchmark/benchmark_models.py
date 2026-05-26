import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "model_benchmark.yaml"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"

COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark YOLO pose models on sample videos.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to benchmark YAML config.")
    parser.add_argument("--video-dir", default="sample_videos", help="Directory containing input videos.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="auto", help="CUDA device index, 'cpu', or 'auto'.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional maximum measured frames per video.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory for CSV/Markdown outputs.")
    return parser.parse_args()


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def resolve_device(requested):
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if str(requested).lower() == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return str(requested)
    print("CUDA is not available. Falling back to CPU and recording device=cpu.")
    return "cpu"


def is_cuda_device(device):
    return str(device).lower() != "cpu" and torch.cuda.is_available()


def list_videos(video_dir, extensions):
    root = Path(video_dir)
    if not root.exists():
        return []
    normalized = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in normalized)


def model_display_name(model_entry):
    return model_entry.get("name") or ", ".join(model_entry.get("candidates", [])) or "unknown"


def load_first_available_model(model_entry):
    errors = []
    for candidate in model_entry.get("candidates", []):
        try:
            return candidate, YOLO(candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    detail = " | ".join(errors) if errors else "no candidates configured"
    raise RuntimeError(f"No loadable model candidate for {model_display_name(model_entry)}. Tried: {detail}")


def mean_or_zero(values):
    return float(statistics.fmean(values)) if values else 0.0


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return float(ordered[index])


def is_fall_candidate(kp_xy, kp_conf, threshold):
    required = [COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER, COCO_LEFT_HIP, COCO_RIGHT_HIP]
    if any(idx >= kp_conf.numel() or float(kp_conf[idx]) < threshold for idx in required):
        return False

    left_shoulder = kp_xy[COCO_LEFT_SHOULDER]
    right_shoulder = kp_xy[COCO_RIGHT_SHOULDER]
    left_hip = kp_xy[COCO_LEFT_HIP]
    right_hip = kp_xy[COCO_RIGHT_HIP]
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2

    torso_dx = abs(float(shoulder_center[0] - hip_center[0]))
    torso_dy = abs(float(shoulder_center[1] - hip_center[1]))
    return torso_dx > 0 and torso_dx / max(torso_dy, 1.0) >= 1.3


def extract_pose_metrics(result, keypoint_conf_threshold):
    person_confidences = []
    keypoint_confidences = []
    missing_keypoints = 0
    total_keypoints = 0
    fall_candidates = 0

    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.conf is not None:
        person_confidences = boxes.conf.detach().float().cpu().tolist()

    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.conf is None:
        return person_confidences, keypoint_confidences, missing_keypoints, total_keypoints, fall_candidates

    conf_tensor = keypoints.conf.detach().float().cpu()
    xy_tensor = keypoints.xy.detach().float().cpu() if keypoints.xy is not None else None

    for person_idx in range(conf_tensor.shape[0]):
        kp_conf = conf_tensor[person_idx]
        total_keypoints += int(kp_conf.numel())
        missing_keypoints += int((kp_conf < keypoint_conf_threshold).sum().item())
        visible = kp_conf[kp_conf >= keypoint_conf_threshold]
        if visible.numel() > 0:
            keypoint_confidences.extend(visible.tolist())
        if xy_tensor is not None and is_fall_candidate(xy_tensor[person_idx], kp_conf, keypoint_conf_threshold):
            fall_candidates += 1

    return person_confidences, keypoint_confidences, missing_keypoints, total_keypoints, fall_candidates


def run_warmup(model, video_path, warmup_frames, imgsz, device):
    if warmup_frames <= 0:
        return
    cap = cv2.VideoCapture(str(video_path))
    try:
        for _ in range(warmup_frames):
            ok, frame = cap.read()
            if not ok:
                break
            model.predict(frame, imgsz=imgsz, device=device, verbose=False)
    finally:
        cap.release()


def benchmark_video(model, model_name, video_path, imgsz, device, config, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    threshold = float(config.get("keypoint_conf_threshold", 0.3))
    cuda = is_cuda_device(device)
    latencies = []
    person_confidences = []
    keypoint_confidences = []
    missing_keypoints = 0
    total_keypoints = 0
    fall_candidate_count = 0
    processed_frames = 0

    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()
    progress_total = total_frames if total_frames > 0 else None
    if max_frames > 0 and progress_total:
        progress_total = min(progress_total, max_frames)

    try:
        with tqdm(total=progress_total, desc=f"{model_name} | {video_path.name}", unit="frame") as progress:
            while True:
                if max_frames > 0 and processed_frames >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break

                if cuda:
                    torch.cuda.synchronize()
                frame_started = time.perf_counter()
                results = model.predict(frame, imgsz=imgsz, device=device, verbose=False)
                if cuda:
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - frame_started) * 1000.0)

                for result in results:
                    person_conf, keypoint_conf, missing, total, falls = extract_pose_metrics(result, threshold)
                    person_confidences.extend(person_conf)
                    keypoint_confidences.extend(keypoint_conf)
                    missing_keypoints += missing
                    total_keypoints += total
                    fall_candidate_count += falls

                processed_frames += 1
                progress.update(1)
    finally:
        cap.release()

    elapsed = time.perf_counter() - started
    gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if cuda else 0.0
    missing_rate = missing_keypoints / total_keypoints if total_keypoints else 0.0

    return {
        "model_name": model_name,
        "video_name": video_path.name,
        "device": "cuda" if cuda else "cpu",
        "imgsz": imgsz,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "avg_fps": processed_frames / elapsed if elapsed > 0 else 0.0,
        "avg_latency_ms": mean_or_zero(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "gpu_memory_mb": gpu_memory_mb,
        "avg_person_confidence": mean_or_zero(person_confidences),
        "avg_keypoint_confidence": mean_or_zero(keypoint_confidences),
        "keypoint_missing_rate": missing_rate,
        "fall_candidate_count": fall_candidate_count,
        "error_or_status": "OK",
    }


def failure_row(model_name, video_name, device, imgsz, status):
    return {
        "model_name": model_name,
        "video_name": video_name,
        "device": "cpu" if str(device).lower() == "cpu" else "cuda",
        "imgsz": imgsz,
        "total_frames": 0,
        "processed_frames": 0,
        "avg_fps": 0.0,
        "avg_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "gpu_memory_mb": 0.0,
        "avg_person_confidence": 0.0,
        "avg_keypoint_confidence": 0.0,
        "keypoint_missing_rate": 0.0,
        "fall_candidate_count": 0,
        "error_or_status": status,
    }


def round_rows(rows):
    numeric_columns = [
        "avg_fps",
        "avg_latency_ms",
        "p95_latency_ms",
        "gpu_memory_mb",
        "avg_person_confidence",
        "avg_keypoint_confidence",
        "keypoint_missing_rate",
    ]
    rounded = []
    for row in rows:
        next_row = dict(row)
        for column in numeric_columns:
            next_row[column] = round(float(next_row.get(column, 0.0)), 4)
        rounded.append(next_row)
    return rounded


def rows_to_markdown(rows, columns):
    rounded_rows = round_rows(rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rounded_rows]
    return "\n".join([header, separator, *body])


def build_model_summary(df):
    ok = df[df["error_or_status"] == "OK"].copy()
    if ok.empty:
        return "\n\n## Model Summary\n\nNo successful benchmark rows were produced.\n"

    grouped = ok.groupby("model_name", as_index=False).agg(
        avg_fps=("avg_fps", "mean"),
        avg_latency_ms=("avg_latency_ms", "mean"),
        p95_latency_ms=("p95_latency_ms", "mean"),
        gpu_memory_mb=("gpu_memory_mb", "max"),
        avg_keypoint_confidence=("avg_keypoint_confidence", "mean"),
        keypoint_missing_rate=("keypoint_missing_rate", "mean"),
        fall_candidate_count=("fall_candidate_count", "sum"),
    )

    for column in grouped.columns:
        if column != "model_name":
            grouped[column] = grouped[column].astype(float).round(4)

    fastest = grouped.sort_values("avg_fps", ascending=False).iloc[0]["model_name"]
    lowest_latency = grouped.sort_values("avg_latency_ms", ascending=True).iloc[0]["model_name"]
    best_keypoints = grouped.sort_values("avg_keypoint_confidence", ascending=False).iloc[0]["model_name"]
    lowest_missing = grouped.sort_values("keypoint_missing_rate", ascending=True).iloc[0]["model_name"]

    lines = [
        "\n\n## Model Summary",
        "",
        rows_to_markdown(grouped.to_dict(orient="records"), list(grouped.columns)),
        "",
        "## Notes",
        "",
        f"- Highest average FPS: {fastest}",
        f"- Lowest average latency: {lowest_latency}",
        f"- Highest average keypoint confidence: {best_keypoints}",
        f"- Lowest keypoint missing rate: {lowest_missing}",
        "- Lower latency/FPS models are usually better for real-time control rooms; higher confidence and lower missing rate are usually better for fall-rule stability.",
    ]
    return "\n".join(lines) + "\n"


def save_results(rows, results_dir):
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_benchmark.csv"
    md_path = output_dir / "model_benchmark.md"

    fieldnames = [
        "model_name",
        "video_name",
        "device",
        "imgsz",
        "total_frames",
        "processed_frames",
        "avg_fps",
        "avg_latency_ms",
        "p95_latency_ms",
        "gpu_memory_mb",
        "avg_person_confidence",
        "avg_keypoint_confidence",
        "keypoint_missing_rate",
        "fall_candidate_count",
        "error_or_status",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows, columns=fieldnames)
    md_content = "# YOLO Pose Model Benchmark\n\n"
    md_content += rows_to_markdown(df.to_dict(orient="records"), fieldnames)
    md_content += build_model_summary(df)
    md_path.write_text(md_content, encoding="utf-8")
    return csv_path, md_path


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(str(args.device))
    videos = list_videos(args.video_dir, config.get("video_extensions", [".mp4", ".avi", ".mov", ".mkv"]))

    if not videos:
        print(f"No videos found in {args.video_dir}. Add videos to this folder and run the benchmark again.")
        rows = [
            failure_row(
                model_display_name(model_entry),
                "(no video)",
                device,
                args.imgsz,
                "SKIPPED: sample_videos folder is empty or has no supported video files",
            )
            for model_entry in config.get("models", [])
        ]
        csv_path, md_path = save_results(rows, args.results_dir)
        print(f"Saved empty benchmark report: {csv_path}")
        print(f"Saved empty benchmark report: {md_path}")
        return 0

    rows = []
    for model_entry in config.get("models", []):
        configured_name = model_display_name(model_entry)
        try:
            loaded_name, model = load_first_available_model(model_entry)
        except Exception as exc:
            status_prefix = "SKIPPED" if "yolo26" in configured_name.lower() else "FAILED"
            status = f"{status_prefix}: {exc}"
            print(status)
            for video_path in videos:
                rows.append(failure_row(configured_name, video_path.name, device, args.imgsz, status))
            continue

        model_name = model_entry.get("name", loaded_name)
        for video_path in videos:
            try:
                run_warmup(model, video_path, int(config.get("warmup_frames", 5)), args.imgsz, device)
                rows.append(benchmark_video(model, model_name, video_path, args.imgsz, device, config, args.max_frames))
            except Exception as exc:
                status = f"FAILED: {exc}"
                print(f"{model_name} | {video_path.name}: {status}")
                rows.append(failure_row(model_name, video_path.name, device, args.imgsz, status))

    csv_path, md_path = save_results(rows, args.results_dir)
    print(f"Saved CSV results: {csv_path}")
    print(f"Saved Markdown results: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
