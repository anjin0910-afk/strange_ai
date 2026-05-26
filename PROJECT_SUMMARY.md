# Project Summary

## YOLO Pose Model Benchmark

This project includes a benchmarking pipeline for selecting a YOLO pose model for a smart safety monitoring system.

The benchmark compares these configured model groups on the same video inputs, image size, and device:

- YOLOv8n-pose (`yolo8n-pose.pt`, fallback `yolov8n-pose.pt`)
- YOLOv11n-pose (`yolo11n-pose.pt`)
- YOLO26n-pose (`yolo26n-pose.pt`)
- YOLOv8s-pose (`yolo8s-pose.pt`, fallback `yolov8s-pose.pt`)
- YOLOv11s-pose (`yolo11s-pose.pt`)
- YOLO26s-pose (`yolo26s-pose.pt`)

YOLO model names can vary by installed Ultralytics version. The benchmark tries each configured candidate in order. If a model cannot be loaded or downloaded, that model is marked as `FAILED` or `SKIPPED`, and the remaining models continue.

### Input Videos

Place benchmark videos in:

```bash
sample_videos/
```

Supported extensions are configured in:

```bash
configs/model_benchmark.yaml
```

If the folder is empty, the script prints a guidance message and writes skipped result reports.

### Run

```bash
python benchmark/benchmark_models.py --video-dir sample_videos --imgsz 640 --device 0
```

Use automatic device selection:

```bash
python benchmark/benchmark_models.py --video-dir sample_videos --imgsz 640 --device auto
```

If CUDA is unavailable, the benchmark falls back to CPU and records `device=cpu`.

### Results

The benchmark writes:

```bash
benchmark/results/model_benchmark.csv
benchmark/results/model_benchmark.md
```

Recorded metrics:

- `model_name`
- `video_name`
- `device`
- `imgsz`
- `total_frames`
- `processed_frames`
- `avg_fps`
- `avg_latency_ms`
- `p95_latency_ms`
- `gpu_memory_mb`
- `avg_person_confidence`
- `avg_keypoint_confidence`
- `keypoint_missing_rate`
- `fall_candidate_count`
- `error_or_status`

GPU memory is measured with `torch.cuda.max_memory_allocated()` when CUDA is used. The script performs warm-up inference before measured frames, does not save raw frames, and does not persist private or sensitive video data.

### Fall Rule

The optional fall candidate rule uses COCO pose shoulder and hip keypoints. A person is counted as a `FALL_DOWN` candidate when the shoulder-to-hip torso vector is much more horizontal than vertical and required keypoints are above the configured confidence threshold.

This is a simple screening rule for model comparison, not a final safety decision engine.
