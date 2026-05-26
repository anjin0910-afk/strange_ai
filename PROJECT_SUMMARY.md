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

## Mock Edge AI Redis Publisher

`mock_edge_ai.py` publishes random safety event JSON messages to the Redis Pub/Sub channel used by the local development pipeline.

```text
Python Edge AI -> Redis Message Broker -> Spring Boot Backend -> React Frontend
```

This script is a mock publisher for integration testing before OpenCV, YOLOv8-Pose, and RTSP inference are connected.

### Redis Environment

Set these environment variables when you need values other than the defaults:

```text
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_CHANNEL=safety-events
PUBLISH_INTERVAL_SECONDS=3
```

### Install

```bash
pip install -r requirements.txt
```

### Run

Make sure Redis is running, then start the mock publisher:

```bash
python mock_edge_ai.py
```

The script publishes events to:

```text
safety-events
```

Each published event is also printed to the console.

### Verify Pub/Sub

In another terminal, subscribe to the Redis channel:

```bash
redis-cli SUBSCRIBE safety-events
```

If `redis-cli` is not installed locally, use the Redis Docker container:

```bash
docker exec -it strange-redis redis-cli SUBSCRIBE safety-events
```

Expected event shape:

```json
{
  "type": "fall_detected",
  "camera_id": "cam_01",
  "timestamp": "2026-05-26T10:00:00Z",
  "severity": "HIGH",
  "message": "Fall detected from mock edge AI"
}
```

### Scope

- This mock does not implement YOLO, OpenCV, or RTSP processing.
- It does not include real videos, personal data, API keys, or passwords.
- It can be replaced later by the real edge AI inference pipeline while keeping the Redis channel contract.
