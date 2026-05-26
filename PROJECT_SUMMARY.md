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

## Mock Edge AI MQTT Publisher

`mock_edge_ai.py` publishes random safety event JSON messages to the MQTT topic used by the local development pipeline.

```text
Python Edge AI -> MQTT Broker (Mosquitto) -> Spring Boot MQTT Subscriber -> WebSocket -> React Frontend
```

This script is a mock publisher for integration testing before OpenCV, YOLOv8-Pose, and RTSP inference are connected.

### MQTT Environment

Set these environment variables when you need values other than the defaults:

```text
MQTT_HOST=localhost
MQTT_PORT=1883
MQTT_TOPIC=safety/events
MQTT_CLIENT_ID=edge-ai-mock-001
PUBLISH_INTERVAL_SECONDS=3
```

### Install

```bash
pip install -r requirements.txt
```

### Run

Make sure Mosquitto MQTT Broker is running, then start the mock publisher:

```bash
python mock_edge_ai.py
```

The script publishes events to:

```text
safety/events
```

Each published event is also printed to the console.

### Verify MQTT Publish

In another terminal, subscribe to the MQTT topic:

```bash
mosquitto_sub -h localhost -p 1883 -t safety/events
```

If `mosquitto_sub` is not installed locally, use the Mosquitto Docker container:

```bash
docker exec -it strange-mosquitto mosquitto_sub -h localhost -p 1883 -t safety/events
```

Expected event shape:

```json
{
  "type": "fall_detected",
  "camera_id": "cam_01",
  "timestamp": "2026-05-26T10:00:00Z",
  "severity": "HIGH",
  "message": "Fall detected from mock edge AI",
  "source": "edge-ai-mock"
}
```

### Scope

- This mock does not implement YOLO, OpenCV, or RTSP processing.
- It does not include real videos, personal data, API keys, or passwords.
- It can be replaced later by the real edge AI inference pipeline while keeping the MQTT topic contract.

## Edge AI Pipeline MVP

`main.py` provides the initial runtime shape for the real CCTV Edge AI server.

```text
CCTV / Sample Video
-> RTSP Stream
-> OpenCV RTSP Reader
-> Latest-frame Queue
-> YOLO Pose Detector or Mock Detector
-> Fall Rule Engine
-> MQTT safety/events
-> Spring Boot Backend
```

### Current Structure

```text
main.py
config.py
stream/frame_queue.py
stream/rtsp_reader.py
detector/mock_detector.py
detector/yolo_pose_detector.py
rules/fall_rule.py
messaging/event_schema.py
messaging/mqtt_publisher.py
tests/test_fall_rule.py
tests/test_event_schema.py
```

The frame queue is intentionally small and drops old frames so inference latency does not grow when processing is slower than the RTSP input.

### Pipeline Environment

```text
RTSP_URL=rtsp://localhost:8554/cam01
CAMERA_ID=cam_01
DETECTOR_MODE=mock
YOLO_MODEL=yolov8n-pose.pt
YOLO_DEVICE=auto
FRAME_QUEUE_SIZE=2
ALLOW_MOCK_FALLBACK=true
FALL_MIN_DURATION_SECONDS=1.5
FALL_DEBOUNCE_SECONDS=10
MAX_FRAMES=0
MQTT_HOST=localhost
MQTT_PORT=1883
MQTT_TOPIC=safety/events
MQTT_CLIENT_ID=edge-ai-001
```

### Run The Pipeline

Mock detector mode, no RTSP required:

```bash
python main.py --dry-run --once
```

Publish to MQTT with mock detector:

```bash
python main.py --once
```

Run with RTSP and YOLO pose model:

```bash
DETECTOR_MODE=yolo RTSP_URL=rtsp://localhost:8554/cam01 YOLO_MODEL=yolov8n-pose.pt python main.py
```

On Windows PowerShell:

```powershell
$env:DETECTOR_MODE="yolo"
$env:RTSP_URL="rtsp://localhost:8554/cam01"
$env:YOLO_MODEL="yolov8n-pose.pt"
python main.py
```

### MQTT Verification

Start Mosquitto in `strange_infra`, then subscribe:

```bash
mosquitto_sub -h localhost -p 1883 -t safety/events
```

Run the AI server:

```bash
python main.py --once
```

Expected event shape:

```json
{
  "type": "fall_detected",
  "camera_id": "cam_01",
  "timestamp": "2026-05-26T10:00:00Z",
  "severity": "HIGH",
  "message": "쓰러짐 의심 상황이 감지되었습니다.",
  "source": "edge-ai",
  "track_id": 1,
  "metadata": {
    "bbox": [100, 150, 280, 390],
    "confidence": 0.87,
    "rule_score": 0.91,
    "pose_state": "LYING",
    "model_name": "yolov8n-pose"
  }
}
```

### Rule Engine Notes

The first stabilized rule is `fall_detected`. It combines bbox aspect ratio, pose-horizontal signal, detector confidence, and a minimum duration threshold before emitting an event. Repeated events for the same track are debounced.

TODO:

- Add ByteTrack or another tracker for stable `track_id` across real streams.
- Expand rule modules for unconscious, bed fall, unauthorized exit, and violence detection.
- Add RTSP benchmark tooling in `benchmark/benchmark_rtsp.py`.
- Calibrate thresholds with real non-sensitive sample videos.
