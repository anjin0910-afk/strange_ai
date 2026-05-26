import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import redis
from redis.exceptions import RedisError


EVENT_TYPES = [
    "fall_detected",
    "unconscious_detected",
    "violence_detected",
    "unauthorized_exit",
    "bed_fall_detected",
]

SEVERITIES = ["LOW", "MEDIUM", "HIGH"]
CAMERA_IDS = ["cam_01", "cam_02", "cam_03"]
EVENT_MESSAGES = {
    "fall_detected": "Fall detected from mock edge AI",
    "unconscious_detected": "Unconscious person detected from mock edge AI",
    "violence_detected": "Violence detected from mock edge AI",
    "unauthorized_exit": "Unauthorized exit detected from mock edge AI",
    "bed_fall_detected": "Bed fall detected from mock edge AI",
}


def get_env_int(name, default):
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got: {value}")


def build_event():
    event_type = random.choice(EVENT_TYPES)
    return {
        "type": event_type,
        "camera_id": random.choice(CAMERA_IDS),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "severity": random.choice(SEVERITIES),
        "message": EVENT_MESSAGES[event_type],
    }


def connect_redis(host, port):
    client = redis.Redis(host=host, port=port, decode_responses=True)
    client.ping()
    return client


def main():
    host = os.getenv("REDIS_HOST", "localhost")
    channel = os.getenv("REDIS_CHANNEL", "safety-events")

    try:
        port = get_env_int("REDIS_PORT", 6379)
        interval_seconds = get_env_int("PUBLISH_INTERVAL_SECONDS", 3)
    except ValueError as exc:
        print(f"[mock-edge-ai] Configuration error: {exc}", file=sys.stderr)
        return 1

    if interval_seconds <= 0:
        print(
            "[mock-edge-ai] Configuration error: PUBLISH_INTERVAL_SECONDS must be greater than 0",
            file=sys.stderr,
        )
        return 1

    try:
        redis_client = connect_redis(host, port)
    except RedisError as exc:
        print(
            f"[mock-edge-ai] Redis connection failed: host={host}, port={port}, error={exc}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[mock-edge-ai] Publishing mock safety events to Redis channel '{channel}' "
        f"at redis://{host}:{port} every {interval_seconds}s"
    )

    try:
        while True:
            event = build_event()
            payload = json.dumps(event, ensure_ascii=False)
            redis_client.publish(channel, payload)
            print(f"[mock-edge-ai] published: {payload}", flush=True)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n[mock-edge-ai] Stopped by user")
        return 0
    except RedisError as exc:
        print(f"[mock-edge-ai] Redis publish failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
