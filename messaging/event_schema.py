from datetime import datetime, timezone


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_safety_event(
    event_type,
    camera_id,
    severity,
    message,
    source="edge-ai",
    track_id=None,
    metadata=None,
    timestamp=None,
):
    event = {
        "type": event_type,
        "camera_id": camera_id,
        "timestamp": timestamp or utc_now_iso(),
        "severity": severity,
        "message": message,
        "source": source,
    }
    if track_id is not None:
        event["track_id"] = int(track_id)
    if metadata is not None:
        event["metadata"] = metadata
    return event
