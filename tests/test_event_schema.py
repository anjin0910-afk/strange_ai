import unittest

from messaging.event_schema import build_safety_event


class EventSchemaTest(unittest.TestCase):
    def test_build_safety_event_contains_backend_friendly_fields(self):
        event = build_safety_event(
            event_type="fall_detected",
            camera_id="cam_01",
            severity="HIGH",
            message="쓰러짐 의심 상황이 감지되었습니다.",
            track_id=1,
            metadata={"rule_score": 0.91},
            timestamp="2026-05-26T10:00:00Z",
        )

        self.assertEqual(event["type"], "fall_detected")
        self.assertEqual(event["camera_id"], "cam_01")
        self.assertEqual(event["source"], "edge-ai")
        self.assertEqual(event["track_id"], 1)
        self.assertEqual(event["metadata"]["rule_score"], 0.91)


if __name__ == "__main__":
    unittest.main()
