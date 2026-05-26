def draw_overlay(frame, boxes, prediction, frame_idx):
    import cv2

    output = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, f"person {box['score']:.2f}", (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if prediction:
        label = prediction["label"]
        score = prediction["score"]
        text = f"[OK] {label} detected score={score:.2f} frame={frame_idx}"
        cv2.putText(output, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(output, f"frame={frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return output
