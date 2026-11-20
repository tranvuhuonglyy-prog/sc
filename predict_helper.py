
import base64
import io
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "damage_detector.pt"
yolo_model = YOLO(str(MODEL_PATH))


def predict_damage(img):
    img = img.convert("RGB")

    results = yolo_model.predict(
    source=img,
    conf=0.05,
    imgsz=640,
    verbose=False
)

    result = results[0]
    boxes = result.boxes

    damage_detected = False
    damage_type = "No damage detected"
    confidence_score = 0.0
    bboxes = []

    if boxes is not None and len(boxes) > 0:
        damage_detected = True

        best_index = int(boxes.conf.argmax().item())
        best_box = boxes[best_index]

        class_id = int(best_box.cls.item())
        confidence_score = float(best_box.conf.item())
        damage_type = result.names[class_id]

        for box in boxes:
            coords = box.xyxy[0].tolist()
            cls_id = int(box.cls.item())

            bboxes.append({
                "bbox": [float(x) for x in coords],
                "class": result.names[cls_id],
                "confidence": float(box.conf.item())
            })

    # Vẽ vùng hư hỏng
    annotated_array = result.plot()
    annotated_image = Image.fromarray(
        annotated_array[..., ::-1]
    )

    # Chuyển ảnh sang Base64
    buffered = io.BytesIO()
    annotated_image.save(buffered, format="JPEG")

    img_str = base64.b64encode(
        buffered.getvalue()
    ).decode("utf-8")

    return {
        "damage_detected": damage_detected,
        "damage_type": damage_type,
        "confidence": confidence_score,
        "annotated_image": img_str,
        "bboxes": bboxes
    }
