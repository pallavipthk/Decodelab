"""
DecodeLabs - Project 4: Image / Text Recognition (Advanced, Single File)

Combines both execution paths from the brief into one script:
  --mode ocr     -> OCR pipeline (pytesseract)
  --mode detect  -> Object detection pipeline (MobileNet-SSD)

Both paths share the same architecture:
  INPUT (raw image) -> PROCESS (pre-processing + pre-trained model)
  -> OUTPUT (confidence-gated, visually annotated result)

Usage:
    python3 project4_advanced.py --mode ocr --image sample_document.png
    python3 project4_advanced.py --mode detect --image sample_street_scene.jpg

Requires (same folder):
    MobileNetSSD_deploy.prototxt
    MobileNetSSD_deploy.caffemodel
"""

import argparse
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

CONFIDENCE_THRESHOLD = 80  # shared "80% gate" standard for both paths

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]


# ---------------------------------------------------------------------
# PATH 1: OCR
# ---------------------------------------------------------------------
def deskew(gray):
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


def run_ocr(image_path, out_path="output.png"):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    straightened = deskew(blurred)
    _, binary = cv2.threshold(straightened, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    data = pytesseract.image_to_data(binary, config="--psm 11", output_type=Output.DICT)

    accepted_text, confs = [], []
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if word.strip() and conf >= CONFIDENCE_THRESHOLD:
            accepted_text.append(word)
            confs.append(conf)
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(image, f"{word} ({conf:.0f}%)", (x, max(y - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    cv2.imwrite(out_path, image)
    avg_conf = sum(confs) / len(confs) if confs else 0
    print(f"[OCR] Words passing {CONFIDENCE_THRESHOLD}% gate: {len(accepted_text)}")
    print(f"[OCR] Average confidence: {avg_conf:.1f}%")
    print("[OCR] Text:", " ".join(accepted_text) or "(none passed the gate)")
    print(f"[OCR] Annotated image saved to {out_path}")


# ---------------------------------------------------------------------
# PATH 2: OBJECT DETECTION
# ---------------------------------------------------------------------
def run_detection(image_path, out_path="output.png",
                   prototxt="MobileNetSSD_deploy.prototxt",
                   weights="MobileNetSSD_deploy.caffemodel"):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    net = cv2.dnn.readNetFromCaffe(prototxt, weights)
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5))
    net.setInput(blob)
    detections = net.forward()

    accepted = 0
    confs = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= CONFIDENCE_THRESHOLD / 100:
            class_id = int(detections[0, 0, i, 1])
            label = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"
            box = (detections[0, 0, i, 3:7] * [w, h, w, h]).astype(int)
            (sx, sy, ex, ey) = box
            cv2.rectangle(image, (sx, sy), (ex, ey), (0, 200, 0), 2)
            cv2.putText(image, f"{label}: {confidence*100:.0f}%", (sx, max(sy - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            accepted += 1
            confs.append(confidence * 100)
            print(f"[DETECT] {label} — {confidence*100:.1f}%")

    cv2.imwrite(out_path, image)
    avg_conf = sum(confs) / len(confs) if confs else 0
    print(f"[DETECT] Detections passing {CONFIDENCE_THRESHOLD}% gate: {accepted}")
    print(f"[DETECT] Average confidence: {avg_conf:.1f}%")
    print(f"[DETECT] Annotated image saved to {out_path}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ocr", "detect"], required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="output.png")
    args = parser.parse_args()

    if args.mode == "ocr":
        run_ocr(args.image, args.out)
    else:
        run_detection(args.image, args.out)
