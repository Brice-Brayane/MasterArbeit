from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

COCO = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

def preprocess_bgr_to_model(bgr: np.ndarray, input_shape, dtype) -> np.ndarray:
    _, in_h, in_w, _ = input_shape
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    if dtype == np.float32:
        x = (resized.astype(np.float32) / 255.0)
    else:
        x = resized.astype(np.uint8)
    return np.expand_dims(x, 0)

def draw_detections(frame: np.ndarray, boxes, classes, scores, thr: float = 0.4) -> None:
    h, w = frame.shape[:2]
    for i, s in enumerate(scores):
        if s < thr: continue
        c = int(classes[i])
        y1, x1, y2, x2 = boxes[i]
        x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        label = COCO[c] if 0 <= c < len(COCO) else f"id_{c}"
        txt = f"{label} {s*100:.1f}%"
        cv2.putText(frame, txt, (x1, max(14, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (x1, max(14, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

def iter_mjpeg_bytes(stream):
    buf = bytearray(); SOI, EOI = b"\xff\xd8", b"\xff\xd9"
    while True:
        chunk = stream.read(4096)
        if not chunk: break
        buf.extend(chunk)
        while True:
            s = buf.find(SOI)
            if s < 0:
                if len(buf) > 1_000_000: del buf[:-2]
                break
            e = buf.find(EOI, s+2)
            if e < 0:
                if s > 0: del buf[:s]
                break
            jpg = bytes(buf[s:e+2]); del buf[:e+2]
            yield jpg
