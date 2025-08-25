#!/usr/bin/env python3
from pathlib import Path
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from app.core import preprocess_bgr_to_model, draw_detections

MODEL = Path("~/MasterArbeit/models/efficientdet-lite2.tflite").expanduser()
IMAGE = Path("~/MasterArbeit/data/test.jpg").expanduser()
OUT   = Path("~/MasterArbeit/out/test_out.jpg").expanduser()
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    interpreter = tflite.Interpreter(model_path=str(MODEL), num_threads=4)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()
    bgr = cv2.imread(str(IMAGE))
    if bgr is None: raise SystemExit(f"Image introuvable: {IMAGE}")
    x = preprocess_bgr_to_model(bgr, inp['shape'], inp['dtype'])
    interpreter.set_tensor(inp['index'], x)
    interpreter.invoke()
    boxes  = interpreter.get_tensor(out[0]['index'])[0]
    classes= interpreter.get_tensor(out[1]['index'])[0]
    scores = interpreter.get_tensor(out[2]['index'])[0]
    draw_detections(bgr, boxes, classes, scores, thr=0.4)
    cv2.imwrite(str(OUT), bgr)
    print(f"✅ Résultat sauvegardé: {OUT}")

if __name__ == "__main__":
    main()
