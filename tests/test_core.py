import numpy as np
import cv2
from app.core import preprocess_bgr_to_model, draw_detections

def test_preprocess_shape_uint8():
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    x = preprocess_bgr_to_model(dummy, input_shape=(1, 448, 448, 3), dtype=np.uint8)
    assert x.shape == (1, 448, 448, 3)
    assert x.dtype == np.uint8

def test_draw_no_crash():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    boxes = np.array([[0.1, 0.1, 0.3, 0.3]], dtype=np.float32)
    classes = np.array([0], dtype=np.float32)
    scores = np.array([0.95], dtype=np.float32)
    draw_detections(img, boxes, classes, scores, thr=0.5)
