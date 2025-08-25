#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, time, sys
from pathlib import Path
import numpy as np
import cv2
from app.core import preprocess_bgr_to_model, draw_detections, iter_mjpeg_bytes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Chemin du modèle .tflite")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--score", type=float, default=0.4)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--rpicam_cmd", default="rpicam-vid")
    ap.add_argument("--extra", default="")
    args = ap.parse_args()

    model_path = str(Path(args.model).expanduser())
    use_edgetpu = model_path.endswith("_edgetpu.tflite")
    backend = "CPU"

    if use_edgetpu:
        try:
            from pycoral.utils.edgetpu import make_interpreter
            from pycoral.adapters.common import input_size, set_input
            from pycoral.adapters.detect import get_objects
            interpreter = make_interpreter(model_path)
            interpreter.allocate_tensors()
            in_w, in_h = input_size(interpreter)
            backend = "EdgeTPU"
            def infer(frame_bgr, thr):
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
                set_input(interpreter, resized)
                interpreter.invoke()
                objs = get_objects(interpreter, score_threshold=thr)
                boxes, classes, scores = [], [], []
                for o in objs:
                    boxes.append([o.bbox.ymin, o.bbox.xmin, o.bbox.ymax, o.bbox.xmax])
                    classes.append(o.id); scores.append(o.score)
                return (np.array(boxes, np.float32) if boxes else np.zeros((0,4), np.float32),
                        np.array(classes, np.float32),
                        np.array(scores, np.float32))
        except Exception:
            print("⚠️ pycoral indisponible, retour CPU.")
            use_edgetpu = False

    if not use_edgetpu:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=args.threads)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()
        in_shape, in_dtype = inp["shape"], inp["dtype"]
        def infer(frame_bgr, thr):
            x = preprocess_bgr_to_model(frame_bgr, in_shape, in_dtype)
            interpreter.set_tensor(inp["index"], x)
            interpreter.invoke()
            boxes  = interpreter.get_tensor(out[0]["index"])[0]
            classes= interpreter.get_tensor(out[1]["index"])[0]
            scores = interpreter.get_tensor(out[2]["index"])[0]
            return boxes, classes, scores

    cmd = [args.rpicam_cmd, "-n", "--codec", "mjpeg",
           "--width", str(args.width), "--height", str(args.height),
           "--framerate", str(args.fps), "-t", "0", "-o", "-"]
    if args.extra.strip():
        cmd.extend(args.extra.strip().split())

    print("Backend:", backend)
    print("Lancement:", " ".join(cmd))
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    except FileNotFoundError:
        print("❌ rpicam-vid introuvable. Installe : sudo apt install -y rpicam-apps")
        sys.exit(1)

    prev = time.time(); fps = 0.0
    try:
        for jpg in iter_mjpeg_bytes(proc.stdout):
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None: continue
            boxes, classes, scores = infer(frame, args.score)
            draw_detections(frame, boxes, classes, scores, thr=args.score)
            now = time.time()
            fps = 0.9*fps + 0.1*(1.0/(now-prev)); prev = now
            cv2.putText(frame, f"EffDet-Lite2 [{backend}] {fps:.1f} FPS",
                        (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"EffDet-Lite2 [{backend}] {fps:.1f} FPS",
                        (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow("rpicam-vid (q pour quitter)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        try: proc.terminate()
        except Exception: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
