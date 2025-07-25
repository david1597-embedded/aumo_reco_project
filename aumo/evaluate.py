# evaluate.py
import cv2
import time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score

class Evaluator:
    def __init__(self, model, dataset):
        """
        model: .predict(image) -> List[[x1, y1, x2, y2, conf, class_id]]
        dataset: List of tuples (image_path, gt_boxes), 
                 gt_boxes: List[[x1, y1, x2, y2, class_id]]
        """
        self.model = model
        self.dataset = dataset

    def benchmark(self, num_runs=100):
        times = []
        for i, (img_path, _) in enumerate(self.dataset[:num_runs]):
            img = cv2.imread(img_path)
            if img is None:
                continue
            start = time.time()
            _ = self.model.predict(img, verbose=False)
            end = time.time()
            times.append(end - start)

        avg_time = np.mean(times)
        fps = 1 / avg_time if avg_time > 0 else 0
        print(f"[Benchmark] 평균 추론 시간: {avg_time*1000:.2f} ms | FPS: {fps:.2f}")
        return fps

    def evaluate_map(self, iou_threshold=0.5):
        all_detections = []
        all_annotations = []

        for img_path, gt_boxes in self.dataset:
            img = cv2.imread(img_path)
            if img is None:
                continue
            results = self.model.predict(img, verbose=False)
            results = results[0]

            boxes = results.boxes

            preds = []
            for i in range(len(boxes)):
                box = boxes[i]
                xyxy = box.xyxy.cpu().numpy().flatten()  # shape (4,)
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                preds.append([*xyxy, conf, cls_id])

            all_detections.extend(preds)
            all_annotations.extend(gt_boxes)

        # 간단한 mAP 계산 (class 무시, binary 기준)
        y_true, y_scores = [], []
        for pred in all_detections:
            best_iou = 0
            matched = False
            for gt in all_annotations:
                iou = self._iou(pred[:4], gt[:4])
                if iou >= iou_threshold:
                    matched = True
                    break
            y_true.append(1 if matched else 0)
            y_scores.append(pred[4])  # conf

        if not y_true:
            print("정답 없음")
            return 0.0

        ap = average_precision_score(y_true, y_scores)
        print(f"[mAP] IoU ≥ {iou_threshold} 기준 AP: {ap:.4f}")
        return ap

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou
