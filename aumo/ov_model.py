import openvino as ov
import numpy as np
import cv2
from .model import BaseModel

class OVModel(BaseModel):
    def __init__(self, model_path: str, device: str = "CPU"):
        self.model_path = model_path
        self.device = device
        self.core = ov.Core()
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None

    def load(self):
        model = self.core.read_model(self.model_path)
        self.compiled_model = self.core.compile_model(model, self.device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        print(f"[OpenVINO] 모델 로드 완료: {self.model_path}")

    def predict(self, inputs):
        if self.compiled_model is None:
            raise RuntimeError("load()를 먼저 호출하세요.")
        
        # GPU 텐서 -> CPU 텐서 -> numpy 배열 변환
        if hasattr(inputs, 'device') and str(inputs.device).startswith('cuda'):
            inputs = inputs.cpu().numpy()
        elif hasattr(inputs, 'numpy'):
            inputs = inputs.numpy()
            
        results = self.compiled_model(inputs)[self.output_layer]
        return results

class OVClassificationModel(OVModel):
    def predict(self, inputs):
        logits = super().predict(inputs)
        probs = logits.squeeze()
        pred_class = int(np.argmax(probs))
        return pred_class, probs

class OVDetectionModel(OVModel):
    def predict(self, inputs, score_threshold=0.25):
        output = super().predict(inputs)  # (1, 84, 8400)

        # reshape
        output = np.squeeze(output, axis=0).T

        boxes = []
        scores = []
        cls_ids = []

        for det in output:
            x1, y1, x2, y2 = det[:4]
            obj_conf = det[4]
            cls_conf = det[5:]  # (num_classes,)

            cls_id = np.argmax(cls_conf)
            cls_score = cls_conf[cls_id]
            score = obj_conf * cls_score

            if score > score_threshold:
                boxes.append(np.array([x1, y1, x2, y2]))
                scores.append(score)
                cls_ids.append(cls_id)

        # return boxes
        draw_detections(inputs, boxes, scores, cls_ids)
