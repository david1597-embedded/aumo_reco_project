import openvino as ov
import numpy as np
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
        results = self.compiled_model(inputs)[self.output_layer]
        return results

class OVClassificationModel(OVModel):
    def predict(self, inputs):
        logits = super().predict(inputs)
        probs = logits.squeeze()
        pred_class = int(np.argmax(probs))
        return pred_class, probs

class OVDetectionModel(OVModel):
    def predict(self, inputs, score_threshold=0.5):
        output = super().predict(inputs)
        boxes = []
        for det in output:
            x1, y1, x2, y2, score, cls = det
            if score > score_threshold:
                boxes.append({
                    "box": [x1, y1, x2, y2],
                    "score": score,
                    "class": int(cls)
                })
        return boxes
