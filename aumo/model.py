import os
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
from abc import ABC, abstractmethod
from .config import MODEL_PATH, ONNX_PATH, XML_PATH, DETECT_IMAGE_SIZE, CLASSIFY_IMAGE_SIZE
from .utils import ensure_dirs, export_model, convert_model

class BaseModel(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, inputs):
        pass


class YOLOModel(BaseModel):
    def __init__(self, model_name: str, imgsz: int = DETECT_IMAGE_SIZE):
        self.model_name = model_name
        self.imgsz = imgsz
        ensure_dirs(MODEL_PATH, ONNX_PATH, XML_PATH)
        self.model = None

    def load(self):
        model_path = os.path.join(MODEL_PATH, f"{self.model_name}.pt")
        if not os.path.exists(model_path):
            if os.path.exists(f"{self.model_name}.pt"):
                os.rename(f"{self.model_name}.pt", model_path)
            else:
                raise FileNotFoundError(f"모델 파일 {self.model_name}.pt를 찾을 수 없습니다.")
        self.model = YOLO(model_path)
        print(f"YOLO 모델 로드 완료: {self.model.model_name}")

    def predict(self, inputs):
        if self.model is None:
            raise RuntimeError("load()를 먼저 실행하세요.")
        return self.model(inputs)

    def export(self, output_path: str = None):
        if self.model is None:
            raise RuntimeError("load()를 먼저 실행하세요.")
        onnx_file = output_path or os.path.join(ONNX_PATH, f"{self.model_name}.onnx")
        if not os.path.exists(onnx_file):
            print(f"ONNX export 중... {onnx_file}")
            export_model(self.model, 'yolo', imgsz=self.imgsz, output=onnx_file)
        else:
            print(f"ONNX 파일 이미 존재: {onnx_file}")
        return onnx_file

    def convert(self, output_path: str = None):
        xml_file = output_path or os.path.join(XML_PATH, f"{self.model_name}.xml")
        onnx_file = os.path.join(ONNX_PATH, f"{self.model_name}.onnx")
        if not os.path.exists(xml_file):
            print(f"OpenVINO 변환 중... {xml_file}")
            convert_model(onnx_file, output=xml_file)
        else:
            print(f"OpenVINO 파일 이미 존재: {xml_file}")
        return xml_file


class ResNetModel(BaseModel):
    def __init__(self, num_classes: int = 13, device: str = "cpu", model_path: str = None, pretrained: bool =False):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.to(self.device)

    def load(self):
        if self.model_path is None:
            print("모델 경로가 지정되지 않아 사전학습 모델을 사용합니다.")
            self.model.eval()
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print(f"ResNet 모델 로드 완료: {self.model_path}")

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"ResNet 모델 저장 완료: {path}")

    def predict(self, inputs):
        self.model.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs

    def export(self, output_path: str = "resnet50.onnx"):
        export_model(self.model, 'pth', CLASSIFY_IMAGE_SIZE, output_path)
        return output_path

    def convert(self, output_path: str = None):
        onnx_path = output_path or "resnet50.onnx"
        xml_path = onnx_path.replace(".onnx", ".xml")
        convert_model(onnx_path, output=xml_path)
        print(f"OpenVINO 변환 완료: {xml_path}")
        return xml_path
