import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
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
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(self.model_name)
            if os.path.exists(f"{self.model_name}.pt"):
                os.rename(f"{self.model_name}.pt", model_path)
            else:
                raise FileNotFoundError(f"모델 파일 {self.model_name}.pt를 찾을 수 없습니다.")
        print(f"YOLO 모델 로드 완료: {self.model.model_name}")

    def predict(self, inputs):
        if self.model is None:
            raise RuntimeError("load()를 먼저 실행하세요.")
        return self.model(inputs)

    def export(self, output_path: str = None, overwrite: bool =False):
        if self.model is None:
            raise RuntimeError("load()를 먼저 실행하세요.")
        onnx_file = output_path or os.path.join(ONNX_PATH, f"{self.model_name}.onnx")
        export_model(self.model, 'yolo', imgsz=self.imgsz, output_path=onnx_file, overwrite=overwrite)
        return onnx_file

    def convert(self, output_path: str = None, overwrite: bool =False):
        xml_file = output_path or os.path.join(XML_PATH, f"{self.model_name}.xml")
        onnx_file = os.path.join(ONNX_PATH, f"{self.model_name}.onnx")
        convert_model(onnx_file, output_path=xml_file, overwrite=overwrite)
        return xml_file


class ResNetModel(BaseModel):
    def __init__(self, num_classes: int = 13, device: str = "cpu", model_path: str = None, pretrained: bool =False):
        self.model_path = model_path
        if model_path is not None: self.model_name = os.path.splitext(os.path.basename(model_path))[0]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # self.model = models.resnet50(pretrained=pretrained)
        self.model = None
        self.pretrained = pretrained
        self.num_classes = num_classes

    def load(self):
        weights = ResNet50_Weights.DEFAULT if self.pretrained else None
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.to(self.device)

        # 모델 파라미터 로드
        if self.model_path is not None:
            if os.path.exists(self.model_path):
                print(f"모델 파라미터 로드 중: {self.model_path}")
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            else:
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        self.model.eval()

        print("ResNet 모델 준비 완료.")

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"ResNet 모델 저장 완료: {path}")

    def predict(self, inputs):
        if self.model is None:
            raise RuntimeError("load()를 먼저 실행하세요.")
        self.model.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs

    def export(self, output_path: str = None, overwrite: bool =False):
        output_path = f"{self.model_name}.onnx"
        export_model(self.model, 'pt', CLASSIFY_IMAGE_SIZE, output_path, overwrite=overwrite)
        return output_path

    def convert(self, output_path: str = None, overwrite: bool =False):
        output_path = output_path or f"{XML_PATH}/{self.model_name}.xml"
        result = convert_model(self.model_path, output_path=output_path, model=self.model, overwrite=overwrite)
        if result is None:
            print(f"OpenVINO 변환 실패")
        else:
            print(f"OpenVINO 변환 완료: {output_path}")
        return output_path
