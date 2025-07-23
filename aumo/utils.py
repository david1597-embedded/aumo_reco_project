import os
import torch
from ultralytics import YOLO
import openvino as ov
from torchvision import transforms
from .config import TYPE_TORCH, TYPE_YOLO, TYPE_ONNX, CLASSIFY_IMAGE_SIZE

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def check_existing_file(ext: str, path: str) -> bool:
    if os.path.exists(path):
        print(f"{ext} 파일 이미 존재: {path}")
        return True
    else:
        print(f"{ext} export 중... {path}")
        return False

def export_model(model, model_type, imgsz=640, output_path=None, overwrite:bool=False):
    if output_path is None:
        output_path = f"{model.model_name}.onnx"
    
    if check_existing_file('ONNX', output_path) and not overwrite:
        return output_path
    
    if model_type in TYPE_YOLO:
        print(f"[export_model] YOLO ONNX export: {output_path}")
        model.export(format='onnx', imgsz=imgsz)
        default_onnx = f"{os.path.splitext(model.model_name)[0]}.onnx"
        if output_path != default_onnx:
            os.rename(default_onnx, output_path)
        return output_path

    elif model_type in TYPE_TORCH:
        model.eval()
        dummy_input = torch.randn(1, 3, imgsz, imgsz)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
        )
        print(f"ONNX export 완료: {output_path}")
        return output_path
        # print("[export_model] PyTorch 모델은 직접 변환 가능하므로 export 생략")
        # return None

    elif model_type in TYPE_ONNX:
        print("[export_model] ONNX 파일은 export 불필요")
        return output_path

    else:
        raise ValueError(f"알 수 없는 모델 타입: {model_type}")

def convert_model(
    input_path,
    output_path=None,
    device="CPU",
    model=None,
    input_size=(1, 3, 512, 512),
    overwrite:bool = False
):
    """
    모델 변환 함수 (OpenVINO 변환)
    
    - input_path: 변환할 파일 경로 (.pt/.pth, .onnx, .xml)
    - output_path: 변환 결과 경로 (.xml), 지정 안하면 자동 생성
    - device: OpenVINO device (기본 CPU)
    - model: .pt 변환 시 사용될 PyTorch 모델 클래스 (예: torchvision.models.resnet50)
    - input_size: .pt 변환 시 dummy input 크기 (튜플)
    """

    ext = os.path.splitext(input_path)[1].lower()
    if output_path is None: 
        output_path = input_path.replace(ext, ".xml")

    if check_existing_file('OpenVINO', output_path) and not overwrite:
        return output_path
    
    core = ov.Core()

    if ext in TYPE_TORCH:
        if model is None:
            raise ValueError("`.pt` 변환 시 model 인자를 반드시 지정해야 합니다.")
        print(f"[convert_model] PyTorch 모델({input_path}) OpenVINO 변환 시작")
        dummy_input = torch.randn(input_size)
        try:
            ov_model = ov.convert_model(model, example_input=dummy_input)
            ov.save_model(ov_model, output_path)
            print(f"[convert_model] PyTorch 모델 OpenVINO 변환 완료: {output_path}")
            return output_path
        except Exception as e:
            print(f"[convert_model] 변환 실패: {e}")
            return None

    elif ext in TYPE_ONNX:
        if output_path is None:
            output_path = input_path.replace(".onnx", ".xml")
        try:
            model = core.read_model(input_path)
            ov.save_model(model, output_path)
            # ov_model = core.compile_model(model, device)
            # core.save_model(model, output_path)
            print(f"[convert_model] ONNX -> OpenVINO 변환 완료: {output_path}")
            return output_path
        except Exception as e:
            print(f"[convert_model] 변환 실패: {e}")
            return None

    elif ext == '.xml':
        print("[convert_model] 이미 OpenVINO IR(.xml) 파일입니다.")
        return input_path

    else:
        raise ValueError(f"지원하지 않는 확장자: {ext}")
