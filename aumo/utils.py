import os
import torch
from ultralytics import YOLO
import openvino as ov
from .config import TORCH, YOLO, ONNX

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def export_model(model, model_type, imgsz=640, output_path=None):
    if output_path is None:
        output_path = f"{model.model_name}.onnx"
    
    if model_type in YOLO:
        print(f"[export_model] YOLO ONNX export: {output_path}")
        model.export(format='onnx', imgsz=imgsz)
        default_onnx = f"{os.path.splitext(model.model_name)[0]}.onnx"
        if output_path != default_onnx:
            os.rename(default_onnx, output_path)
        return output_path

    elif model_type in TORCH:
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

    elif model_type in ONNX:
        print("[export_model] ONNX 파일은 export 불필요")
        return output_path

    else:
        raise ValueError(f"알 수 없는 모델 타입: {model_type}")

def convert_model(
    input_path,
    output_path=None,
    device="CPU",
    model_class=None,
    input_size=(1, 3, 512, 512),
):
    """
    모델 변환 함수 (OpenVINO 변환)
    
    - input_path: 변환할 파일 경로 (.pt/.pth, .onnx, .xml)
    - output_path: 변환 결과 경로 (.xml), 지정 안하면 자동 생성
    - device: OpenVINO device (기본 CPU)
    - model_class: .pt 변환 시 사용될 PyTorch 모델 클래스 (예: torchvision.models.resnet50)
    - input_size: .pt 변환 시 dummy input 크기 (튜플)
    """

    ext = os.path.splitext(input_path)[1].lower()
    core = ov.Core()

    if ext in TORCH:
        if model_class is None:
            raise ValueError("`.pt` 변환 시 model_class 인자를 반드시 지정해야 합니다.")
        print(f"[convert_model] PyTorch 모델({input_path}) 로드 및 OpenVINO 변환 시작")
        model = model_class(pretrained=False)
        model.load_state_dict(torch.load(input_path, map_location="cpu"))
        model.eval()
        dummy_input = torch.randn(input_size)
        try:
            ov_model = ov.convert_model(model, example_input=dummy_input)
            if output_path is None:
                output_path = input_path.replace(ext, ".xml")
            ov.save_model(ov_model, output_path)
            print(f"[convert_model] PyTorch 모델 OpenVINO 변환 완료: {output_path}")
            return output_path
        except Exception as e:
            print(f"[convert_model] 변환 실패: {e}")
            return None

    elif ext in ONNX:
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
