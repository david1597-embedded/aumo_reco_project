from ultralytics import YOLO
import openvino as ov
import os
from typing import Tuple
from .utils import path_and_name
from .config import MODEL_PATH

def export_onnx(model: YOLO, imgsz: int =640, output: str = None) -> bool:
    try:
        model.export(format='onnx', imgsz=imgsz)
        default_onnx = f"{os.path.splitext(model.model_name)[0]}.onnx"

        if output is None:output = default_onnx
        else : os.rename(default_onnx, output)
        print(f"[OK] ONNX export complete: {output}")
        
        return True
    
    except Exception as e:
        print(f"[Error] ONNX export failed: {e}")
        
        return False

def convert_openvino(onnx_path: str, output: str = None) -> bool:
    if output is None:
        dir_path, name = path_and_name(onnx_path)
        output = os.path.join(dir_path, f"{name}.xml")
    try:
        ov_model = ov.convert_model(onnx_path)
        ov.save_model(ov_model, output)
        print(f"[OK] OpenVINO conversion complete: {output}")
        return True
    except Exception as e:
        print(f"[Error] OpenVINO conversion failed: {e}")
        return False