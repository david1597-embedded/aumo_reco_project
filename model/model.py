from ultralytics import YOLO
import os
from .convert import *
from .config import *
from .utils import ensure_dirs

def load_model(model_name:str, export: bool = True, convert: bool = True, imgsz: int = DEFAULT_IMGSZ):
    ensure_dirs(MODEL_PATH, ONNX_PATH, XML_PATH)
    try:
        model = YOLO(f"{MODEL_PATH}/{model_name}.pt")
    except:
        model = YOLO(f"{model_name}.pt")
        os.rename(f"{model_name}.pt", MODEL_PATH)

    print(model.model_name)

    onnx_file = f"{ONNX_PATH}/{model_name}.onnx"
    xml_file = f"{XML_PATH}/{model_name}.xml"

    if export and not os.path.exists(onnx_file):
        export_onnx(model, output=onnx_file)

    if convert and not os.path.exists(xml_file):  # 이미 있으면 스킵
        convert_openvino(onnx_file, output=xml_file)