MODEL_PATH = 'models'
ONNX_PATH = f'{MODEL_PATH}/onnx'
XML_PATH = f'{MODEL_PATH}/xml'

TORCH = ('pt', 'pth')
YOLO = ('yolo',)
ONNX = ('onnx',)

DETECT_IMAGE_SIZE = 640
CLASSIFY_IMAGE_SIZE = 512

IMAGES = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")

TARGETS = {
    0: "no_gesture",
    1: "fist",
    2: "one",
    3: "peace", # two
    4: "peace_inverted",
    5: "two_up",
    6: "two_up_inverted",
    7: "three",
    8: "three2",
    9: "four",
    10: "stop",
    11: "stop_inverted",
    12: "rock"
}
