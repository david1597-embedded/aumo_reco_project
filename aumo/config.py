MODEL_PATH = 'models'
ONNX_PATH = f'{MODEL_PATH}/onnx'
XML_PATH = f'{MODEL_PATH}/xml'

TYPE_TORCH = ('pt', 'pth', '.pt', '.pth')
TYPE_YOLO = ('yolo',)
TYPE_ONNX = ('onnx','.onnx')

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

CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']