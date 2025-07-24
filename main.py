import cv2
import time
import torch
import numpy as np
from openvino import Core
import torchvision.models as models
import torchvision.transforms as transforms
from aumo.ov_model import OVDetectionModel, OVClassificationModel
from aumo.model import YOLOModel
from aumo.config import CLASS_NAMES
# from aumo.config import TARGETS
# from ultralytics.yolo.utils import ROOT, yaml_load
# from ultralytics.yolo.utils.checks import check_yaml
from aumo import YOLOModel, ResNetModel

yolo = YOLOModel('yolov5nu')
yolo.load()
yolo.export()
yolo.convert()

resnet = ResNetModel(model_path='models/resnet50_512_10.pt')
resnet.load()
resnet.convert()

# MODEL_NAME = "yolov8n"
CLASSES = CLASS_NAMES
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

TARGETS = {0:'fist', 1:'four', 2:'no_gesture', 3:'one', 4:'peace', 5:'peace_inverted', 6:'rock'
          , 7:'stop', 8:'stop_inverted', 9:'three', 10:'three2', 11:'two_up', 12:'two_up_inverted'}

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# detection = YOLOModel('yolov8s')
yolo = OVDetectionModel('models/xml/yolov5nu.xml')
classfication = OVClassificationModel('models/xml/resnet50_512_10.xml')

yolo.load()
classfication.load()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(TARGETS)

output_node = yolo.compiled_model.outputs[0]
ir = yolo.compiled_model.create_infer_request()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

prev_time = time.time()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    [height, width, _] = frame.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame
    scale = length / 640

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

    outputs = ir.infer(blob)[output_node]

    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxClassIndex != 0: continue
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            'class_id': class_ids[index],
            'class_name': Ctry:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 필터 초기화
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        hole_filling = rs.hole_filling_filter()LASSES[class_ids[index]],
            'confidence': scores[index],
            'box': try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 필터 초기화
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        hole_filling = rs.hole_filling_filter()box,
            'scale': scale}
        detections.append(detection)
        draw_bounding_box(frame, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                          round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

    # 추론
    with torch.no_grad():
        input_tensor = transform(frame).unsqueeze(0).to(device)
        output,_ = classfication.predict(input_tensor)

    # 결과 출력
    cv2.putText(frame, f"Predicted: {TARGETS[output]}", (0, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,TARGETS 0), 2)
    
    end = time.time()
    # show FPS
    fps = (1 / (end - start)) 
    fps_label = "Throughput: %.2f FPS" % fps
    cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('YOLOv8 OpenVINO Infer Demo on AIxBoard', frame)
    # wait key for ending
    if cv2.waitKey(1) > -1:
        print("finished by user")
        cap.release()
        cv2.destroyAllWindows()
        break