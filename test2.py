from camera.realsense import RealSense
#from motors.motors import MotorController
from aumo.ov_model import OVDetectionModel, OVClassificationModel
from aumo.model import YOLOModel
from aumo.config import CLASS_NAMES, TARGETS
from aumo import YOLOModel, ResNetModel
import torchvision.models as models
import torchvision.transforms as transforms

from openvino import Core

import cv2
import time
import torch
import numpy as np


realsenseCamera=RealSense(filter_size=3, filter_use= False)

color_image, depth_image = realsenseCamera.getframe()

#오리지널 프레임과 dpeth map 프레임 받아오기
stacked_frame= np.hstack((color_image, depth_image))

#모터 컨트롤러 (전진, 후진 , 제자리 좌 우 회전, 정지 기능)
#motorController = MotorController()

#카메라 파라미터 불러오기
#fx , fy, cx, cy = realsenseCamera.get_intrinsic_camera('./camera/camera_parmeter.npz')

# ====================모델 관련 작업=========================#

yolo = YOLOModel('yolov5nu')
yolo.load()


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

#TARGETS = {0:'no_gesture', 1:'four', 2:'fist', 3:'one', 4:'peace', 5:'peace_inverted', 6:'rock'
 #         , 7:'stop', 8:'stop_inverted', 9:'three', 10:'three2', 11:'two_up', 12:'two_up_inverted'}

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
try:
    while True:
        frame, depth_image = realsenseCamera.getframe()

        start = time.time()
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # 원본 이미지 크기 (640x480)
        original_height, original_width = frame.shape[:2]
        
        # YOLO 입력을 위한 정사각형 이미지 생성 (640x640)
        model_input_size = 640
        
        # 원본 이미지를 640x640 크기로 리사이즈하면서 비율 유지
        # letterbox 방식으로 패딩 추가
        scale = min(model_input_size / original_width, model_input_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # 이미지 리사이즈
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # 패딩 계산
        pad_x = (model_input_size - new_width) // 2
        pad_y = (model_input_size - new_height) // 2
        
        # 패딩된 이미지 생성
        padded_image = np.zeros((model_input_size, model_input_size, 3), np.uint8)
        padded_image[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized_frame

        # YOLO 입력을 위한 blob 생성
        blob = cv2.dnn.blobFromImage(padded_image, scalefactor=1/255, size=(640, 640), swapRB=True)

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
                # YOLO 출력 좌표를 640x640 기준으로 받음
                center_x = outputs[0][i][0]
                center_y = outputs[0][i][1]
                width = outputs[0][i][2]
                height = outputs[0][i][3]
                
                # 바운딩 박스 좌표 계산 (640x640 기준)
                x1 = center_x - width / 2
                y1 = center_y - height / 2
                
                # 패딩 제거 후 원본 이미지 좌표로 변환
                x1_orig = (x1 - pad_x) / scale
                y1_orig = (y1 - pad_y) / scale
                width_orig = width / scale
                height_orig = height / scale
                
                box = [x1_orig, y1_orig, width_orig, height_orig]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections = []
        if len(result_boxes) > 0:  # result_boxes가 비어있지 않은지 확인
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                detection = {
                    'class_id': class_ids[index],
                    'class_name': CLASSES[class_ids[index]],
                    'confidence': scores[index],
                    'box': box,
                    'scale': 1.0  # 이미 원본 좌표로 변환했으므로 scale은 1.0
                }
                detections.append(detection)
                
                # 바운딩 박스 그리기 (원본 이미지 좌표 사용)
                x1 = int(max(0, box[0]))
                y1 = int(max(0, box[1]))
                x2 = int(min(original_width, box[0] + box[2]))
                y2 = int(min(original_height, box[1] + box[3]))
                
                draw_bounding_box(frame, class_ids[index], scores[index], x1, y1, x2, y2)

        # Classification 추론
        with torch.no_grad():
            input_tensor = transform(frame).unsqueeze(0).to(device)
            output, _ = classfication.predict(input_tensor)

        # 결과 출력
        cv2.putText(frame, f"Predicted: {TARGETS[output]}", (0, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        end = time.time()
        # show FPS
        fps = (1 / (end - start)) 
        fps_label = "Throughput: %.2f FPS" % fps
        cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        stacked_frame = np.hstack((frame, depth_image))
        cv2.imshow('RealSense RGB (left) + Depth (right)', stacked_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

finally:
    # Stop streaming
    realsenseCamera.pipeline.stop()
    cv2.destroyAllWindows()