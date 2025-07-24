import cv2
import time
import torch
import numpy as np
from camera.realsense import RealSense
from aumo.ov_model import OVDetectionModel, OVClassificationModel
from aumo.model import YOLOModel, ResNetModel
from aumo.config import CLASS_NAMES, TARGETS
import torchvision.transforms as transforms
from openvino import Core

# RealSense 카메라 초기화
realsenseCamera = RealSense(filter_size=3, filter_use=False)

# 모델 초기화
yolo = OVDetectionModel('models/xml/yolov5nu.xml')
classfication = OVClassificationModel('models/xml/resnet50_512_10.xml')
yolo.load()
classfication.load()

# 상수 및 설정
CLASSES = CLASS_NAMES
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_node = yolo.compiled_model.outputs[0]
ir = yolo.compiled_model.create_infer_request()

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 바운딩 박스 그리기 함수
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 상태 변수 초기화
state = "IDLE"  # 기본 상태: IDLE
operator = None  # 고정된 객체 정보 (operator로 명명)
lock_timeout = 10  # 고정 유지 시간(초)
lock_start_time = None  # 고정 시작 시간
tracking_threshold = 150  # 객체 추적 임계값(픽셀)

prev_time = time.time()
try:
    while True:
        frame, depth_image = realsenseCamera.getframe()

        start = time.time()
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # 원본 이미지 크기
        original_height, original_width = frame.shape[:2]
        
        # YOLO 입력 이미지 준비 (640x640, 패딩 포함)
        model_input_size = 640
        scale = min(model_input_size / original_width, model_input_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        pad_x = (model_input_size - new_width) // 2
        pad_y = (model_input_size - new_height) // 2
        resized_frame = cv2.resize(frame, (new_width, new_height))
        padded_image = np.zeros((model_input_size, model_input_size, 3), np.uint8)
        padded_image[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized_frame
        blob = cv2.dnn.blobFromImage(padded_image, scalefactor=1/255, size=(640, 640), swapRB=True)

        # YOLO 추론
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
                center_x = outputs[0][i][0]
                center_y = outputs[0][i][1]
                width = outputs[0][i][2]
                height = outputs[0][i][3]
                x1 = center_x - width / 2
                y1 = center_y - height / 2
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

        # 상태에 따른 객체 처리
        if state == "IDLE":
            # IDLE 상태: 모든 객체 감지
            if len(result_boxes) > 0:
                for i in range(len(result_boxes)):
                    index = result_boxes[i]
                    box = boxes[index]
                    detection = {
                        'class_id': class_ids[index],
                        'class_name': CLASSES[class_ids[index]],
                        'confidence': scores[index],
                        'box': box,
                        'scale': 1.0
                    }
                    detections.append(detection)

                    # 바운딩 박스 그리기
                    x1 = int(max(0, box[0]))
                    y1 = int(max(0, box[1]))
                    x2 = int(min(original_width, box[0] + box[2]))
                    y2 = int(min(original_height, box[1] + box[3]))
                    draw_bounding_box(frame, class_ids[index], scores[index], x1, y1, x2, y2)

                    # 손동작 분류
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        with torch.no_grad():
                            input_tensor = transform(roi).unsqueeze(0).to(device)
                            output, _ = classfication.predict(input_tensor)
                            gesture = TARGETS[output]
                            cv2.putText(frame, f"Hand: {gesture}", (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            # "wake_up" 감지 시 operator 지정 및 NORMAL 모드로 전환
                            if gesture == "wake_up":
                                state = "NORMAL"
                                operator = detection
                                lock_start_time = time.time()
                                cv2.putText(frame, "operator", (x1, y1 - 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        elif state == "NORMAL":
            # NORMAL 상태: operator 객체만 감지
            found_operator = False
            if len(result_boxes) > 0:
                for i in range(len(result_boxes)):
                    index = result_boxes[i]
                    box = boxes[index]
                    # operator와 비교 (x, y 좌표 차이 100픽셀 이내)
                    if operator and abs(operator['box'][0] - box[0]) < tracking_threshold and \
                            abs(operator['box'][1] - box[1]) < tracking_threshold:
                        found_operator = True
                        detection = {
                            'class_id': class_ids[index],
                            'class_name': CLASSES[class_ids[index]],
                            'confidence': scores[index],
                            'box': box,
                            'scale': 1.0
                        }
                        detections.append(detection)

                        # 바운딩 박스 그리기
                        x1 = int(max(0, box[0]))
                        y1 = int(max(0, box[1]))
                        x2 = int(min(original_width, box[0] + box[2]))
                        y2 = int(min(original_height, box[1] + box[3]))
                        draw_bounding_box(frame, class_ids[index], scores[index], x1, y1, x2, y2)

                        # 손동작 분류
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            with torch.no_grad():
                                input_tensor = transform(roi).unsqueeze(0).to(device)
                                output, _ = classfication.predict(input_tensor)
                                gesture = TARGETS[output]
                                cv2.putText(frame, f"Hand: {gesture}", (x1, y2 + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                cv2.putText(frame, "operator", (x1, y1 - 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # 타임아웃 체크
                        if time.time() - lock_start_time > lock_timeout:
                            state = "IDLE"
                            operator = None
                            break

                # operator가 프레임에서 사라짐
                if not found_operator:
                    state = "IDLE"
                    operator = None

        # 현재 모드 표시 (우측 상단)
        mode_label = f"Mode: {state}"
        cv2.putText(frame, mode_label, (original_width - 150, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # FPS 표시 (좌측 상단)
        end = time.time()
        fps = 1 / (end - start)
        fps_label = f"Throughput: {fps:.2f} FPS"
        cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # RGB와 깊이 이미지 결합
        stacked_frame = np.hstack((frame, depth_image))
        cv2.imshow('RealSense RGB (left) + Depth (right)', stacked_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC로 종료
            break
finally:
    realsenseCamera.pipeline.stop()
    cv2.destroyAllWindows()