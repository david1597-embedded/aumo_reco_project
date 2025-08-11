import cv2
import time
import torch
import numpy as np
from aumo.ov_model import OVDetectionModel, OVClassificationModel
from aumo.model import YOLOModel, ResNetModel
from aumo.config import CLASS_NAMES, TARGETS
import torchvision.transforms as transforms
from openvino import Core

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

# IoU 계산 함수
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

# 상태 변수 초기화
state = "IDLE"
operator = None
lock_timeout = 10
min_lock_duration = 3
lock_start_time = None
iou_threshold = 0.5
gesture_timeout = 10
gesture_start_time = None
last_gesture = None
last_position_time = None
position_interval = 3

cap = cv2.VideoCapture(0)

prev_time = time.time()
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 캡처 실패")
            break
        start = time.time()
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
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
        try:
            outputs = ir.infer(blob)[output_node]
            outputs = np.array([cv2.transpose(outputs[0])])
        except Exception as e:
            print(f"YOLO 추론 오류: {e}")
            continue
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
                    if roi.size > 0 and y2 > y1 and x2 > x1:
                        try:
                            with torch.no_grad():
                                input_tensor = transform(roi).unsqueeze(0).to(device)
                                output, _ = classfication.predict(input_tensor)
                                gesture = TARGETS[output]
                                cv2.putText(frame, f"Hand: {gesture}", (x1, y1 - 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                print(f"IDLE 모드: 제스처 {gesture} 감지")
                        except Exception as e:
                            print(f"IDLE 모드 제스처 분류 오류: {e}")
                            gesture = "no_gesture"
                    else:
                        print("IDLE 모드: 유효하지 않은 ROI")
                        gesture = "no_gesture"

                    # "wake_up" 감지 시 operator 지정 및 NORMAL 모드로 전환
                    if gesture == "wake_up":
                        state = "NORMAL"
                        operator = detection
                        lock_start_time = time.time()
                        gesture_start_time = time.time()
                        last_gesture = gesture
                        cv2.putText(frame, "operator", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # 사진 촬영
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"capture_wakeup_{timestamp}.jpg", frame)
                        print(f"NORMAL 모드로 전환, 프레임 캡처: capture_wakeup_{timestamp}.jpg")
                    # "follow" 감지 시 operator 지정 및 FOLLOW 모드로 전환
                    elif gesture == "follow":
                        state = "FOLLOW"
                        operator = detection
                        lock_start_time = time.time()
                        gesture_start_time = time.time()
                        last_gesture = gesture
                        last_position_time = time.time()
                        cv2.putText(frame, "operator", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # 사진 촬영
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"capture_follow_{timestamp}.jpg", frame)
                        print(f"FOLLOW 모드로 전환, 프레임 캡처: capture_follow_{timestamp}.jpg")

        elif state == "NORMAL":
            found_operator = False
            best_iou = 0
            best_detection = None
            gesture = "no_gesture"  # 기본값 설정
            if len(result_boxes) > 0:
                for i in range(len(result_boxes)):
                    index = result_boxes[i]
                    box = boxes[index]
                    if operator:
                        iou = calculate_iou(operator['box'], box)
                        if iou > iou_threshold and iou > best_iou:
                            best_iou = iou
                            best_detection = {
                                'class_id': class_ids[index],
                                'class_name': CLASSES[class_ids[index]],
                                'confidence': scores[index],
                                'box': box,
                                'scale': 1.0
                            }

                if best_detection:
                    found_operator = True
                    operator = best_detection

                    # 바운딩 박스 그리기
                    x1 = int(max(0, best_detection['box'][0]))
                    y1 = int(max(0, best_detection['box'][1]))
                    x2 = int(min(original_width, best_detection['box'][0] + best_detection['box'][2]))
                    y2 = int(min(original_height, best_detection['box'][1] + best_detection['box'][3]))
                    draw_bounding_box(frame, best_detection['class_id'], best_detection['confidence'], x1, y1, x2, y2)

                    # 손동작 분류
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0 and y2 > y1 and x2 > x1:
                        try:
                            with torch.no_grad():
                                input_tensor = transform(roi).unsqueeze(0).to(device)
                                output, _ = classfication.predict(input_tensor)
                                gesture = TARGETS[output]
                                cv2.putText(frame, f"Hand: {gesture}", (x1, y1 - 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                print(f"NORMAL 모드: 제스처 {gesture} 감지")
                        except Exception as e:
                            print(f"NORMAL 모드 제스처 분류 오류: {e}")
                            gesture = "no_gesture"
                    else:
                        print("NORMAL 모드: 유효하지 않은 ROI")
                        gesture = "no_gesture"
                    cv2.putText(frame, "operator", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 모터 제어 (주석 처리된 부분 유지)
                    if gesture == "forward":
                        print("전진 명령 수행")
                    elif gesture == "backward":
                        print("후진 명령 수행")
                    elif gesture == "turn_left":
                        print("좌회전 명령 수행")
                    elif gesture == "turn_right":
                        print("우회전 명령 수행")
                    elif gesture == "my_position":
                        print("오퍼레이터 위치로 이동 명령 수행")
                    elif gesture == "stop":
                        print("정지 명령 수행")
                    elif gesture == "follow":
                        state = "FOLLOW"
                        lock_start_time = time.time()
                        gesture_start_time = time.time()
                        last_gesture = gesture
                        last_position_time = time.time()
                        print("FOLLOW 모드로 전환")
                        continue

                    # 손동작 타임아웃 관리 (wake_up 제외)
                    if gesture != last_gesture and gesture != "wake_up":
                        gesture_start_time = time.time()
                        last_gesture = gesture
                        print(f"제스처 {gesture}(으)로 변경, gesture_start_time 재설정")

                    # no_gesture나 stop이 아닌 제스처 감지 시 lock_start_time 재설정
                    if gesture not in ["no_gesture", "stop"]:
                        lock_start_time = time.time()
                        print(f"비활성 제스처 {gesture} 감지, lock_start_time 재설정")

                # 최소 3초 유지 체크
                elapsed_time = time.time() - lock_start_time if lock_start_time else 0
                if elapsed_time < min_lock_duration:
                    continue

                # "wake_up" 감지 시 IDLE로 전환
                if best_detection and gesture == "wake_up" and elapsed_time >= min_lock_duration:
                    state = "IDLE"
                    operator = None
                    gesture_start_time = None
                    last_gesture = None
                    last_position_time = None
                    print("wake_up 제스처로 인해 IDLE 모드로 전환")
                    continue

                # "no_gesture" 또는 "stop" 10초 유지 시 IDLE로 전환
                if best_detection and gesture in ["no_gesture", "stop"]:
                    if gesture_start_time and (time.time() - gesture_start_time) > gesture_timeout:
                        state = "IDLE"
                        operator = None
                        gesture_start_time = None
                        last_gesture = None
                        last_position_time = None
                        print("제스처 타임아웃으로 인해 IDLE 모드로 전환")
                        continue

                # 타임아웃 체크
                if elapsed_time > lock_timeout:
                    state = "IDLE"
                    operator = None
                    gesture_start_time = None
                    last_gesture = None
                    last_position_time = None
                    print("잠금 타임아웃으로 인해 IDLE 모드로 전환")
                    continue

                # operator가 프레임에서 사라짐
                if not found_operator:
                    state = "IDLE"
                    operator = None
                    gesture_start_time = None
                    last_gesture = None
                    last_position_time = None
                    print("오퍼레이터 손실로 인해 IDLE 모드로 전환")
                    continue

        elif state == "FOLLOW":
            found_operator = False
            best_iou = 0
            best_detection = None
            gesture = "no_gesture"  # 기본값 설정
            if len(result_boxes) > 0:
                for i in range(len(result_boxes)):
                    index = result_boxes[i]
                    box = boxes[index]
                    if operator:
                        iou = calculate_iou(operator['box'], box)
                        if iou > iou_threshold and iou > best_iou:
                            best_iou = iou
                            best_detection = {
                                'class_id': class_ids[index],
                                'class_name': CLASSES[class_ids[index]],
                                'confidence': scores[index],
                                'box': box,
                                'scale': 1.0
                            }

                if best_detection:
                    found_operator = True
                    operator = best_detection

                    # 바운딩 박스 그리기
                    x1 = int(max(0, best_detection['box'][0]))
                    y1 = int(max(0, best_detection['box'][1]))
                    x2 = int(min(original_width, best_detection['box'][0] + best_detection['box'][2]))
                    y2 = int(min(original_height, best_detection['box'][1] + best_detection['box'][3]))
                    draw_bounding_box(frame, best_detection['class_id'], best_detection['confidence'], x1, y1, x2, y2)

                    # 손동작 분류
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0 and y2 > y1 and x2 > x1:
                        try:
                            with torch.no_grad():
                                input_tensor = transform(roi).unsqueeze(0).to(device)
                                output, _ = classfication.predict(input_tensor)
                                gesture = TARGETS[output]
                                cv2.putText(frame, f"Hand: {gesture}", (x1, y1 - 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                print(f"FOLLOW 모드: 제스처 {gesture} 감지")
                        except Exception as e:
                            print(f"FOLLOW 모드 제스처 분류 오류: {e}")
                            gesture = "no_gesture"
                    else:
                        print("FOLLOW 모드: 유효하지 않은 ROI")
                        gesture = "no_gesture"
                    cv2.putText(frame, "operator", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 주기적으로 my_position 호출 (3초마다)
                    if last_position_time is None or (time.time() - last_position_time) >= position_interval:
                        print("오퍼레이터 따라가는 중...")
                        last_position_time = time.time()

                    # "follow" 감지 시 IDLE로 전환
                    if gesture == "follow":
                        state = "IDLE"
                        operator = None
                        gesture_start_time = None
                        last_gesture = None
                        last_position_time = None
                        print("follow 제스처로 인해 IDLE 모드로 전환")
                        continue

                    # 손동작 타임아웃 관리
                    if gesture != last_gesture:
                        gesture_start_time = time.time()
                        last_gesture = gesture
                        print(f"제스처 {gesture}(으)로 변경, gesture_start_time 재설정")

                    # no_gesture나 stop이 아닌 제스처 감지 시 lock_start_time 재설정
                    if gesture not in ["no_gesture", "stop"]:
                        lock_start_time = time.time()
                        print(f"비활성 제스처 {gesture} 감지, lock_start_time 재설정")

                # 최소 3초 유지 체크
                elapsed_time = time.time() - lock_start_time if lock_start_time else 0
                if elapsed_time < min_lock_duration:
                    continue

                # "no_gesture" 또는 "stop" 10초 유지 시 IDLE로 전환
                if best_detection and gesture in ["no_gesture", "stop"]:
                    if gesture_start_time and (time.time() - gesture_start_time) > gesture_timeout:
                        state = "IDLE"
                        operator = None
                        gesture_start_time = None
                        last_gesture = None
                        last_position_time = None
                        print("제스처 타임아웃으로 인해 IDLE 모드로 전환")
                        continue

                # 타임아웃 체크
                if elapsed_time > lock_timeout:
                    state = "IDLE"
                    operator = None
                    gesture_start_time = None
                    last_gesture = None
                    last_position_time = None
                    print("잠금 타임아웃으로 인해 IDLE 모드로 전환")
                    continue

                # operator가 프레임에서 사라짐
                if not found_operator:
                    state = "IDLE"
                    operator = None
                    gesture_start_time = None
                    last_gesture = None
                    last_position_time = None
                    print("오퍼레이터 손실로 인해 IDLE 모드로 전환")
                    continue

        # 현재 모드 표시 (우측 상단)
        mode_label = f"Mode: {state}"
        cv2.putText(frame, mode_label, (original_width - 180, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # FPS 표시 (좌측 상단)
        end = time.time()
        fps = 1 / (end - start) if (end - start) > 0 else 0
        fps_label = f"Throughput: {fps:.2f} FPS"
        cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 스페이스바로 화면 캡처
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # 스페이스바
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"capture_spacebar_{timestamp}.jpg", frame)
            print(f"프레임 캡처: capture_spacebar_{timestamp}.jpg")
        elif key == 27:  # ESC로 종료
            print("프로그램 종료")
            break

        cv2.imshow('Video Capture Frame', frame)
finally:
    cap.release()
    cv2.destroyAllWindows()