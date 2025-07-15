# main.py - 전체 실행 진입점
from camera import CameraStream
from ai import GestureAI
from motor import MecanumMotor
from feedback import FeedbackSystem
import yaml

# 설정 로드
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 모듈 초기화
cam = CameraStream(config["camera_index"])
ai = GestureAI(config["model_xml"], config["model_bin"])
motor = MecanumMotor()
feedback = FeedbackSystem()

# 프레임 기반 제어 루프 (간단화)
frame = cam.get_frame()
if frame:
    gesture = ai.predict(frame)
    if gesture == "forward":
        motor.move_forward()
        feedback.update_status("FORWARD")
    elif gesture == "stop":
        motor.stop()
        feedback.update_status("STOP")

cam.release()
