import os

# 생성할 폴더 및 파일 구조 정의
folders = [
    "project-root/models",
    "project-root/data",
    "project-root/results"
]

files = {
    "project-root/ai.py": "# ai.py - 제스처 인식 + OpenVINO 포함\n",
    "project-root/motor.py": """# motor.py - 메카넘 휠 제어
class MecanumMotor:
    def move_forward(self): print("[MOTOR] Moving forward")
    def move_backward(self): print("[MOTOR] Moving backward")
    def rotate(self): print("[MOTOR] Rotating")
    def call(self): print("[MOTOR] Calling")
    def follow(self): print("[MOTOR] Following")
    def stop(self): print("[MOTOR] Stopping")
""",
    "project-root/camera.py": "# camera.py - 카메라 스트림 + 제스처 감지 처리\n",
    "project-root/feedback.py": "# feedback.py - 상태 모니터링 및 오류 감지\n",
    "project-root/config.yaml": """camera_index: 0
model_xml: "models/hagrid_gesture.xml"
model_bin: "models/hagrid_gesture.bin"
input_image: "data/demo_input.jpg"
output_result: "results/prediction_output.jpg"
""",
    "project-root/main.py": "# main.py - 전체 실행 진입점\n",
    "project-root/README.md": "# 프로젝트 설명서\n\n제스처 기반 메카넘 휠 제어 시스템입니다.",
    "project-root/.gitignore": "__pycache__/\n*.pyc\n.vscode/\n.env/\nresults/\n",
    "project-root/models/hagrid_gesture.xml": "",  # 빈 파일로 생성
    "project-root/models/hagrid_gesture.bin": "",
    "project-root/data/demo_input.jpg": "",        # 이미지 실제 내용은 없음
    "project-root/results/prediction_output.jpg": ""
}

# 폴더 생성
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# 파일 생성
for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ 프로젝트 구조 생성 완료!")
