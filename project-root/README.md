# 프로젝트 설명서

제스처 기반 메카넘 휠 제어 시스템입니다.
# Project: Gesture-based Mecanum Wheel Control

## 구조 설명
- `ai.py`: 제스처 인식 AI(OpenVINO)
- `motor.py`: 메카넘 휠 전/후/회전 제어
- `camera.py`: 카메라 입력
- `feedback.py`: 시스템 피드백 관리
- `main.py`: 전체 실행 진입점

## 실행 방법
```bash
python main.py

project-root/
├── ai.py                   # 제스처 인식 + OpenVINO 포함
├── motor.py                # 메카넘 휠 제어 (클래스 내 전진/후진 등 메서드 포함)
├── camera.py               # 카메라 스트림 + 제스처 감지 처리
├── feedback.py             # 상태 모니터링 및 오류 감지
├── config.yaml             # 전체 설정 관리
├── main.py                 # 전체 실행 진입점
├── README.md               # 프로젝트 설명
├── .gitignore              # Git 제외 파일

├── models/                 # 🔸 모델 파일 (XML, BIN 등) 저장
│   ├── hagrid_gesture.xml
│   └── hagrid_gesture.bin

├── data/                   # 🔸 데모 입력 데이터 (예: 이미지, 영상 등)
│   └── demo_input.jpg

├── results/                # 🔸 추론 결과 저장 (이미지, 로그 등)
│   └── prediction_output.jpg