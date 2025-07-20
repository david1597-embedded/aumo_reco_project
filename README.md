# 🚗 Au-mo Deto Solution | Auto-Moto Call with Object & Gesture Detection  
![Framework](https://img.shields.io/badge/Framework-Hadgrid-blue)
![YOLO](https://img.shields.io/badge/YOLOv10-blue)
![Language](https://img.shields.io/badge/Language-Python-yellow)
![Vision](https://img.shields.io/badge/Vision-StereoVision-critical)
![Library](https://img.shields.io/badge/Library-OpenCV-blueviolet)
![Toolkit](https://img.shields.io/badge/Toolkit-OpenVINO-success)

스테레오 비전 시스템과 객체 인식 모델을 융합한 팔로잉 로봇 카 프로젝트입니다.  
손동작 인식과 객체 탐지를 통해 다양한 제스처 기반 이동 및 제어 기능을 지원합니다.

---

## 🎯 프로젝트 개요

- 🤖 **팔로잉 로봇 카(Following Robot Car)**  
- 🧠 **손동작/객체 인식 기반 제어 시스템**  
- 🌐 **다양한 시나리오 대응 가능**  
- 📊 **여러 객체 인식/분류 모델 벤치마킹 기능 포함**

---

## 🏗 High Level Design

![high-level-desing-img](./doc/hld.png)

---

## 💡 Use Case

![use-case-img](./doc/usecase.jpg)

---

## ✋ 손동작 출력 라벨

| 제스처 | 의미 |
|--------|------|
| 🖐 one     | 전진 |
| ✌ two     | 후진 |
| 🤟 three2  | 제자리 회전 (우) |
| 🤘 three   | 제자리 회전 (좌) |
| ✊ fist    | 정지 |
| ✋ four    | 내 자리로 오기 |
| ✋✊ stop   | 따라오게 하기 |
| 🤟 rock    | 일반 모드 전환 (대기 상태 해제) |

---

## 🔁 시스템 흐름도 (Flowchart)

![flow-chart](./doc/flowchart.png)

---

## 📌 주요 기술 스택

- `🎥 Stereo Vision`: 깊이 인식 기반 거리 측정 및 방향 추정
- `👋 Hand Gesture Detection`: 손동작 기반 명령어 처리
- `🎯 Object Detection`: 사용자를 비롯한 주요 대상 객체 탐지
- `🧠 모델 비교 및 벤치마킹`: 다양한 분류기/탐지 모델 성능 비교
- `🚗 Robot Motion Control`: 명령어 기반 자율 주행

---

## 📂 구조 및 폴더 설명 *(선택 사항)*

> 원하시면 코드 구조나 실행 방법도 이어서 작성해드릴 수 있습니다.

---

## 📎 라이선스 및 참고

> 이 프로젝트는 연구 및 학습 목적이며, 관련된 모든 기술은 각 라이브러리의 라이선스를 따릅니다.
