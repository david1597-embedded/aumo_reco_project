# ai.py - 제스처 인식 + OpenVINO 포함
class GestureAI:
    def __init__(self, model_path_xml, model_path_bin):
        self.model_xml = model_path_xml
        self.model_bin = model_path_bin
        print(f"[AI] Loading model: {model_path_xml}, {model_path_bin}")
        # 여기 OpenVINO Inference Engine 등 로딩이 들어갈 자리

    def predict(self, frame):
        # 더미 처리 예시
        print("[AI] Predicting gesture from frame...")
        return "forward"  # 예시 출력
