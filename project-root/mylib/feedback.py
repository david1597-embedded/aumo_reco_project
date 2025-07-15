# feedback.py - 상태 모니터링 및 오류 감지
class FeedbackSystem:
    def __init__(self):
        self.status = "IDLE"

    def update_status(self, status):
        self.status = status
        print(f"[FEEDBACK] Status updated to: {status}")

    def check_error(self):
        # 더미 에러 체크
        return False
