# camera.py - 카메라 스트림 + 제스처 감지 처리
import cv2

class CameraStream:
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
