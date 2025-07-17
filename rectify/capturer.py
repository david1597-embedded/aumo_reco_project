import cv2
import os

# 카메라 초기화
left_cam = cv2.VideoCapture(0)
right_cam = cv2.VideoCapture(2)

# 카메라가 제대로 열렸는지 확인
if not left_cam.isOpened() or not right_cam.isOpened():
    print("Error: One or both cameras could not be opened.")
    exit()

# 카메라 해상도 설정 (640x480)
left_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
left_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
right_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
right_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 출력 디렉토리 설정 (현재 디렉토리)
output_dir = "."
left_output_path = os.path.join(output_dir, "IMG_for_corresponing_point_left.jpeg")
right_output_path = os.path.join(output_dir, "IMG_for_corresponing_point_right.jpeg")

# 윈도우 생성 및 크기 설정
cv2.namedWindow("Left Camera (0)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Right Camera (2)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Left Camera (0)", 640, 480)
cv2.resizeWindow("Right Camera (2)", 640, 480)

print("Press SPACE to capture and save images, or ESC to quit.")

while True:
    # 양쪽 카메라에서 프레임 읽기
    ret_left, frame_left = left_cam.read()
    ret_right, frame_right = right_cam.read()

    if not ret_left or not ret_right:
        print("Error: Failed to capture images from one or both cameras.")
        break

    # 프레임 표시
    cv2.imshow("Left Camera (0)", frame_left)
    cv2.imshow("Right Camera (2)", frame_right)

    # 키 입력 대기
    key = cv2.waitKey(1) & 0xFF

    # 스페이스 키(32)를 누르면 이미지 각각 저장
    if key == 32:
        # 왼쪽 이미지 저장
        cv2.imwrite(left_output_path, frame_left)
        # 오른쪽 이미지 저장
        cv2.imwrite(right_output_path, frame_right)
        print(f"Left image saved as {left_output_path}")
        print(f"Right image saved as {right_output_path}")

    # ESC 키(27)를 누르면 종료
    elif key == 27:
        break

# 자원 해제
left_cam.release()
right_cam.release()
cv2.destroyAllWindows()