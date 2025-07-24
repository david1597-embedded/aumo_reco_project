import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
# Create a pipeline
pipeline = rs.pipeline()

# Configure streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
def create_directory(path):
    """디렉토리가 없으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"디렉토리 생성: {path}")

def get_next_image_number(directory):
    """다음 이미지 번호를 찾아서 반환"""
    if not os.path.exists(directory):
        return 1
    
    existing_files = [f for f in os.listdir(directory) if f.startswith('IMG_') and f.endswith('.jpg')]
    if not existing_files:
        return 1
    
    max_num = 0
    for filename in existing_files:
        try:
            num_str = filename.replace('IMG_', '').replace('.jpg', '')
            num = int(num_str)
            max_num = max(max_num, num)
        except ValueError:
            continue
    
    return max_num + 1

def main():
    save_directory = "./camera/calibration/captured_image"
    create_directory(save_directory)


    image_counter = get_next_image_number(save_directory)

    print("웹캠 캡처 프로그램이 시작되었습니다.")
    print("스페이스바를 누르면 이미지가 저장됩니다.")
    print("ESC 키를 누르면 프로그램이 종료됩니다.")
    print(f"이미지는 {save_directory} 디렉토리에 저장됩니다.")
    print(f"다음 이미지 번호: {image_counter}")
    print("-" * 50)

    window_name = 'capturere'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    while True:
        frames = pipeline.wait_for_frames()
        #depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
  
        color_image = np.asanyarray(color_frame.get_data())

        # 디스플레이용 리사이즈
        resized_display_frame = cv2.resize(color_image, (640, 480))
        cv2.imshow(window_name, resized_display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("프로그램을 종료합니다.")
            break
        elif key == 32:  # Space
            filename = f"IMG_{image_counter}.jpg"
            filepath = os.path.join(save_directory, filename)

            # 저장용 리사이즈
            resized_save_frame = cv2.resize(color_image, (640, 480))
            success = cv2.imwrite(filepath, resized_save_frame)
            if success:
                print(f"이미지 저장 완료: {filepath}")
                image_counter += 1
            else:
                print(f"이미지 저장 실패: {filepath}")

    pipeline.stop()
    cv2.destroyAllWindows()
    print("웹캠과 창이 정상적으로 종료되었습니다.")

if __name__ == "__main__":
    main()
