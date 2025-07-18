import cv2
import os
import sys

def create_directory(path):
    """디렉토리가 없으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"디렉토리 생성: {path}")

def get_next_image_number(directory):
    """다음 이미지 번호를 찾아서 반환"""
    if not os.path.exists(directory):
        return 1
    
    existing_files = [f for f in os.listdir(directory) if f.startswith('IMG_right_') and f.endswith('.jpg')]
    if not existing_files:
        return 1
    
    # 기존 파일들에서 가장 큰 번호 찾기
    max_num = 0
    for filename in existing_files:
        try:
            # IMG_left_숫자.jpg에서 숫자 부분 추출
            num_str = filename.replace('IMG_right_', '').replace('.jpg', '')
            num = int(num_str)
            max_num = max(max_num, num)
        except ValueError:
            continue
    
    return max_num + 1

def main():
    # 저장할 디렉토리 경로
    save_directory = "./rectify/captured_image_right"
    
    # 디렉토리 생성
    create_directory(save_directory)
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)  # 0은 기본 카메라
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        sys.exit()
    
    # 이미지 카운터 초기화
    image_counter = get_next_image_number(save_directory)
    
    print("웹캠 캡처 프로그램이 시작되었습니다.")
    print("스페이스바를 누르면 이미지가 저장됩니다.")
    print("ESC 키를 누르면 프로그램이 종료됩니다.")
    print(f"이미지는 {save_directory} 디렉토리에 저장됩니다.")
    print(f"다음 이미지 번호: {image_counter}")
    print("-" * 50)
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 프레임 표시
        cv2.imshow('capturere', frame)
        
        # 키 입력 대기
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC 키
            print("프로그램을 종료합니다.")
            break
        elif key == 32:  # 스페이스바
            # 파일명 생성
            filename = f"IMG_right_{image_counter}.jpg"
            filepath = os.path.join(save_directory, filename)
            
            # 이미지 저장
            success = cv2.imwrite(filepath, frame)
            
            if success:
                print(f"이미지 저장 완료: {filepath}")
                image_counter += 1
            else:
                print(f"이미지 저장 실패: {filepath}")
    
    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()
    print("웹캠과 창이 정상적으로 종료되었습니다.")

if __name__ == "__main__":
    main()