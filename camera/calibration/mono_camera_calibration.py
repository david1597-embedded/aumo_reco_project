import cv2
import numpy as np
import os
import glob

def create_directory(path):
    """디렉토리가 없으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"디렉토리 생성: {path}")

def preprocess_image(img):
    """이미지 전처리를 통해 코너 검출 성능 향상"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 히스토그램 균등화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return blurred

def try_multiple_checkerboard_sizes(gray, base_size=(11, 8)):
    """여러 체커보드 크기로 시도"""
    # 일반적인 체커보드 크기들
    sizes_to_try = [
        (11, 8),   # 원본
        (10, 7),   # 가장자리 제외
        (9, 6),    # 더 작게
        (8, 5),    # 가장 작게
        (8, 11),   # 회전된 형태
        (7, 10),   # 회전된 형태
        (6, 9),    # 회전된 형태
        (5, 8),    # 회전된 형태
    ]
    
    for size in sizes_to_try:
        print(f"    체커보드 크기 {size} 시도중...")
        ret, corners = cv2.findChessboardCorners(gray, size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE + 
                                                cv2.CALIB_CB_FILTER_QUADS)
        if ret:
            print(f"    ✓ 체커보드 크기 {size}에서 성공!")
            return ret, corners, size
    
    return False, None, None

def calibrate_camera():
    # 체커보드 설정
    SQUARE_SIZE = 25  # 체커보드 사각형 크기 (mm)
    
    # 이미지 디렉토리 경로
    image_dir = "./camera/calibration/captured_image"
    calibration_dir = "./camera"
    
    # 캘리브레이션 데이터 디렉토리 생성
    create_directory(calibration_dir)
    
    # 체커보드 코너 검출 기준
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # 3D 점과 2D 점 저장을 위한 배열
    objpoints = []  # 3D 점 (실제 세계)
    imgpoints = []  # 2D 점 (이미지 평면)
    
    # 이미지 파일 찾기
    image_files = []
    for i in range(1, 16):  # IMG_left_1.jpg부터 IMG_left_15.jpg까지
        filepath = os.path.join(image_dir, f"IMG_{i}.jpg")
        if os.path.exists(filepath):
            image_files.append(filepath)
    
    if not image_files:
        print(f"이미지 파일을 찾을 수 없습니다: {image_dir}")
        return None
    
    print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")
    print("체커보드 코너 검출을 시작합니다...")
    
    successful_images = []
    used_checkerboard_size = None
    
    for i, filepath in enumerate(image_files):
        print(f"\n처리 중: {os.path.basename(filepath)} ({i+1}/{len(image_files)})")
        
        # 이미지 읽기
        img = cv2.imread(filepath)
        if img is None:
            print(f"  ✗ 이미지를 읽을 수 없습니다: {filepath}")
            continue
        
        print(f"  이미지 크기: {img.shape[1]}x{img.shape[0]}")
        
        # 이미지 전처리
        gray = preprocess_image(img)
        
        # 원본 그레이스케일도 시도
        gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 두 가지 전처리 방법 시도
        gray_versions = [
            ("전처리된 이미지", gray),
            ("원본 그레이스케일", gray_original)
        ]
        
        corners_found = False
        
        for version_name, gray_version in gray_versions:
            print(f"  {version_name}으로 시도중...")
            
            # 여러 체커보드 크기 시도
            ret, corners, detected_size = try_multiple_checkerboard_sizes(gray_version)
            
            if ret:
                print(f"  ✓ 체커보드 코너 검출 성공! 크기: {detected_size}")
                
                # 첫 번째 성공한 크기를 기준으로 설정
                if used_checkerboard_size is None:
                    used_checkerboard_size = detected_size
                    print(f"  기준 체커보드 크기 설정: {used_checkerboard_size}")
                
                # 기준 크기와 일치하는 경우만 사용
                if detected_size == used_checkerboard_size:
                    # 코너 정밀도 향상
                    corners2 = cv2.cornerSubPix(gray_version, corners, (11, 11), (-1, -1), criteria)
                    
                    # 3D 점 생성 (검출된 크기에 맞춰)
                    objp = np.zeros((detected_size[0] * detected_size[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:detected_size[0], 0:detected_size[1]].T.reshape(-1, 2)
                    objp *= SQUARE_SIZE
                    
                    # 3D 점과 2D 점 저장
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    successful_images.append(filepath)
                    
                    # 코너가 검출된 이미지 저장 (디버깅용)
                    img_with_corners = cv2.drawChessboardCorners(img.copy(), detected_size, corners2, ret)
                    debug_path = os.path.join(calibration_dir, f"debug_{os.path.basename(filepath)}")
                    cv2.imwrite(debug_path, img_with_corners)
                    
                    print(f"  ✓ 캘리브레이션 데이터로 사용됨")
                    corners_found = True
                    break
                else:
                    print(f"  ⚠ 기준 크기 {used_checkerboard_size}와 다른 크기 {detected_size} 검출됨 - 제외")
        
        if not corners_found:
            print(f"  ✗ 체커보드 코너 검출 실패")
            
            # 실패한 이미지 저장 (디버깅용)
            debug_path = os.path.join(calibration_dir, f"failed_{os.path.basename(filepath)}")
            cv2.imwrite(debug_path, img)
    
    print(f"\n{'='*50}")
    print(f"총 {len(successful_images)}개 이미지에서 체커보드 코너 검출 성공")
    print(f"사용된 체커보드 크기: {used_checkerboard_size}")
    print(f"{'='*50}")
    
    if len(successful_images) < 3:
        print("❌ 캘리브레이션을 위해서는 최소 3개 이상의 유효한 이미지가 필요합니다.")
        print("다음 사항을 확인해보세요:")
        print("1. 체커보드가 완전히 보이는지 확인")
        print("2. 조명이 균일한지 확인")
        print("3. 체커보드가 평평하고 왜곡되지 않았는지 확인")
        print("4. 이미지가 선명한지 확인 (블러되지 않음)")
        print("5. 체커보드 크기가 11x8 내부 코너인지 확인")
        return None
    
    # 마지막 이미지의 크기 가져오기
    img_shape = gray.shape[::-1]
    
    print("카메라 캘리브레이션을 시작합니다...")
    
    # 카메라 캘리브레이션 수행
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    if not ret:
        print("캘리브레이션 실패")
        return None
    
    print("✓ 캘리브레이션 완료")
    
    # 캘리브레이션 품질 평가
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    mean_error /= len(objpoints)
    
    # 캘리브레이션 결과 저장
    calibration_file = os.path.join(calibration_dir, "camera_parameter.npz")
    np.savez(calibration_file,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rvecs=rvecs,
             tvecs=tvecs,
             image_shape=img_shape,
             mean_error=mean_error,
             successful_images=successful_images,
             checkerboard_size=used_checkerboard_size)
    
    print(f"캘리브레이션 파라미터 저장 완료: {calibration_file}")
    
    # 결과 출력
    print("\n" + "="*80)
    print("🎯 카메라 캘리브레이션 결과")
    print("="*80)
    
    print(f"📊 기본 정보:")
    print(f"   사용된 이미지 수: {len(successful_images)}")
    print(f"   체커보드 크기: {used_checkerboard_size[0]}x{used_checkerboard_size[1]} (내부 코너)")
    print(f"   이미지 크기: {img_shape[0]} x {img_shape[1]}")
    print(f"   재투영 오차 (평균): {mean_error:.4f} pixels")
    
    print("\n📷 카메라 매트릭스 (Camera Matrix):")
    print(f"   [[{camera_matrix[0,0]:8.2f}, {camera_matrix[0,1]:8.2f}, {camera_matrix[0,2]:8.2f}],")
    print(f"    [{camera_matrix[1,0]:8.2f}, {camera_matrix[1,1]:8.2f}, {camera_matrix[1,2]:8.2f}],")
    print(f"    [{camera_matrix[2,0]:8.2f}, {camera_matrix[2,1]:8.2f}, {camera_matrix[2,2]:8.2f}]]")
    
    print("\n🎯 초점거리 (Focal Length):")
    print(f"   fx = {camera_matrix[0,0]:.2f} pixels")
    print(f"   fy = {camera_matrix[1,1]:.2f} pixels")
    
    print("\n📍 주점 (Principal Point):")
    print(f"   cx = {camera_matrix[0,2]:.2f} pixels")
    print(f"   cy = {camera_matrix[1,2]:.2f} pixels")
    
    print("\n🔍 왜곡 계수 (Distortion Coefficients):")
    print(f"   k1 = {dist_coeffs[0,0]:10.6f} (방사형 왜곡)")
    print(f"   k2 = {dist_coeffs[0,1]:10.6f} (방사형 왜곡)")
    print(f"   p1 = {dist_coeffs[0,2]:10.6f} (접선 왜곡)")
    print(f"   p2 = {dist_coeffs[0,3]:10.6f} (접선 왜곡)")
    print(f"   k3 = {dist_coeffs[0,4]:10.6f} (방사형 왜곡)")
    
    print("\n📏 추가 정보:")
    print(f"   화각 (수평): {2 * np.arctan(img_shape[0] / (2 * camera_matrix[0,0])) * 180 / np.pi:.1f}°")
    print(f"   화각 (수직): {2 * np.arctan(img_shape[1] / (2 * camera_matrix[1,1])) * 180 / np.pi:.1f}°")
    
    # 캘리브레이션 품질 평가
    print("\n📊 캘리브레이션 품질 평가:")
    if mean_error < 0.5:
        print("   ✅ 매우 좋음 (오차 < 0.5 pixels)")
    elif mean_error < 1.0:
        print("   ✅ 좋음 (오차 < 1.0 pixels)")
    elif mean_error < 2.0:
        print("   ⚠️ 보통 (오차 < 2.0 pixels)")
    else:
        print("   ❌ 나쁨 (오차 >= 2.0 pixels)")
    
    print("\n📁 저장된 파일:")
    print(f"   캘리브레이션 파라미터: {calibration_file}")
    print(f"   성공한 이미지 (디버그): {calibration_dir}/debug_IMG_*.jpg")
    print(f"   실패한 이미지 (디버그): {calibration_dir}/failed_IMG_*.jpg")
    
    print("\n📋 사용된 이미지 목록:")
    for img_path in successful_images:
        print(f"   ✓ {os.path.basename(img_path)}")
    
    print("="*80)
    
    return {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'mean_error': mean_error,
        'successful_images': successful_images,
        'checkerboard_size': used_checkerboard_size
    }

def main():
    print("🎯 카메라 캘리브레이션 프로그램 시작")
    print("체커보드 설정: 자동 크기 검출, 25mm 사각형")
    print("개선된 기능: 다중 크기 시도, 이미지 전처리, 상세 디버깅")
    print("-" * 60)
    
    try:
        result = calibrate_camera()
        
        if result is None:
            print("\n❌ 캘리브레이션 실패")
            print("디버깅을 위해 다음 파일들을 확인하세요:")
            print("  - ./rectify/calibration_data/debug_*.jpg (성공한 이미지)")
            print("  - ./rectify/calibration_data/failed_*.jpg (실패한 이미지)")
            return
        
        print("\n🎉 캘리브레이션이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()