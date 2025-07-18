import cv2
import numpy as np
import re
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_camera_intrinsics(npz_path):
    """
    .npz 파일로부터 카메라 내부 파라미터와 왜곡 계수를 로드합니다.
    """
    try:
        data = np.load(npz_path)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        
        # 카메라 파라미터 유효성 검증
        if camera_matrix.shape != (3, 3):
            raise ValueError(f"카메라 행렬 크기가 잘못됨: {camera_matrix.shape}")
        
        if dist_coeffs.shape[1] != 5:
            raise ValueError(f"왜곡 계수 개수가 잘못됨: {dist_coeffs.shape}")
        
        # 파라미터 추출
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        # 왜곡 계수 검증 및 경고
        k1, k2, p1, p2, k3 = dist_coeffs[0, :]
        if abs(k1) > 1.0 or abs(k2) > 1.0:
            logger.warning(f"큰 방사형 왜곡 계수 감지: k1={k1:.3f}, k2={k2:.3f}")
        if abs(k3) > 5.0:
            logger.warning(f"과도한 k3 왜곡 계수: {k3:.3f}")
        
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        
        # 정보 출력
        logger.info(f"카메라 파라미터 로드 완료: {npz_path}")
        logger.info(f"초점거리: fx={fx:.2f}, fy={fy:.2f}")
        logger.info(f"주점: cx={cx:.2f}, cy={cy:.2f}")
        
        return K, dist
        
    except Exception as e:
        logger.error(f"카메라 파라미터 로드 실패: {e}")
        raise

def load_correspondences_from_file(filepath):
    """
    대응점 파일에서 점들을 추출하고 유효성을 검증합니다.
    """
    points_left = []
    points_right = []
    
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                match = re.search(r'Left\(([\d.]+),\s*([\d.]+)\),\s*Right\(([\d.]+),\s*([\d.]+)\)', line)
                if match:
                    lx, ly, rx, ry = map(float, match.groups())
                    
                    # 좌표 유효성 검증
                    if all(0 <= coord <= 1000 for coord in [lx, ly, rx, ry]):
                        points_left.append([lx, ly])
                        points_right.append([rx, ry])
                    else:
                        logger.warning(f"라인 {line_num}: 비정상적인 좌표값 무시")
        
        if len(points_left) < 8:
            raise ValueError("최소 8개의 유효한 대응점이 필요합니다.")
        
        pts1 = np.array(points_left, dtype=np.float32)
        pts2 = np.array(points_right, dtype=np.float32)
        
        logger.info(f"대응점 {len(pts1)}개 로드 완료")
        return pts1, pts2
        
    except Exception as e:
        logger.error(f"대응점 로드 실패: {e}")
        raise

def compute_fundamental_matrix(pts1, pts2, debug=False):
    """
    개선된 Fundamental Matrix 계산
    """
    assert pts1.shape == pts2.shape, "입력 점의 크기가 일치하지 않습니다."
    assert pts1.shape[0] >= 8, "최소 8쌍 이상의 대응점이 필요합니다."
    
    # 원본 점들로 직접 계산 (정규화 없이)
    pts1 = pts1.astype(np.float32)
    pts2 = pts2.astype(np.float32)
    
    # 여러 방법으로 시도
    methods = [
        (cv2.FM_RANSAC, "RANSAC"),
        (cv2.FM_LMEDS, "LMEDS"),
        (cv2.FM_8POINT, "8-Point")
    ]
    
    best_F = None
    best_mask = None
    best_inliers = 0
    best_method = None
    
    for method, name in methods:
        try:
            if method == cv2.FM_8POINT:
                # 8-point 알고리즘은 RANSAC 파라미터 없음
                F, mask = cv2.findFundamentalMat(pts1, pts2, method=method)
            else:
                F, mask = cv2.findFundamentalMat(
                    pts1, pts2, 
                    method=method,
                    ransacReprojThreshold=1.0,
                    confidence=0.99,
                    maxIters=2000
                )
            
            if F is not None and mask is not None:
                inliers = np.sum(mask)
                logger.info(f"{name}: {inliers}/{len(mask)} inliers")
                
                if inliers > best_inliers:
                    best_F = F
                    best_mask = mask
                    best_inliers = inliers
                    best_method = name
        except Exception as e:
            logger.warning(f"{name} 방법 실패: {e}")
    
    if best_F is None:
        raise ValueError("모든 방법으로 Fundamental Matrix 계산 실패")
    
    # 조건수 개선을 위한 정규화
    U, S, Vt = np.linalg.svd(best_F)
    S[-1] = 0  # 마지막 특이값을 0으로 설정
    best_F = U @ np.diag(S) @ Vt
    
    inlier_ratio = best_inliers / len(best_mask)
    logger.info(f"최적 방법 {best_method}: {best_inliers}/{len(best_mask)} inliers ({inlier_ratio:.1%})")
    
    if debug:
        print(f"=== Fundamental Matrix (F) - {best_method} ===")
        print(best_F)
        print(f"F 행렬식: {np.linalg.det(best_F):.2e}")
        print(f"F 조건수: {np.linalg.cond(best_F):.2e}")
        print(f"F 특이값: {np.linalg.svd(best_F)[1]}")
    
    return best_F, best_mask

def normalize_points(points):
    """점들을 정규화하여 수치적 안정성을 향상시킵니다."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    scale = np.sqrt(2) / np.mean(np.linalg.norm(centered, axis=1))
    return centered * scale + centroid

def compute_pose_from_essential(F, K1, K2, pts1, pts2, dist1, dist2, debug=False):
    """
    개선된 Essential Matrix 및 pose 계산
    """
    # Essential Matrix 계산
    E = K2.T @ F @ K1
    
    # 점 정규화 (올바른 방법)
    pts1_undist = cv2.undistortPoints(pts1.reshape(-1,1,2), K1, dist1, None, K1)
    pts2_undist = cv2.undistortPoints(pts2.reshape(-1,1,2), K2, dist2, None, K2)
    
    pts1_norm = cv2.undistortPoints(pts1_undist, K1, None)
    pts2_norm = cv2.undistortPoints(pts2_undist, K2, None)
    
    # Pose 복원
    retval, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm)
    
    if retval == 0:
        raise ValueError("Pose 복원 실패")
    
    # Pose 품질 검증
    baseline = np.linalg.norm(t)
    if baseline < 0.1:
        logger.warning(f"작은 베이스라인: {baseline:.3f}")
    
    # 회전 행렬 검증
    if abs(np.linalg.det(R) - 1.0) > 1e-6:
        logger.warning("회전 행렬이 직교행렬이 아닙니다.")
    
    if debug:
        print("=== Essential Matrix (E) ===")
        print(E)
        print(f"E 특이값: {np.linalg.svd(E)[1]}")
        print("\n=== Rotation Matrix (R) ===")
        print(R)
        print(f"R 행렬식: {np.linalg.det(R):.6f}")
        print("\n=== Translation Vector (t) ===")
        print(t)
        print(f"베이스라인: {baseline:.3f}")
        print(f"유효한 pose 개수: {retval}")
    
    return E, R, t, mask

def setup_rectification(K1, D1, K2, D2, R, t, image_size, alpha=0.8):
    """
    개선된 스테레오 정렬 설정
    """
    try:
        # 다양한 alpha 값으로 시도
        best_alpha = alpha
        best_roi_area = 0
        best_results = None
        
        for test_alpha in [0.0, 0.3, 0.5, 0.8, 1.0]:
            try:
                # 스테레오 정렬 계산
                R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                    K1, D1, K2, D2, image_size, R, t,
                    flags=cv2.CALIB_ZERO_DISPARITY,
                    alpha=test_alpha
                )
                
                # ROI 면적 계산
                roi1_area = roi1[2] * roi1[3] if roi1[2] > 0 and roi1[3] > 0 else 0
                roi2_area = roi2[2] * roi2[3] if roi2[2] > 0 and roi2[3] > 0 else 0
                total_roi_area = roi1_area + roi2_area
                
                if total_roi_area > best_roi_area:
                    best_roi_area = total_roi_area
                    best_alpha = test_alpha
                    best_results = (R1, R2, P1, P2, Q, roi1, roi2)
                
                logger.info(f"Alpha {test_alpha}: ROI 면적 = {total_roi_area}")
                
            except Exception as e:
                logger.warning(f"Alpha {test_alpha} 실패: {e}")
                continue
        
        if best_results is None:
            # 기본 설정으로 강제 실행
            logger.warning("최적 alpha를 찾지 못함. 기본값으로 실행")
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                K1, D1, K2, D2, image_size, R, t,
                flags=0,  # 플래그 제거
                alpha=1.0
            )
        else:
            R1, R2, P1, P2, Q, roi1, roi2 = best_results
            logger.info(f"최적 alpha: {best_alpha}")
        
        # 정렬 맵 생성
        map1_x, map1_y = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, image_size, cv2.CV_32FC1
        )
        map2_x, map2_y = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2, image_size, cv2.CV_32FC1
        )
        
        # 최종 ROI 검증
        roi1_area = roi1[2] * roi1[3] if roi1[2] > 0 and roi1[3] > 0 else 0
        roi2_area = roi2[2] * roi2[3] if roi2[2] > 0 and roi2[3] > 0 else 0
        total_area = image_size[0] * image_size[1]
        
        logger.info(f"최종 유효 영역: 좌측 {roi1_area/total_area:.1%}, 우측 {roi2_area/total_area:.1%}")
        logger.info(f"ROI 좌측: {roi1}, 우측: {roi2}")
        
        # ROI가 여전히 0이면 전체 이미지를 ROI로 설정
        if roi1_area == 0:
            roi1 = (0, 0, image_size[0], image_size[1])
            logger.info("좌측 ROI를 전체 이미지로 설정")
        if roi2_area == 0:
            roi2 = (0, 0, image_size[0], image_size[1])
            logger.info("우측 ROI를 전체 이미지로 설정")
        
        return map1_x, map1_y, map2_x, map2_y, Q, roi1, roi2
        
    except Exception as e:
        logger.error(f"스테레오 정렬 설정 실패: {e}")
        raise

def validate_epipolar_geometry(img1, img2, F, pts1, pts2):
    """
    에피폴라 기하학 검증
    """
    errors = []
    for p1, p2 in zip(pts1, pts2):
        # 호모지니어스 좌표
        p1_h = np.array([p1[0], p1[1], 1.0])
        p2_h = np.array([p2[0], p2[1], 1.0])
        
        # 에피폴라 제약 조건: p2^T * F * p1 = 0
        error = p2_h.T @ F @ p1_h
        errors.append(abs(error))
    
    mean_error = np.mean(errors)
    logger.info(f"에피폴라 제약 조건 평균 오차: {mean_error:.6f}")
    
    if mean_error > 0.1:
        logger.warning("에피폴라 제약 조건 오차가 큽니다.")
    
    return errors

def run_live_rectification(K1, dist1, K2, dist2, R, t, image_size, pts1=None, pts2=None):
    """
    개선된 실시간 스테레오 정렬
    """
    # 정렬 설정
    map1_x, map1_y, map2_x, map2_y, Q, roi1, roi2 = setup_rectification(
        K1, dist1, K2, dist2, R, t, image_size, alpha=0.8
    )
    
    # 카메라 초기화
    cap1 = cv2.VideoCapture(2)
    cap2 = cv2.VideoCapture(0)
    
    # 카메라 설정
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])
    
    if not cap1.isOpened() or not cap2.isOpened():
        logger.error("카메라 초기화 실패")
        return
    
    logger.info("실시간 정렬 시작. 'q' 키로 종료, 's' 키로 스크린샷 저장, 'r' 키로 ROI 토글")
    
    frame_count = 0
    show_roi = True
    
    try:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                logger.warning("프레임 읽기 실패")
                break
            
            # 원본 프레임 (대응점 표시)
            orig1, orig2 = frame1.copy(), frame2.copy()
            if pts1 is not None and pts2 is not None:
                orig1, orig2 = draw_correspondences(orig1, orig2, pts1, pts2, (0, 255, 0))
            
            # 정렬된 프레임
            rectified1 = cv2.remap(frame1, map1_x, map1_y, cv2.INTER_LINEAR)
            rectified2 = cv2.remap(frame2, map2_x, map2_y, cv2.INTER_LINEAR)
            
            # ROI 크롭 및 표시
            if show_roi:
                # ROI 영역 크롭
                if roi1[2] > 0 and roi1[3] > 0:
                    roi_rectified1 = rectified1[roi1[1]:roi1[1]+roi1[3], roi1[0]:roi1[0]+roi1[2]]
                    cv2.rectangle(rectified1, (roi1[0], roi1[1]), 
                                (roi1[0]+roi1[2], roi1[1]+roi1[3]), (0, 255, 255), 2)
                else:
                    roi_rectified1 = rectified1
                
                if roi2[2] > 0 and roi2[3] > 0:
                    roi_rectified2 = rectified2[roi2[1]:roi2[1]+roi2[3], roi2[0]:roi2[0]+roi2[2]]
                    cv2.rectangle(rectified2, (roi2[0], roi2[1]), 
                                (roi2[0]+roi2[2], roi2[1]+roi2[3]), (0, 255, 255), 2)
                else:
                    roi_rectified2 = rectified2
            
            # 에피폴라 라인 표시 (더 촘촘하게)
            stacked_rectified = np.hstack((rectified1, rectified2))
            for y in range(10, stacked_rectified.shape[0], 30):
                cv2.line(stacked_rectified, (0, y), (stacked_rectified.shape[1], y), (0, 255, 0), 1)
            
            # 중앙선 표시
            center_y = stacked_rectified.shape[0] // 2
            cv2.line(stacked_rectified, (0, center_y), (stacked_rectified.shape[1], center_y), (0, 0, 255), 2)
            
            # 결과 표시
            stacked_original = np.hstack((orig1, orig2))
            display = np.vstack((stacked_original, stacked_rectified))
            
            # 정보 표시
            info_text = [
                f"Frame: {frame_count}",
                f"ROI1: {roi1[2]}x{roi1[3]} at ({roi1[0]},{roi1[1]})",
                f"ROI2: {roi2[2]}x{roi2[3]} at ({roi2[0]},{roi2[1]})",
                "Keys: q=quit, s=save, r=toggle ROI"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(display, text, (10, 25 + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 상태 표시
            cv2.putText(display, "Original + Correspondences", (10, stacked_original.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display, "Rectified + Epipolar Lines", (10, stacked_original.shape[0] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Stereo Rectification", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 또는 ESC
                break
            elif key == ord('s'):  # 스크린샷 저장
                timestamp = frame_count
                cv2.imwrite(f"rectified_full_{timestamp:04d}.png", display)
                cv2.imwrite(f"rectified_left_{timestamp:04d}.png", rectified1)
                cv2.imwrite(f"rectified_right_{timestamp:04d}.png", rectified2)
                logger.info(f"스크린샷 저장: frame {timestamp:04d}")
            elif key == ord('r'):  # ROI 토글
                show_roi = not show_roi
                logger.info(f"ROI 표시: {'ON' if show_roi else 'OFF'}")
            elif key == ord('h'):  # 도움말
                help_text = [
                    "=== 키보드 단축키 ===",
                    "q, ESC: 종료",
                    "s: 스크린샷 저장",
                    "r: ROI 표시 토글",
                    "h: 도움말"
                ]
                for text in help_text:
                    logger.info(text)
            
            frame_count += 1
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        logger.info("카메라 및 창 해제 완료")

def draw_correspondences(img1, img2, pts1, pts2, color=(0, 255, 0)):
    """향상된 대응점 표시"""
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts1, pts2)):
        # 원 그리기
        cv2.circle(img1, (int(x1), int(y1)), 3, color, -1)
        cv2.circle(img2, (int(x2), int(y2)), 3, color, -1)
        
        # 번호 표시 (선택적)
        if i < 10:  # 처음 10개만 번호 표시
            cv2.putText(img1, str(i), (int(x1)+5, int(y1)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            cv2.putText(img2, str(i), (int(x2)+5, int(y2)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    return img1, img2

def main():
    """메인 실행 함수"""
    try:
        # 설정
        file_path = './rectify/correspond_point.txt'
        left_param_path = './rectify/calibration_data/left_camera_parameter.npz'
        right_param_path = './rectify/calibration_data/right_camera_parameter.npz'
        
        # 1. 데이터 로드
        logger.info("=== 데이터 로드 시작 ===")
        pts1, pts2 = load_correspondences_from_file(file_path)
        K1, dist1 = load_camera_intrinsics(left_param_path)
        K2, dist2 = load_camera_intrinsics(right_param_path)
        
        # 2. Fundamental Matrix 계산
        logger.info("=== Fundamental Matrix 계산 ===")
        F, mask = compute_fundamental_matrix(pts1, pts2, debug=True)
        
        # 3. 에피폴라 기하학 검증
        logger.info("=== 에피폴라 기하학 검증 ===")
        validate_epipolar_geometry(None, None, F, pts1, pts2)
        
        # 4. Pose 계산
        logger.info("=== Pose 계산 ===")
        E, R, t, pose_mask = compute_pose_from_essential(F, K1, K2, pts1, pts2, dist1, dist2, debug=True)
        
        # 5. 이미지 크기 확인
        cap_temp = cv2.VideoCapture(0)
        ret, frame = cap_temp.read()
        cap_temp.release()
        
        if not ret:
            raise RuntimeError("웹캠에서 프레임을 읽을 수 없습니다.")
        
        image_size = (frame.shape[1], frame.shape[0])
        logger.info(f"이미지 크기: {image_size}")
        
        # 6. 실시간 정렬 실행
        logger.info("=== 실시간 스테레오 정렬 시작 ===")
        run_live_rectification(K1, dist1, K2, dist2, R, t, image_size, pts1, pts2)
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
