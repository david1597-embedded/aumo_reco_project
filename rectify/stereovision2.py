import cv2
import numpy as np
import re
import logging
from cv2.ximgproc import createDisparityWLSFilter, createRightMatcher

prev_depth = None
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
        
        if camera_matrix.shape != (3, 3):
            raise ValueError(f"카메라 행렬 크기가 잘못됨: {camera_matrix.shape}")
        
        dist_coeffs = np.squeeze(dist_coeffs)
        if dist_coeffs.shape != (5,) or len(dist_coeffs.shape) != 1:
            raise ValueError(f"왜곡 계수 형태가 잘못됨: {dist_coeffs.shape}, 기대: (5,)")
        
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        k1, k2, p1, p2, k3 = dist_coeffs
        
        if abs(k1) > 1.0 or abs(k2) > 1.0:
            logger.warning(f"큰 방사형 왜곡 계수 감지: k1={k1:.3f}, k2={k2:.3f}")
        if abs(k3) > 5.0:
            logger.warning(f"과도한 k3 왜곡 계수: {k3:.3f}")
        
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        
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
    
    pts1 = pts1.astype(np.float32)
    pts2 = pts2.astype(np.float32)
    
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
    
    U, S, Vt = np.linalg.svd(best_F)
    S[-1] = 0
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
    E = K2.T @ F @ K1
    
    pts1_undist = cv2.undistortPoints(pts1.reshape(-1,1,2), K1, dist1, None, K1)
    pts2_undist = cv2.undistortPoints(pts2.reshape(-1,1,2), K2, dist2, None, K2)
    
    pts1_norm = cv2.undistortPoints(pts1_undist, K1, None)
    pts2_norm = cv2.undistortPoints(pts2_undist, K2, None)
    
    retval, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm)
    
    if retval == 0:
        raise ValueError("Pose 복원 실패")
    
    baseline = np.linalg.norm(t)
    if baseline < 0.1:
        logger.warning(f"작은 베이스라인: {baseline:.3f}")
    
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

def compute_Q_matrix(K1, K2, t, cx_left, cx_right, cy_left):
    """
    Q 행렬을 수동으로 계산합니다.
    """
    try:
        if not all(np.isscalar(x) for x in [cx_left, cx_right, cy_left]):
            raise ValueError(f"주점 좌표는 스칼라여야 합니다: cx_left={cx_left}, cx_right={cx_right}, cy_left={cy_left}")
        
        f = (K1[0, 0] + K2[0, 0]) / 2
        cx = float(cx_left)
        cx_prime = float(cx_right)
        cy = float(cy_left)
        T_x = abs(float(t[0]))
        if T_x < 1e-6:
            raise ValueError(f"베이스라인이 너무 작음: T_x={T_x}")
        
        Q = np.array([
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 0.0, f],
            [0.0, 0.0, -1.0/T_x, (cx - cx_prime)/T_x]
        ], dtype=np.float64)
        
        logger.info(f"수동 계산된 Q 행렬:\n{Q}")
        return Q
    
    except Exception as e:
        logger.error(f"Q 행렬 계산 실패: {e}")
        raise

def setup_rectification(K1, D1, K2, D2, R, t, image_size, alpha=0.8):
    """
    개선된 스테레오 정렬 설정
    """
    try:
        best_alpha = alpha
        best_roi_area = 0
        best_results = None
        
        t_normalized = t / np.linalg.norm(t)
        
        for test_alpha in [0.0, 0.3, 0.5, 0.8, 1.0]:
            try:
                R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                    K1, D1, K2, D2, image_size, R, t_normalized,
                    alpha=test_alpha
                )
                
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
            logger.warning("최적 alpha를 찾지 못함. 기본값으로 실행")
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                K1, D1, K2, D2, image_size, R, t_normalized,
                alpha=1.0
            )
        else:
            R1, R2, P1, P2, Q, roi1, roi2 = best_results
            logger.info(f"최적 alpha: {best_alpha}")
        
        Q_manual = compute_Q_matrix(K1, K2, t_normalized, K1[0, 2], K2[0, 2], K1[1, 2])
        
        map1_x, map1_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map2_x, map2_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
        
        # 매핑 배열 검증
        for map_arr, name in [(map1_x, "map1_x"), (map1_y, "map1_y"), (map2_x, "map2_x"), (map2_y, "map2_y")]:
            if map_arr.shape != image_size[::-1]:
                raise ValueError(f"{name}의 크기가 이미지 크기 {image_size}와 일치하지 않습니다: {map_arr.shape}")
        logger.info(f"매핑 배열 크기: map1_x={map1_x.shape}, map1_y={map1_y.shape}, map2_x={map2_x.shape}, map2_y={map2_y.shape}")
        
        roi1_area = roi1[2] * roi1[3] if roi1[2] > 0 and roi1[3] > 0 else 0
        roi2_area = roi2[2] * roi2[3] if roi2[2] > 0 and roi2[3] > 0 else 0
        total_area = image_size[0] * image_size[1]
        
        logger.info(f"최종 유효 영역: 좌측 {roi1_area/total_area:.1%}, 우측 {roi2_area/total_area:.1%}")
        logger.info(f"ROI 좌측: {roi1}, 우측: {roi2}")
        
        if roi1_area == 0:
            roi1 = (0, 0, image_size[0], image_size[1])
            logger.info("좌측 ROI를 전체 이미지로 설정")
        if roi2_area == 0:
            roi2 = (0, 0, image_size[0], image_size[1])
            logger.info("우측 ROI를 전체 이미지로 설정")
        
        return map1_x, map1_y, map2_x, map2_y, Q_manual, roi1, roi2, P1, P2
        
    except Exception as e:
        logger.error(f"스테레오 정렬 설정 실패: {e}")
        raise

def validate_epipolar_geometry(img1, img2, F, pts1, pts2):
    """
    에피폴라 기하학 검증
    """
    errors = []
    for p1, p2 in zip(pts1, pts2):
        p1_h = np.array([p1[0], p1[1], 1.0])
        p2_h = np.array([p2[0], p2[1], 1.0])
        error = p2_h.T @ F @ p1_h
        errors.append(abs(error))
    
    mean_error = np.mean(errors)
    logger.info(f"에피폴라 제약 조건 평균 오차: {mean_error:.6f}")
    if mean_error > 0.1:
        logger.warning("에피폴라 제약 조건 오차가 큽니다.")
    
    return errors

def compute_depth_from_disparity(disparity, Q):
    """
    시차 맵에서 Q 행렬을 사용해 깊이 맵을 계산
    """
    try:
        if len(disparity.shape) != 2:
            raise ValueError(f"Disparity 맵은 2D 배열이어야 합니다: {disparity.shape}")
        if Q.shape != (4, 4):
            raise ValueError(f"Q 행렬은 4x4이어야 합니다: {Q.shape}")
        
        h, w = disparity.shape
        
        # 유효한 disparity가 있는 픽셀만 처리
        valid_mask = disparity > 0
        valid_indices = np.where(valid_mask)
        
        if len(valid_indices[0]) == 0:
            logger.warning("유효한 disparity 값이 없습니다.")
            return np.zeros((h, w), dtype=np.float32)
        
        # 유효한 픽셀의 좌표와 disparity 값 추출
        y_coords = valid_indices[0]
        x_coords = valid_indices[1]
        disp_values = disparity[valid_mask]
        
        # homogeneous coordinates 생성 (x, y, disparity, 1)
        points_4d = np.column_stack([
            x_coords.astype(np.float32),
            y_coords.astype(np.float32), 
            disp_values.astype(np.float32),
            np.ones(len(x_coords), dtype=np.float32)
        ])
        
        # Q 행렬을 사용하여 3D 점 계산
        Q_float32 = Q.astype(np.float32)
        points_3d = np.dot(points_4d, Q_float32.T)
        
        # homogeneous coordinates에서 실제 3D 좌표로 변환
        # W 성분으로 나누어 정규화
        w_coords = points_3d[:, 3]
        valid_w = w_coords != 0
        
        depth_map = np.zeros((h, w), dtype=np.float32)
        
        if np.any(valid_w):
            # W가 0이 아닌 점들만 처리
            valid_3d_indices = np.where(valid_w)[0]
            normalized_z = points_3d[valid_3d_indices, 2] / w_coords[valid_3d_indices]
            
            # 원래 이미지 좌표에 깊이 값 할당
            valid_y = y_coords[valid_3d_indices]
            valid_x = x_coords[valid_3d_indices]
            depth_map[valid_y, valid_x] = normalized_z
        
        return depth_map
    
    except Exception as e:
        logger.error(f"깊이 맵 계산 실패: {e}")
        raise

def compute_depth_map(rectified_left, rectified_right, Q, use_wls=True):
    """
    시차 맵에서 깊이 맵 계산 (Q 행렬 사용)
    """
    global prev_depth
    
    min_disp = 0
    num_disp = 16 * 6
    block_size = 7
    
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    if use_wls:
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        disp_left = left_matcher.compute(rectified_left, rectified_right)
        disp_right = right_matcher.compute(rectified_right, rectified_left)
        
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(8000)
        wls_filter.setSigmaColor(1.5)
        
        filtered_disp = wls_filter.filter(disp_left, rectified_left, None, disp_right)
        disparity = filtered_disp.astype(np.float32) / 16.0
    else:
        disparity = left_matcher.compute(rectified_left, rectified_right).astype(np.float32) / 16.0
    
    valid_mask = disparity > 0.0
    disparity = np.where(valid_mask, disparity, np.nan)
    
    depth = compute_depth_from_disparity(disparity, Q)
    depth = np.where(np.isfinite(depth), depth, 0)
    
    if prev_depth is not None and prev_depth.shape == depth.shape:
        alpha = 0.8
        depth = alpha * depth + (1 - alpha) * prev_depth
    prev_depth = depth.copy()
    
    depth = cv2.bilateralFilter(depth.astype(np.float32), 5, 75, 75)
    depth = cv2.GaussianBlur(depth, (5, 5), 0)
    
    depth_min, depth_max = 200, 4000
    depth = np.clip(depth, depth_min, depth_max)
    
    depth_visual = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
    
    return depth, depth_colormap

def draw_correspondences(img1, img2, pts1, pts2, color=(0, 255, 0)):
    """향상된 대응점 표시"""
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts1, pts2)):
        cv2.circle(img1, (int(x1), int(y1)), 3, color, -1)
        cv2.circle(img2, (int(x2), int(y2)), 3, color, -1)
        if i < 10:
            cv2.putText(img1, str(i), (int(x1)+5, int(y1)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            cv2.putText(img2, str(i), (int(x2)+5, int(y2)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    return img1, img2

def run_live_rectification(K1, dist1, K2, dist2, R, t, image_size, pts1=None, pts2=None):
    """
    개선된 실시간 스테레오 정렬
    """
    map1_x, map1_y, map2_x, map2_y, Q, roi1, roi2, P1, P2 = setup_rectification(
        K1, dist1, K2, dist2, R, t, image_size, alpha=0.8
    )
    
    cap1 = cv2.VideoCapture(2)
    cap2 = cv2.VideoCapture(0)
    
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
            
            orig1, orig2 = frame1.copy(), frame2.copy()
            if pts1 is not None and pts2 is not None:
                orig1, orig2 = draw_correspondences(orig1, orig2, pts1, pts2, (0, 255, 0))
            
            # cv2.remap 호출 수정
            rectified1 = cv2.remap(frame1, map1_x, map1_y, cv2.INTER_LINEAR)
            rectified2 = cv2.remap(frame2, map2_x, map2_y, cv2.INTER_LINEAR)
            
            depth, depth_colormap = compute_depth_map(rectified1, rectified2, Q, True)
            cv2.imshow("Depth Map", depth_colormap)
            
            if show_roi:
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
            
            stacked_rectified = np.hstack((rectified1, rectified2))
            for y in range(10, stacked_rectified.shape[0], 30):
                cv2.line(stacked_rectified, (0, y), (stacked_rectified.shape[1], y), (0, 255, 0), 1)
            
            center_y = stacked_rectified.shape[0] // 2
            cv2.line(stacked_rectified, (0, center_y), (stacked_rectified.shape[1], center_y), (0, 0, 255), 2)
            
            stacked_original = np.hstack((orig1, orig2))
            display = np.vstack((stacked_original, stacked_rectified))
            
            info_text = [
                f"Frame: {frame_count}",
                f"ROI1: {roi1[2]}x{roi1[3]} at ({roi1[0]},{roi1[1]})",
                f"ROI2: {roi2[2]}x{roi2[3]} at ({roi2[0]},{roi2[1]})",
                "Keys: q=quit, s=save, r=toggle ROI"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(display, text, (10, 25 + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(display, "Original + Correspondences", (10, stacked_original.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display, "Rectified + Epipolar Lines", (10, stacked_original.shape[0] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Stereo Rectification", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                timestamp = frame_count
                cv2.imwrite(f"rectified_full_{timestamp:04d}.png", display)
                cv2.imwrite(f"rectified_left_{timestamp:04d}.png", rectified1)
                cv2.imwrite(f"rectified_right_{timestamp:04d}.png", rectified2)
                logger.info(f"스크린샷 저장: frame {timestamp:04d}")
            elif key == ord('r'):
                show_roi = not show_roi
                logger.info(f"ROI 표시: {'ON' if show_roi else 'OFF'}")
            elif key == ord('h'):
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

def main():
    """메인 실행 함수"""
    global prev_depth
    prev_depth = None
    try:
        file_path = './rectify/correspond_point.txt'
        left_param_path = './rectify/calibration_data/left_camera_parameter.npz'
        right_param_path = './rectify/calibration_data/right_camera_parameter.npz'
        
        logger.info("=== 데이터 로드 시작 ===")
        pts1, pts2 = load_correspondences_from_file(file_path)
        K1, dist1 = load_camera_intrinsics(left_param_path)
        K2, dist2 = load_camera_intrinsics(right_param_path)
        
        logger.info("=== Fundamental Matrix 계산 ===")
        F, mask = compute_fundamental_matrix(pts1, pts2, debug=True)
        
        logger.info("=== 에피폴라 기하학 검증 ===")
        validate_epipolar_geometry(None, None, F, pts1, pts2)
        
        logger.info("=== Pose 계산 ===")
        E, R, t, pose_mask = compute_pose_from_essential(F, K1, K2, pts1, pts2, dist1, dist2, debug=True)
        
        cap_temp = cv2.VideoCapture(0)
        ret, frame = cap_temp.read()
        cap_temp.release()
        
        if not ret:
            raise RuntimeError("웹캠에서 프레임을 읽을 수 없습니다.")
        
        image_size = (frame.shape[1], frame.shape[0])
        logger.info(f"이미지 크기: {image_size}")
        
        logger.info("=== 실시간 스테레오 정렬 시작 ===")
        run_live_rectification(K1, dist1, K2, dist2, R, t, image_size, pts1, pts2)
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()