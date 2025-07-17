import cv2
import numpy as np

def calculate_fundamental_matrix(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Calculate Fundamental Matrix using 8-point algorithm with normalization
    """
    # Compute normalization matrix H1
    x1_mean_x = np.mean(x1[:, 0])
    x1_mean_y = np.mean(x1[:, 1])
    centroid = np.array([x1_mean_x, x1_mean_y])
    
    dist = [np.sqrt((x1[i, 0] - centroid[0])**2 + (x1[i, 1] - centroid[1])**2) for i in range(x1.shape[0])]
    mean_dist = np.mean(dist)
    
    H1norm = np.array([
        [np.sqrt(2) / mean_dist, 0, -np.sqrt(2) / mean_dist * centroid[0]],
        [0, np.sqrt(2) / mean_dist, -np.sqrt(2) / mean_dist * centroid[1]],
        [0, 0, 1]
    ])
    
    # Compute normalization matrix H2
    x2_mean_x = np.mean(x2[:, 0])
    x2_mean_y = np.mean(x2[:, 1])
    centroid2 = np.array([x2_mean_x, x2_mean_y])
    
    dist2 = [np.sqrt((x2[i, 0] - centroid2[0])**2 + (x2[i, 1] - centroid2[1])**2) for i in range(x2.shape[0])]
    mean_dist2 = np.mean(dist2)
    
    H2norm = np.array([
        [np.sqrt(2) / mean_dist2, 0, -np.sqrt(2) / mean_dist2 * centroid2[0]],
        [0, np.sqrt(2) / mean_dist2, -np.sqrt(2) / mean_dist2 * centroid2[1]],
        [0, 0, 1]
    ])
    
    # Apply normalization
    x1_norm = (H1norm @ x1.T).T
    x2_norm = (H2norm @ x2.T).T
    
    # Construct matrix A for 8-point algorithm
    A = np.zeros((8, 9))
    A[:, 0] = x1_norm[:, 0] * x2_norm[:, 0]
    A[:, 1] = x1_norm[:, 1] * x2_norm[:, 0]
    A[:, 2] = x2_norm[:, 0]
    A[:, 3] = x1_norm[:, 0] * x2_norm[:, 1]
    A[:, 4] = x1_norm[:, 1] * x2_norm[:, 1]
    A[:, 5] = x2_norm[:, 1]
    A[:, 6] = x1_norm[:, 0]
    A[:, 7] = x1_norm[:, 1]
    A[:, 8] = np.ones(8)
    
    # First SVD to get F0
    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1, :]
    F0 = f.reshape(3, 3)
    
    # Second SVD to enforce rank-2 constraint
    U2, S2, Vt2 = np.linalg.svd(F0)
    S2[2] = 0  # Set smallest singular value to 0
    F_norm = U2 @ np.diag(S2) @ Vt2
    
    # Denormalize
    F = H2norm.T @ F_norm @ H1norm
    
    return F

def draw_epipolar_lines(img_left: np.ndarray, img_right: np.ndarray, 
                       x1: np.ndarray, x2: np.ndarray, F: np.ndarray):
    """
    Draw epipolar lines on stereo images
    """
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (125, 125, 0), (125, 0, 125)
    ]
    
    img_left_copy = img_left.copy()
    img_right_copy = img_right.copy()
    
    for i in range(x1.shape[0]):
        color = colors[i % len(colors)]
        
        # Epipolar line in left image
        x2_vec = x2[i, :]
        epipolar_line_left = F.T @ x2_vec
        a_l, b_l, c_l = epipolar_line_left
        
        if abs(b_l) > 1e-6:
            pt_l1 = (0, int(-c_l / b_l))
            pt_l2 = (img_left.shape[1], int((-a_l * img_left.shape[1] - c_l) / b_l))
            cv2.line(img_left_copy, pt_l1, pt_l2, color, 2)
        
        # Epipolar line in right image
        x1_vec = x1[i, :]
        epipolar_line_right = F @ x1_vec
        a_r, b_r, c_r = epipolar_line_right
        
        if abs(b_r) > 1e-6:
            pt_r1 = (0, int(-c_r / b_r))
            pt_r2 = (img_right.shape[1], int((-a_r * img_right.shape[1] - c_r) / b_r))
            cv2.line(img_right_copy, pt_r1, pt_r2, color, 2)
        
        # Draw points
        cv2.circle(img_left_copy, (int(x1[i, 0]), int(x1[i, 1])), 5, color, -1)
        cv2.circle(img_right_copy, (int(x2[i, 0]), int(x2[i, 1])), 5, color, -1)
    
    cv2.imshow('Epipolar Lines Left', img_left_copy)
    cv2.imshow('Epipolar Lines Right', img_right_copy)
    cv2.moveWindow('Epipolar Lines Left', 100, 100)
    cv2.moveWindow('Epipolar Lines Right', 750, 100)
    
    print("Press any key to close epipolar line windows")
    cv2.waitKey(0)
    cv2.destroyWindow('Epipolar Lines Left')
    cv2.destroyWindow('Epipolar Lines Right')

def decompose_fundamental_matrix(K: np.ndarray, F: np.ndarray, 
                               x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose Fundamental Matrix to get R and t
    """
    E = K.T @ F @ K
    U, S, Vt = np.linalg.svd(E)
    
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    
    # First solution
    t = U[:, 2]
    R = U @ W @ Vt
    if np.linalg.det(R) < 0:
        t = -t
        R = -R
    
    return R, t

def compute_stereo_homography(R: np.ndarray, t: np.ndarray, 
                            K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute homography matrices for stereo rectification
    """
    rx = t / np.linalg.norm(t)
    rz_tilde = np.array([0, 0, 1])
    tmp = rz_tilde - np.dot(rz_tilde, rx) * rx
    rz = tmp / np.linalg.norm(tmp)
    ry = np.cross(rz, rx)
    
    R_rect = np.vstack([rx, ry, rz])
    
    H1 = K @ R_rect @ np.linalg.inv(K)
    H2 = K @ R_rect @ R.T @ np.linalg.inv(K)
    
    return H1, H2

def apply_homography(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography transformation to image
    """
    return cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))

def get_camera_parameters():
    """
    Get camera parameters and new correspondence points
    """
    # Using average intrinsic parameters for simplicity
    K = np.array([
        [(651.81287125788776 + 667.65289322504) / 2, 0, (294.45891563316155 + 322.514281299226) / 2],
        [0, (650.13727059174425 + 668.097062322412) / 2, (255.48924768945898 + 199.28081443084153) / 2],
        [0, 0, 1]
    ])
    
    # New correspondence points in homogeneous coordinates
    x1_points = np.array([
        [311.71, 373.82, 1],
        [312.53, 361.74, 1],
        [313.64, 349.23, 1],
        [313.83, 328.76, 1],
        [314.62, 239.71, 1],
        [314.86, 294.31, 1],
        [315.59, 265.64, 1],
        [318.52, 219.09, 1]
    ])
    
    x2_points = np.array([
        [15.03, 287.03, 1],
        [14.52, 275.54, 1],
        [14.20, 264.21, 1],
        [11.88, 243.86, 1],
        [7.26, 161.80, 1],
        [11.16, 211.00, 1],
        [9.46, 185.58, 1],
        [10.00, 142.43, 1]
    ])
    
    return K, x1_points, x2_points

def run_stereo_webcam():
    """
    Run real-time stereo rectification with webcams 0 (left) and 2 (right)
    """
    cap_left = cv2.VideoCapture(0)  # Left camera
    cap_right = cv2.VideoCapture(2)  # Right camera
    
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Could not open webcams")
        return
    
    # Set resolution
    width, height = 640, 480
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Get calibration parameters
    K, x1_points, x2_points = get_camera_parameters()
    F = calculate_fundamental_matrix(x1_points, x2_points)
    R, t = decompose_fundamental_matrix(K, F, x1_points, x2_points)
    H1, H2 = compute_stereo_homography(R, t, K)
    
    # Create windows
    cv2.namedWindow('Original Left', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Original Right', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Rectified Left', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Rectified Right', cv2.WINDOW_NORMAL)
    
    # Position windows
    cv2.moveWindow('Original Left', 0, 0)
    cv2.moveWindow('Original Right', 650, 0)
    cv2.moveWindow('Rectified Left', 0, 550)
    cv2.moveWindow('Rectified Right', 650, 550)
    
    print("Press 'q' to quit, 'c' to capture and show epipolar lines")
    print("Press 's' to save current frame")
    
    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not ret_left or not ret_right:
            print("Error: Could not read frames")
            break
        
        # Apply rectification
        frame_left_rect = apply_homography(frame_left, H1)
        frame_right_rect = apply_homography(frame_right, H2)
        
        # Add horizontal lines to show rectification
        frame_left_rect_lines = frame_left_rect.copy()
        frame_right_rect_lines = frame_right_rect.copy()
        for y in range(0, height, 50):
            cv2.line(frame_left_rect_lines, (0, y), (width, y), (0, 255, 0), 1)
            cv2.line(frame_right_rect_lines, (0, y), (width, y), (0, 255, 0), 1)
        
        # Display frames
        cv2.imshow('Original Left', frame_left)
        cv2.imshow('Original Right', frame_right)
        cv2.imshow('Rectified Left', frame_left_rect_lines)
        cv2.imshow('Rectified Right', frame_right_rect_lines)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("Capturing frame and showing epipolar lines...")
            draw_epipolar_lines(frame_left, frame_right, x1_points, x2_points, F)
        elif key == ord('s'):
            cv2.imwrite('left_original.jpg', frame_left)
            cv2.imwrite('right_original.jpg', frame_right)
            cv2.imwrite('left_rectified.jpg', frame_left_rect)
            cv2.imwrite('right_rectified.jpg', frame_right_rect)
            print("Frames saved!")
    
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

def main():
    print("Stereo Rectification with New Correspondence Points")
    print("Camera 0 = Left, Camera 2 = Right")
    run_stereo_webcam()

if __name__ == "__main__":
    main()