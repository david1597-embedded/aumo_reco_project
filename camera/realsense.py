import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import math



class RealSense:
    def __init__(self,filter_size = 3, filter_use = False, cam_param_path = './camera/camera_parameter.npz'):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.filter_size = filter_size
        self.filter_use = filter_use
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.fx, self.fy, self.cx, self.cy, _ ,_ ,_ ,_ ,_  = self.get_intrinsic_camera(cam_param_path)
        
    
    def getframe(self):
         # Wait for frames
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()#프레임
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            pass

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())#프레임에서 데이터 받아서 행렬화
        color_image = np.asanyarray(color_frame.get_data())
        if self.filter_use:             
            depth_vis = cv2.GaussianBlur(depth_image, (self.filter_size, self.filter_size), 0)
        else :
            depth_vis = depth_image
        # Apply colormap to depth image
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_vis, alpha=0.03), cv2.COLORMAP_JET
        )
        return color_image, depth_colormap ,depth_frame

    #measuring depth by pixel
    def measuredistance(self, depth_frame, px,py):
        depth_meter = depth_frame.get_distance(px, py)
        return depth_meter
    
    def measureangle(self,  px, py , distance):
        X = (px - self.cx) * distance / self.fx
        Y = (py - self.cy) * distance / self.fy
        Z = distance

        yaw = math.degrees(math.atan2(X,Z))
        pitch = math.degrees(math.atan2(Y, Z))
        return yaw , pitch

    def get_intrinsic_camera(self,cam_param_path):
        data = np.load(cam_param_path)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']

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
            self.logger.warning(f"큰 방사형 왜곡 계수 감지: k1={k1:.3f}, k2={k2:.3f}")
        if abs(k3) > 5.0:
            self.logger.warning(f"과도한 k3 왜곡 계수: {k3:.3f}")
        
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        print(f"현재 카메라 x축 초점거리 : {fx}")
        print(f"현재 카메라 y축 주점 : {fy}")
        print(f"현재 카메라 x축 초점거리 : {cx}")
        print(f"현재 카메라 y축 주점 : {cy}")

        return fx, fy, cx, cy , k1, k2, k3, p1, p2
    
    def calculate_iou(self,box1, box2):
        # box: [x1, y1, width, height]
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # 교차 영역 좌표
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        # 교차 영역 넓이
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # 합집합 영역 넓이
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        # IoU 계산
        return inter_area / union_area if union_area > 0 else 0



        

        
                    



