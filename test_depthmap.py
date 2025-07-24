from camera.realsense import RealSense
#from motors.motors import MotorController
import numpy as np
import cv2


realsenseCamera=RealSense(filter_size=3, filter_use= False)

color_image, depth_image = realsenseCamera.getframe()

#오리지널 프레임과 dpeth map 프레임 받아오기
stacked_frame= np.hstack((color_image, depth_image))

#모터 컨트롤러 (전진, 후진 , 제자리 좌 우 회전, 정지 기능)
#motorController = MotorController()

#카메라 파라미터 불러오기
#fx , fy, cx, cy = realsenseCamera.get_intrinsic_camera('./camera/camera_parmeter.npz')

try:
    while True:
        color_image, depth_image = realsenseCamera.getframe()

        stacked_frame= np.hstack((color_image, depth_image))

        cv2.imshow('RealSense RGB (left) + Depth (right)', stacked_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

finally:
    # Stop streaming
    realsenseCamera.pipeline.stop()
    cv2.destroyAllWindows()


