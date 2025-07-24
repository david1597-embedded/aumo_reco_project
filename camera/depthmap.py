import time
from pathlib import Path
import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
from notebook_utils import download_file, device_widget
import openvino.properties as props

# Global variables for mouse callback
clicked_point = None
raw_depth_map = None
display_frame = None
frame_width = 0

def normalize_minmax(data):
    """Normalizes the values in `data` between 0 and 1"""
    return (data - data.min()) / (data.max() - data.min())

def convert_result_to_image(result, colormap="viridis"):
    """
    Convert a 2D depth map to a color image using a colormap.
    `result` is expected to be 2D (H, W).
    """
    cmap = matplotlib.colormaps.get_cmap(colormap)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    return result.astype(np.uint8)

def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function to handle clicks on the depth map"""
    global clicked_point, raw_depth_map, display_frame, frame_width
    
    if event == cv2.EVENT_LBUTTONDOWN and raw_depth_map is not None:
        # Check if click is on the depth map (right side of combined frame)
        if x > frame_width:
            # Convert click coordinates to depth map coordinates
            depth_x = x - frame_width
            depth_y = y
            
            # Get depth value at clicked position
            if 0 <= depth_y < raw_depth_map.shape[0] and 0 <= depth_x < raw_depth_map.shape[1]:
                depth_value = raw_depth_map[depth_y, depth_x]
                clicked_point = (depth_x, depth_y, depth_value)
                print(f"Clicked at pixel ({depth_x}, {depth_y})")
                print(f"Raw depth value: {depth_value:.3f}")
                print(f"Relative depth (0=far, 1=near): {normalize_minmax(raw_depth_map)[depth_y, depth_x]:.3f}")
                print("-" * 50)

# Model setup
device = device_widget()
model_folder = Path("./camera/model")

ir_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/"
ir_model_name_xml = "MiDaS_small.xml"
ir_model_name_bin = "MiDaS_small.bin"

model_xml_path = model_folder / ir_model_name_xml

# Create cache folder
cache_folder = Path("./camera/cache")
cache_folder.mkdir(exist_ok=True)

# Load and compile model
core = ov.Core()
core.set_property({props.cache_dir(): cache_folder})
model = core.read_model(model_xml_path)
compiled_model = core.compile_model(model=model, device_name=device.value)

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)

network_input_shape = list(input_key.shape)
network_image_height, network_image_width = network_input_shape[2:]

print(f"Model input shape: {network_input_shape}")
print(f"Network image dimensions: {network_image_width}x{network_image_height}")
print("=== DEPTH MAP 정보 ===")
print("• MiDaS는 'inverse depth' 또는 'disparity'를 출력합니다")
print("• 값이 클수록 = 카메라에 가까운 객체")
print("• 값이 작을수록 = 카메라에서 먼 객체")
print("• 절대적인 거리(미터) 값은 아니고 상대적인 깊이 정보입니다")
print("• 정규화 후: 0 = 가장 먼 지점, 1 = 가장 가까운 지점")
print("=" * 50)

# Initialize webcam
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set webcam properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

# Initialize variables for FPS calculation and inference timing
frame_count = 0
start_time = time.time()
fps_update_interval = 30
last_inference_time = 0
inference_interval = 0.5  # Perform inference every 0.5 seconds

# Create window and set mouse callback
window_name = 'Depth Estimation - Click on depth map for values'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

print("Starting real-time depth estimation...")
print("Instructions:")
print("• Press 'q' to quit")
print("• Press 's' to save current frame")
print("• Click on the DEPTH MAP (right side) to see pixel depth values")
print("• Press 'r' to reset clicked point")
alpha = 0.3  # EMA 계수 (0.0: 완전 과거, 1.0: 완전 현재)
smoothed_depth_map = None  # 누적된 깊이 맵

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        display_height, display_width = frame.shape[:2]
        frame_width = display_width

        # Check if it's time to perform inference (every 0.5 seconds)
        current_time = time.perf_counter()
        if current_time - last_inference_time >= inference_interval:
            # MiDaS 입력 전처리
            resized_image = cv2.resize(frame, (network_image_width, network_image_height))
            input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

            # 추론
            inference_start = time.perf_counter()
            result = compiled_model([input_image])[output_key]
            inference_end = time.perf_counter()
            inference_time = inference_end - inference_start
            last_inference_time = current_time

            # 추정된 깊이 맵 후처리
            current_depth_map = result.squeeze(0)
            current_depth_map = cv2.resize(current_depth_map, (display_width, display_height))

            # 지수 평균 필터링 (EMA)
            if smoothed_depth_map is None:
                smoothed_depth_map = current_depth_map
            else:
                smoothed_depth_map = alpha * current_depth_map + (1 - alpha) * smoothed_depth_map

            # 시각화용 컬러 이미지 생성
            raw_depth_map = smoothed_depth_map.copy()
            result_frame = convert_result_to_image(smoothed_depth_map)
            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
            result_frame_resized = cv2.resize(result_frame, (display_width, display_height))

        # Use the last computed depth map if inference was skipped
        combined_frame = np.hstack((frame, result_frame_resized))
        display_frame = combined_frame.copy()

        # Draw clicked point and depth values on the frame
        if clicked_point is not None:
            depth_x, depth_y, depth_value = clicked_point
            # Draw a circle at the clicked point on the depth map
            cv2.circle(combined_frame, (depth_x + frame_width, depth_y), 5, (0, 255, 0), -1)
            # Draw depth value text near the clicked point
            text = f"Depth: {depth_value:.3f} (Norm: {normalize_minmax(raw_depth_map)[depth_y, depth_x]:.3f})"
            cv2.putText(combined_frame, text, (depth_x + frame_width + 10, depth_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS 계산
        frame_count += 1
        if frame_count % fps_update_interval == 0:
            elapsed_time = time.time() - start_time
            fps = fps_update_interval / elapsed_time
            start_time = time.time()
            print(f"FPS: {fps:.1f}, Inference time: {inference_time*1000:.1f}ms")

        # Show window
        cv2.imshow(window_name, combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            filename = f'depth_estimation_{timestamp}.jpg'
            cv2.imwrite(filename, combined_frame)
            print(f"Frame saved as {filename}")
        elif key == ord('r'):
            clicked_point = None
            print("Clicked point reset")

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed")