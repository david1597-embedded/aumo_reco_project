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
    Convert network result of floating point numbers to an RGB image with
    integer values from 0-255 by applying a colormap.

    `result` is expected to be a single network result in 1,H,W shape
    `colormap` is a matplotlib colormap.
    See https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    cmap = matplotlib.cm.get_cmap(colormap)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    result = result.astype(np.uint8)
    return result

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
model_folder = Path("./midas_depth/model")

ir_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/"
ir_model_name_xml = "MiDaS_small.xml"
ir_model_name_bin = "MiDaS_small.bin"

model_xml_path = model_folder / ir_model_name_xml

# Create cache folder
cache_folder = Path("./midas_depth/cache")
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
print("\n=== DEPTH MAP 정보 ===")
print("• MiDaS는 'inverse depth' 또는 'disparity'를 출력합니다")
print("• 값이 클수록 = 카메라에 가까운 객체")
print("• 값이 작을수록 = 카메라에서 먼 객체")
print("• 절대적인 거리(미터) 값은 아니고 상대적인 깊이 정보입니다")
print("• 정규화 후: 0 = 가장 먼 지점, 1 = 가장 가까운 지점")
print("=" * 50)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set webcam properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()
fps_update_interval = 30

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

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Store frame dimensions for mouse callback
        display_height, display_width = frame.shape[:2]
        frame_width = display_width

        # Prepare frame for inference
        resized_image = cv2.resize(src=frame, dsize=(network_image_width, network_image_height))
        input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

        # Perform inference
        inference_start = time.perf_counter()
        result = compiled_model([input_image])[output_key]
        inference_end = time.perf_counter()
        inference_time = inference_end - inference_start

        # Store raw depth map for mouse callback
        raw_depth_map = result.squeeze(0)
        raw_depth_map = cv2.resize(raw_depth_map, (display_width, display_height))

        # Convert result to colored depth map
        result_frame = convert_result_to_image(result)
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
        result_frame_resized = cv2.resize(result_frame, (display_width, display_height))

        # Create side-by-side display
        combined_frame = np.hstack((frame, result_frame_resized))
        display_frame = combined_frame.copy()

        # Calculate and display FPS
        frame_count += 1
        if frame_count % fps_update_interval == 0:
            elapsed_time = time.time() - start_time
            fps = fps_update_interval / elapsed_time
            start_time = time.time()
            print(f"FPS: {fps:.1f}, Inference time: {inference_time*1000:.1f}ms")

        # Add basic text overlays
        cv2.putText(combined_frame, f'Inference: {inference_time*1000:.1f}ms', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_frame, 'Original', (10, display_height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_frame, 'Depth Map (Click for values)', (display_width + 10, display_height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add depth map statistics
        depth_min = raw_depth_map.min()
        depth_max = raw_depth_map.max()
        depth_mean = raw_depth_map.mean()
        
        cv2.putText(combined_frame, f'Min: {depth_min:.2f}', (display_width + 10, display_height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(combined_frame, f'Max: {depth_max:.2f}', (display_width + 10, display_height - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(combined_frame, f'Mean: {depth_mean:.2f}', (display_width + 10, display_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw clicked point and value if available
        if clicked_point is not None:
            px, py, depth_val = clicked_point
            # Draw crosshair on depth map
            cv2.drawMarker(combined_frame, (px + display_width, py), (0, 0, 255), 
                          cv2.MARKER_CROSS, 20, 3)
            
            # Display depth value
            normalized_val = normalize_minmax(raw_depth_map)[py, px]
            text = f'({px},{py}): {depth_val:.3f}'
            text2 = f'Norm: {normalized_val:.3f}'
            
            # Background rectangle for text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(combined_frame, (px + display_width - 10, py - 40), 
                         (px + display_width + text_size[0] + 10, py - 5), (0, 0, 0), -1)
            
            cv2.putText(combined_frame, text, (px + display_width, py - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(combined_frame, text2, (px + display_width, py - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Add color bar legend
        legend_height = 200
        legend_width = 20
        legend_x = combined_frame.shape[1] - 30
        legend_y = 50
        
        # Create color bar
        for i in range(legend_height):
            color_val = 1.0 - (i / legend_height)  # Invert for proper depth representation
            color = plt.cm.viridis(color_val)[:3]
            color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
            cv2.rectangle(combined_frame, (legend_x, legend_y + i), 
                         (legend_x + legend_width, legend_y + i + 1), color_bgr, -1)
        
        # Add legend labels
        cv2.putText(combined_frame, 'Near', (legend_x - 50, legend_y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(combined_frame, 'Far', (legend_x - 40, legend_y + legend_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow(window_name, combined_frame)

        # Handle key presses
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