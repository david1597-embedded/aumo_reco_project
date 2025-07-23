import time
from pathlib import Path
import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
from notebook_utils import download_file, device_widget
import openvino.properties as props


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


# Model setup
device = device_widget()
model_folder = Path("./midas_depth/model")

ir_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/"
ir_model_name_xml = "MiDaS_small.xml"
ir_model_name_bin = "MiDaS_small.bin"

# # Download model files
# download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=model_folder)
# download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=model_folder)

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

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # 0은 기본 웹캠을 의미

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set webcam properties (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()
fps_update_interval = 30  # Update FPS every 30 frames

print("Starting real-time depth estimation...")
print("Press 'q' to quit, 's' to save current frame")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Prepare frame for inference
        # Resize to the input shape for network
        resized_image = cv2.resize(src=frame, dsize=(network_image_width, network_image_height))
        # Reshape the image to network input shape NCHW
        input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

        # Perform inference
        inference_start = time.perf_counter()
        result = compiled_model([input_image])[output_key]
        inference_end = time.perf_counter()
        inference_time = inference_end - inference_start

        # Convert result to colored depth map
        result_frame = convert_result_to_image(result)
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)

        # Resize both frames to same size for display
        display_height, display_width = frame.shape[:2]
        result_frame_resized = cv2.resize(result_frame, (display_width, display_height))

        # Create side-by-side display
        combined_frame = np.hstack((frame, result_frame_resized))

        # Calculate and display FPS
        frame_count += 1
        if frame_count % fps_update_interval == 0:
            elapsed_time = time.time() - start_time
            fps = fps_update_interval / elapsed_time
            start_time = time.time()
            print(f"FPS: {fps:.1f}, Inference time: {inference_time*1000:.1f}ms")

        # Add text overlay
        cv2.putText(combined_frame, f'Inference: {inference_time*1000:.1f}ms', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_frame, 'Original', (10, display_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_frame, 'Depth Map', (display_width + 10, display_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('Real-time Depth Estimation (Press q to quit, s to save)', combined_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = int(time.time())
            filename = f'depth_estimation_{timestamp}.jpg'
            cv2.imwrite(filename, combined_frame)
            print(f"Frame saved as {filename}")

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed")