import numpy as np
import math
import cv2
import time
import csv
from geopy.distance import distance as geo_distance
from pymavlink import mavutil
from geopy import Point
from collections import deque
from ultralytics import YOLO

CAMERA_INDEX = 1 # Change this to your camera's index (e.g., 0, 1, 2)
VIDEO_PATH = "output.avi"

# ---- Drone Connection ----
try:
    master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    #master = mavutil.mavlink_connection('/dev/ttyACM0')
    master.wait_heartbeat()
    print("Heartbeat from system (system %u component %u)" % (master.target_system, master.target_component))
except Exception as e:
    print(f"Failed to connect to drone: {e}")
    exit()

# ---- Request Data Streams ----
# Request POSITION data
master.mav.request_data_stream_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_POSITION,
    1, 1  # 1 Hz, enable stream
)
# Request ATTITUDE data
master.mav.request_data_stream_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
    1, 1 # 1 Hz, enable stream
)

# ----------------------- Camera Parameters -----------------------
# NOTE: For best accuracy, perform camera calibration.
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Camera intrinsic matrix (example for Intel RealSense D455, use your own)
K = np.array([[650.90417285, 0, 318.97278063],
              [0, 651.45358764, 236.01686148],
              [0, 0, 1]])

# ----------------------- YOLO Model -----------------------
try:
    yolo_model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit()

# ----------------------- Helper Functions -----------------------
# def get_heading_deg(yaw_rad):
#     """Converts yaw from radians to degrees [0, 360)."""
#     heading = math.degrees(yaw_rad)
#     return (heading + 360) % 360

def get_body_frame_offsets(u, v, pitch_rad, altitude, K):
    """
    Calculates the forward and rightward offsets (in meters) of a pixel
    in the drone's body frame using 3D ray projection.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Convert pixel to normalized camera coordinates
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy

    # Ray in camera frame (Z-forward, X-right, Y-down)
    ray_camera = np.array([x_cam, y_cam, 1.0])

    # Apply pitch rotation (rotation around X-axis)
    R_pitch = np.array([
        [1, 0, 0],
        [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
        [0, math.sin(pitch_rad),  math.cos(pitch_rad)]
    ])
    ray_pitched = R_pitch @ ray_camera

    # Check if ray points towards ground (positive Z component in pitched frame)
    if ray_pitched[2] <= 1e-6: # Use a small epsilon to avoid division by zero
        return None, None

    # Scale ray to project onto the ground plane (Z = altitude)
    t = altitude / ray_pitched[2]
    ground_point = ray_pitched * t

    # Extract body-frame offsets (assuming drone's body frame is Z-down, X-forward, Y-right)
    # And camera is pitched down, so camera's Y is drone's X (forward)
    # and camera's X is drone's Y (right)
    offset_forward = ground_point[1]
    offset_right = ground_point[0]

    return offset_forward, offset_right

def offset_to_latlon(lat, lon, offset_north, offset_east):
    """
    Calculates a new GPS coordinate given a starting point and North/East offsets.
    """
    origin = Point(latitude=lat, longitude=lon)
    distance_m = math.sqrt(offset_north**2 + offset_east**2)
    if distance_m < 1e-6:
        return lat, lon
    bearing_deg = math.degrees(math.atan2(offset_east, offset_north))
    destination = geo_distance(meters=distance_m).destination(origin, bearing=bearing_deg)
    return destination.latitude, destination.longitude

# ----------------------- Video Capture -----------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

if not cap.isOpened():
    print(f"Error: Camera with index {CAMERA_INDEX} not found.")
    exit()

print("Camera opened successfully. Press 'q' to quit.")

# cap = cv2.VideoCapture(VIDEO_PATH)
# if not cap.isOpened():
#     print(f"Error: Cannot open video source.")
#     exit()
# print("Video source opened successfully. Press 'q' to quit.")

# ----------------------- CSV Logging Setup -----------------------
# START of the added code block
LOG_FILE_NAME = "detection_log_anick.csv"
log_file = None
csv_writer = None
try:
    # Open the file in write mode ('w') with no extra newlines
    log_file = open(LOG_FILE_NAME, 'w', newline='')
    csv_writer = csv.writer(log_file)
    # Write the header row for the CSV file
    csv_writer.writerow([
        "Timestamp", "Drone_Lat", "Drone_Lon", "Drone_Alt_m",
        "Drone_Pitch_deg", "Drone_Heading_deg", "Offset_Fwd_m", "Offset_Right_m",
        "Offset_North_m", "Offset_East_m", "Ground_Distance_m",
        "Estimated_Target_Lat", "Estimated_Target_Lon"
    ])
    print(f"Logging detection data to {LOG_FILE_NAME}")
except IOError as e:
    print(f"I/O error({e.errno}): {e.strerror}")
    print(f"Could not open log file: {LOG_FILE_NAME}")
# END of the added code block

# ----------------------- Main Loop -----------------------
frame_count = 0
prev_time = time.time()
detections = []
n_infer = 4       # Run YOLO every N frames to improve performance

# Variables to store the latest drone state
msg_attitude = None
msg_position = None

while True:
    # --- Non-blocking read of MAVLink messages ---
    msg = master.recv_match(blocking=False)
    if msg:
        if msg.get_type() == 'ATTITUDE':
            msg_attitude = msg
        if msg.get_type() == 'GLOBAL_POSITION_INT':
            msg_position = msg

    # --- Read Camera Frame ---
    ret, frame = cap.read()
    if not ret:
        print("End of video stream. Exiting.")
        break

    frame_count += 1
    run_inference = frame_count % n_infer == 0

    # --- Run Inference ---
    if run_inference:
        results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
        detections = results[0].boxes if results and len(results) > 0 else []

    img_center = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)
    cv2.circle(frame, img_center, 5, (255, 0, 0), -1, cv2.LINE_AA)

    # --- Process Detections ---
    if detections and msg_position and msg_attitude:
        box = detections[0] # Process the first detection
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        obj_center = (int(center_x), int(center_y))

        # --- Get Drone State from stored messages ---
        altitude = msg_position.relative_alt / 1000.0  # meters
        pitch_rad = msg_attitude.pitch                  # radians
        # Heading from GLOBAL_POSITION_INT is in centi-degrees (0-35999)
        heading_deg = msg_position.hdg / 100.0
        current_lat = msg_position.lat / 1e7            # degrees
        current_lon = msg_position.lon / 1e7            # degrees

        # altitude = 4.8
        # pitch_rad = 0
        # heading_deg = 162.0
        # current_lat = 28.75053
        # current_lon = 77.11203

        # --- Geometry Calculation ---
        offset_fwd, offset_right = get_body_frame_offsets(center_x, center_y, pitch_rad, altitude, K)
        
        if offset_fwd is not None:
            # --- Coordinate Transformation to World Frame ---
            heading_rad = math.radians(heading_deg)
            offset_north = offset_fwd * math.cos(heading_rad) - offset_right * math.sin(heading_rad)
            offset_east  = offset_fwd * math.sin(heading_rad) + offset_right * math.cos(heading_rad)
            ground_distance = math.sqrt(offset_north**2 + offset_east**2)
        
            # --- Estimate Target GPS (using raw, unfiltered offsets) ---
            est_lat, est_lon = offset_to_latlon(current_lat, current_lon, offset_north, offset_east)
            drone_to_target_dist = math.sqrt(ground_distance**2 + altitude**2)

            # --- Drawing on Frame ---
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.circle(frame, obj_center, 5, (0, 255, 0), -1)
            cv2.line(frame, img_center, obj_center, (0, 0, 255), 2)
            midpoint = ((img_center[0] + obj_center[0]) // 2, (img_center[1] + obj_center[1]) // 2)
            cv2.putText(frame, f"{ground_distance:.2f} m", midpoint,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- Print Information ---
            print("\n--- OBJECT DETECTED ---")
            print(f"Drone State:  Lat={current_lat:.6f}, Lon={current_lon:.6f}, Alt={altitude:.2f}m")
            print(f"Attitude:     Pitch={math.degrees(pitch_rad):.2f}°, Heading={heading_deg:.2f}°")
            print(f"Offsets(Body):Fwd={offset_fwd:.2f}m, Right={offset_right:.2f}m")
            print(f"Offsets(World):N={offset_north:.2f}m, E={offset_east:.2f}m")
            print(f"Ground Dist:  {ground_distance:.2f} m")
            print(f"3D Drone Dist:{drone_to_target_dist:.2f} m")
            print(f"Est. Target:  ({est_lat:.6f}, {est_lon:.6f})")
            print("-------------------------")

                        # --- Log Data to CSV ---
            # START of the added code block
            if csv_writer:
                log_data = [
                    time.time(),              # Current timestamp
                    current_lat,
                    current_lon,
                    altitude,
                    math.degrees(pitch_rad),
                    heading_deg,
                    offset_fwd,
                    offset_right,
                    offset_north,
                    offset_east,
                    ground_distance,
                    est_lat,
                    est_lon
                ]
                csv_writer.writerow(log_data)


    # --- FPS Display ---
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
if log_file:
    log_file.close()
    print(f"Log file '{LOG_FILE_NAME}' closed.")

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
master.close()
print("Script finished.")