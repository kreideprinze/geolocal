import cv2
import numpy as np
import csv
import threading
import time
import signal
import sys
from pymavlink import mavutil
from ultralytics import YOLO
from math import sin, cos, radians, sqrt, atan2, degrees
import os

# Load YOLO model
model = YOLO("yolov5su.pt")

# Camera intrinsics (D455)
intrinsic_matrix = np.array([[650.90417285, 0, 318.97278063],
                             [0, 651.45358764, 236.01686148],
                             [0, 0, 1]])
image_center = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]])
focal_len_px = (intrinsic_matrix[0, 0] + intrinsic_matrix[1, 1]) / 2

# Telemetry
telemetry_lock = threading.Lock()
current_lat = None
current_lon = None
current_alt = None
current_yaw_deg = None
current_roll = 0
current_pitch = 0
base_alt = None

# Graceful shutdown flag
running = True

# Track detections
tracked_ids = {}

# Weighted result storage
weighted_results = {}

# Video writer
out_writer = None

def telemetry_thread():
    global current_lat, current_lon, current_alt, current_yaw_deg, base_alt, current_roll, current_pitch
    master = mavutil.mavlink_connection("udp:127.0.0.1:14550") # "udp:127.0.0.1:14550"
    master.wait_heartbeat()
    print("[INFO] MAVLink connected.")

    while running:
        msg = master.recv_match(type=['GLOBAL_POSITION_INT', 'ATTITUDE'], blocking=True, timeout=1)
        if msg:
            telemetry_lock.acquire()
            if msg.get_type() == 'GLOBAL_POSITION_INT':
                current_lat = msg.lat / 1e7
                current_lon = msg.lon / 1e7
                alt_m = msg.relative_alt / 1000.0
                if base_alt is None:
                    base_alt = alt_m
                    print(f"[INFO] Base altitude locked at {base_alt:.2f} m")
                if base_alt is not None:
                    current_alt = alt_m - base_alt
            elif msg.get_type() == 'ATTITUDE':
                current_yaw_deg = (np.degrees(msg.yaw)) % 360
                current_roll = np.degrees(msg.roll)
                current_pitch = np.degrees(msg.pitch)
            telemetry_lock.release()

def offset_gps(lat, lon, distance_m, heading_deg):
    R = 6378137
    heading_rad = radians(heading_deg)
    dlat = (distance_m * cos(heading_rad)) / R
    dlon = (distance_m * sin(heading_rad)) / (R * cos(radians(lat)))
    new_lat = lat + dlat * (180 / np.pi)
    new_lon = lon + dlon * (180 / np.pi)
    return new_lat, new_lon

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# CSV logging
csv_file = open('detections.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Frame", "ID", "Altitude_m", "GroundDist_m",
    "Person_Lat", "Person_Lon", "Drone_Lat", "Drone_Lon"
])
csv_lock = threading.Lock()

# Weighted result output
weighted_csv_file = open('weighted_detections.csv', 'w', newline='')
weighted_csv_writer = csv.writer(weighted_csv_file)
weighted_csv_writer.writerow(["ID", "Avg_Lat", "Avg_Lon", "Min_Weighted_Dist"])

# Open video
cap = cv2.VideoCapture(0) #"/dev/video6"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out_writer = cv2.VideoWriter('output_tracked.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

frame_num = 0

# Signal handler
def signal_handler(sig, frame):
    global running
    print("\n[INFO] Ctrl+C detected. Saving and exiting...")
    running = False
signal.signal(signal.SIGINT, signal_handler)

# Start telemetry thread
threading.Thread(target=telemetry_thread, daemon=True).start()

print("[INFO] Waiting for telemetry...")
while True:
    telemetry_lock.acquire()
    lat_ready = current_lat is not None
    lon_ready = current_lon is not None
    alt_ready = current_alt is not None
    yaw_ready = current_yaw_deg is not None
    telemetry_lock.release()

    if lat_ready and lon_ready and alt_ready and yaw_ready:
        print("[INFO] Telemetry initialized. Starting video processing.")
        break
    else:
        missing = []
        if not lat_ready: missing.append("lat")
        if not lon_ready: missing.append("lon")
        if not alt_ready: missing.append("alt")
        if not yaw_ready: missing.append("yaw")
        print(f"[INFO] Waiting for: {', '.join(missing)}")
        time.sleep(1)

while running and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    telemetry_lock.acquire()
    lat = current_lat
    lon = current_lon
    alt = current_alt
    yaw_deg = current_yaw_deg
    roll = current_roll
    pitch = current_pitch
    telemetry_lock.release()

    if None in (lat, lon, alt, yaw_deg) or alt < 0:
        continue

    plane2plane_dist_m = alt
    plane2plane_dist_cm = plane2plane_dist_m * 100

    results = model.predict(source=frame, classes=[0], conf=0.5, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            box_center = np.array([cx, cy])

            pix_d_px = np.linalg.norm(box_center - image_center)
            ground_dist_cm = plane2plane_dist_m * (pix_d_px / focal_len_px)
            ground_dist_m = ground_dist_cm / 100.0

            person_lat, person_lon = offset_gps(lat, lon, ground_dist_m, yaw_deg)
            id_str = f"{person_lat:.6f}_{person_lon:.6f}"

            weight = 1.0 / (1.0 + abs(roll) + abs(pitch) + ground_dist_m)
            if id_str not in weighted_results:
                weighted_results[id_str] = []
            weighted_results[id_str].append((person_lat, person_lon, weight, ground_dist_m))

            tracked_ids[id_str] = (person_lat, person_lon)

            # Heading breakdown
            delta_north = ground_dist_m * cos(radians(yaw_deg))
            delta_east = ground_dist_m * sin(radians(yaw_deg))
            breakdown_label = f"{ground_dist_m:.2f}m | N:{delta_north:.2f} E:{delta_east:.2f}"

            label = f"{id_str}\nLat:{person_lat:.6f}, Lon:{person_lon:.6f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(image_center[0]), int(image_center[1])), 5, (255, 0, 0), -1)
            cv2.line(frame, (int(cx), int(cy)), (int(image_center[0]), int(image_center[1])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 50), 2)
            cv2.putText(frame, breakdown_label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 2)

            csv_lock.acquire() 
            csv_writer.writerow([
                frame_num,
                 id_str,
                 round(alt, 2),
                 round(ground_dist_m, 2),
                 person_lat,
                 person_lon,
                 lat,
                 lon
             ])

            
            csv_lock.release()

    cv2.imshow("Detection Feed", frame)
    out_writer.write(frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out_writer.release()
cv2.destroyAllWindows()
csv_file.close()

for pid, records in weighted_results.items():
    total_weight = sum(w for _, _, w, _ in records)
    avg_lat = sum(lat * w for lat, _, w, _ in records) / total_weight
    avg_lon = sum(lon * w for _, lon, w, _ in records) / total_weight
    min_dist = min(d for _, _, _, d in records)
    weighted_csv_writer.writerow([pid, avg_lat, avg_lon, min_dist])

weighted_csv_file.close()

print("[INFO] All files saved and resources released.")
