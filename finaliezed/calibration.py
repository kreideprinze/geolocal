import cv2
import numpy as np
import csv
from pymavlink import mavutil
import time

# ==== Camera intrinsics (update if needed) ====
camera_matrix = np.array([[448.44050858, 0, 302.36894562],
                          [0, 450.05835973, 244.72255502],
                          [0, 0, 1]])
dist_coeffs = np.array([0.26974184, -1.56360967, -0.00950144,
                        -0.00800682, 3.5658071])

# ==== Chessboard settings ====
chessboard_size = (8,5)
square_size = 0.025  # meters

# ==== Averaging Parameters ====
MAX_FRAMES = 50
r_matrices = []
t_vectors = []

# ==== MAVLink connection ====
master = mavutil.mavlink_connection("/dev/ttyACM0")
master.wait_heartbeat()
print("✅ Connected to Pixhawk")

def get_attitude():
    msg = master.recv_match(type='ATTITUDE', blocking=True, timeout=0.05)
    if msg:
        return msg.roll, msg.pitch, msg.yaw
    return 0.0, 0.0, 0.0

# ==== Chessboard 3D reference points ====
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# ==== OpenCV Video Capture (replace device index as needed) ====
cap = cv2.VideoCapture('/dev/video4')
if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

print("▶ Webcam stream started.")
print("▶ Automatically capturing frames with visible chessboard...\n")

# Time tracking for stability
last_capture_time = 0
min_capture_interval = 1.0  # seconds

try:
    while len(r_matrices) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_cb, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        display = frame.copy()
        msg = "Detecting chessboard..."

        if ret_cb:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            _, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
            R_cam2board, _ = cv2.Rodrigues(rvec)

            roll, pitch, yaw = get_attitude()
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(roll), -np.sin(roll)],
                           [0, np.sin(roll),  np.cos(roll)]])
            Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                           [0, 1, 0],
                           [-np.sin(pitch), 0, np.cos(pitch)]])
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw),  np.cos(yaw), 0],
                           [0, 0, 1]])
            R_drone2world = Rz @ Ry @ Rx

            R_cam2drone = R_cam2board @ R_drone2world.T
            T_cam2drone = tvec.flatten()

            current_time = time.time()
            if current_time - last_capture_time >= min_capture_interval:
                r_matrices.append(R_cam2drone)
                t_vectors.append(T_cam2drone)
                last_capture_time = current_time
                print(f"✅ Auto-captured frame [{len(r_matrices)}/{MAX_FRAMES}]")

            msg = "Chessboard detected ✅"

        cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if ret_cb else (0, 0, 255), 2)
        cv2.putText(display, f"Captured: {len(r_matrices)}/{MAX_FRAMES}", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Auto Chessboard Capture", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

# ==== Post-processing ====
if len(r_matrices) == 0:
    print("⚠️ No frames captured. Exiting.")
    exit()

R_avg = np.mean(np.array(r_matrices), axis=0)
T_avg = np.mean(np.array(t_vectors), axis=0)

# ==== Save to CSV ====
with open("cam_to_drone_transform.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Rotation Matrix (row-wise)"])
    for row in R_avg:
        writer.writerow(row)
    writer.writerow(["Translation Vector"])
    writer.writerow(T_avg)

print("\n✅ Calibration saved to 'cam_to_drone_transform.csv'")
