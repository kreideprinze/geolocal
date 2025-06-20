import cv2
import numpy as np
import time
import csv
import os
import shutil
from cv2 import aruco
from pymavlink import mavutil
from threading import Thread
from collections import defaultdict

# === Constants and Parameters ===
CAMERA_MATRIX = np.array([[448.44, 0, 302.37], [0, 450.06, 244.72], [0, 0, 1]])
DIST_COEFFS = np.array([0.2697, -1.5636, -0.0095, -0.0080, 3.5658])
MARKER_SIZE = 0.5
OUT_DIR = "marker_images"
CSV_FILE = "global_coordinate.csv"
VIDEO_FILE = "output.avi"

# === Load Camera to Drone Transform ===
def load_cam_to_drone_transform(path="cam_to_drone_transform.csv"):
    try:
        with open(path, "r") as f:
            lines = list(csv.reader(f))
        R = np.array([list(map(float, lines[1])),
                      list(map(float, lines[2])),
                      list(map(float, lines[3]))])
        T = np.array(list(map(float, lines[5])))
        return R, T
    except Exception as e:
        raise RuntimeError(f"Error loading cam_to_drone_transform: {e}")

R_cam2drone, T_cam2drone = load_cam_to_drone_transform()

# === MAVLink Setup ===
def connect_to_sitl():
    try:
        master = mavutil.mavlink_connection(input("Enter connection string (e.g. udp:127.0.0.1:14550): "))
        master.wait_heartbeat(timeout=10)
        print("✔ Connected to vehicle")
        return master
    except Exception as e:
        raise ConnectionError(f"MAVLink connection failed: {e}")

def mav_listener(master, state):
    try:
        while True:
            msg = master.recv_match(blocking=True)
            if not msg:
                continue
            t = msg.get_type()
            if t == "GPS_RAW_INT":
                state["lat"] = msg.lat / 1e7
                state["lon"] = msg.lon / 1e7
                state["alt"] = msg.alt / 1000.0
            elif t == "ATTITUDE":
                state["roll"] = msg.roll
                state["pitch"] = msg.pitch
                state["yaw"] = msg.yaw
            elif t == "VFR_HUD":
                state["baro_alt"] = msg.alt
    except Exception as e:
        print(f"❌ MAV listener error: {e}")

# === Coordinate Transform ===
def cam_to_world(tvec_cam, roll, pitch, yaw):
    try:
        tvec_drone = R_cam2drone @ tvec_cam + T_cam2drone
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx @ tvec_drone
    except Exception as e:
        print(f"❌ Transform error: {e}")
        return np.zeros(3)

# === Main Execution ===
def main():
    cap = cv2.VideoCapture("/dev/video4")
    if not cap.isOpened():
        raise IOError("❌ Camera not accessible")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(VIDEO_FILE, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (frame_width, frame_height))

    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

    master = connect_to_sitl()
    state = defaultdict(lambda: None)
    Thread(target=mav_listener, args=(master, state), daemon=True).start()

    marker_data = defaultdict(list)
    t0 = time.time()
    fps = 0
    f_counter = 0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            lat, lon, alt = state["lat"], state["lon"], state["alt"]
            roll, pitch, yaw = state["roll"], state["pitch"], state["yaw"]
            baro_alt = state["baro_alt"]

            if ids is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, CAMERA_MATRIX, DIST_COEFFS)
                aruco.drawDetectedMarkers(frame, corners, ids)

                for i, marker_id in enumerate(ids.flatten()):
                    t_world = cam_to_world(tvecs[i][0], roll, pitch, yaw)
                    dist = np.linalg.norm(t_world)
                    R_earth = 6371000.0
                    dlat = (t_world[1] / R_earth) * 180 / np.pi
                    dlon = (t_world[0] / (R_earth * np.cos(np.radians(lat)))) * 180 / np.pi
                    m_lat, m_lon = lat + dlat, lon + dlon
                    m_alt = (baro_alt or alt) - t_world[2]
                    marker_data[marker_id].append((m_lat, m_lon, m_alt, dist, frame_idx, frame.copy()))
                    cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvecs[i], tvecs[i], 0.1)

            # Display telemetry and orientation
            def draw_text(label, val, y, color):
                val_text = f"{np.degrees(val):.1f}°" if val is not None else "—"
                cv2.putText(frame, f"{label}: {val_text}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            gps_text = f"GPS: {lat:.6f}, {lon:.6f}, Alt: {alt:.2f}m" if lat and lon and alt else "NO GPS LOCK FOUND"
            cv2.putText(frame, gps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255 if lat else 0, 0 if lat else 255), 2)
            draw_text("Roll", roll, 60, (255, 100, 100))
            draw_text("Pitch", pitch, 90, (100, 255, 100))
            draw_text("Yaw", yaw, 120, (100, 100, 255))

            f_counter += 1
            if time.time() - t0 >= 1:
                fps = f_counter / (time.time() - t0)
                t0 = time.time()
                f_counter = 0

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            video_writer.write(frame)
            cv2.imshow("ArUco Detection", frame)

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Marker ID", "Latitude", "Longitude", "Altitude", "Dist(m)", "Frame"])
            for m_id, detections in marker_data.items():
                close = min(detections, key=lambda d: d[3])
                writer.writerow([m_id, close[0], close[1], close[2], close[3], close[4]])
                img_path = os.path.join(OUT_DIR, f"marker_{m_id}_frame_{close[4]}.png")
                cv2.imwrite(img_path, close[5])
                print(f"✔ Saved snapshot → {img_path}")


if __name__ == "__main__":
    main()

