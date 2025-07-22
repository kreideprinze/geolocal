#!/usr/bin/env python3
import sys
import subprocess
import importlib
import time
import csv
import os
import shutil
from threading import Thread
from collections import defaultdict

# Dependency management
def ensure_package(module_name, pip_name=None):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pkg = pip_name or module_name
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(module_name)

np = ensure_package("numpy")
cv2 = ensure_package("cv2", "opencv-python")
aruco = cv2.aruco
mavutil = ensure_package("pymavlink.mavutil")

import numpy as np
from cv2 import aruco

# Camera calibration parameters
camera_matrix = np.array([[448.44050858, 0, 302.36894562],
                          [0, 450.05835973, 244.72255502],
                          [0, 0, 1]])
dist_coeffs = np.array([0.26974184, -1.56360967, -0.00950144,
                        -0.00800682, 3.5658071])
marker_size = 0.5

# Load transform from camera frame to drone frame
def load_cam_to_drone_transform(path="cam_to_drone_transform.csv"):
    with open(path, "r") as f:
        lines = list(csv.reader(f))
    R = np.array([list(map(float, lines[1])),
                  list(map(float, lines[2])),
                  list(map(float, lines[3]))])
    T = np.array(list(map(float, lines[5])))
    return R, T

# Ensure output directory
def ensure_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

# Connect to SITL
def connect_to_sitl():
    master = mavutil.mavlink_connection(
        input("Enter connection string (e.g. udp:127.0.0.1:14550) : ")
    )
    master.wait_heartbeat(timeout=10)
    return master

# Load camera-to-drone transform
R_cam2drone, T_cam2drone = load_cam_to_drone_transform()

# ArUco marker detection setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# State tracking
drone_state = {
    'lat': None, 'lon': None, 'alt': None,
    'roll': None, 'pitch': None, 'yaw': None,
    'baro_alt': None
}
marker_data = {}

# MAVLink listener
def mav_listener(master):
    while True:
        msg = master.recv_match(blocking=True)
        if not msg:
            continue
        t = msg.get_type()
        if t == 'GPS_RAW_INT':
            drone_state['lat'] = msg.lat / 1e7
            drone_state['lon'] = msg.lon / 1e7
            drone_state['alt'] = msg.alt / 1000.0
        elif t == 'ATTITUDE':
            drone_state['roll'], drone_state['pitch'], drone_state['yaw'] = msg.roll, msg.pitch, msg.yaw
        elif t == 'VFR_HUD':
            drone_state['baro_alt'] = msg.alt

# Project camera ray to 3D position
def compute_marker_position(corner, altitude):
    center_px = np.mean(corner[0], axis=0)  # (u, v)
    uv_hom = np.array([center_px[0], center_px[1], 1.0])
    undistorted = np.linalg.inv(camera_matrix) @ uv_hom
    ray_cam = undistorted / np.linalg.norm(undistorted)
    position_cam = ray_cam * altitude
    position_drone = R_cam2drone @ position_cam + T_cam2drone
    return position_drone, center_px

# Transform camera-based world position to global coordinates
def cam_to_global(pos_cam, roll, pitch, yaw, lat, lon, ref_alt):
    Rx = np.array([[1,0,0], [0,np.cos(roll),-np.sin(roll)], [0,np.sin(roll),np.cos(roll)]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)], [0,1,0], [-np.sin(pitch),0,np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0], [np.sin(yaw),np.cos(yaw),0], [0,0,1]])
    R = Rz @ Ry @ Rx
    pos_world = R @ pos_cam
    dist = np.linalg.norm(pos_world)
    R_earth = 6_371_000.0
    dlat = (pos_world[1] / R_earth) * 180/np.pi
    dlon = (pos_world[0] / (R_earth*np.cos(lat*np.pi/180))) * 180/np.pi
    return lat + dlat, lon + dlon, ref_alt - pos_world[2], dist, dlat, dlon

# Main function
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit(1)

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc       = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

    ensure_dir("marker_images")
    master = connect_to_sitl()
    Thread(target=mav_listener, args=(master,), daemon=True).start()

    t0, f_counter, fps, frame_idx = time.time(), 0, 0, 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            lat, lon, alt = drone_state['lat'], drone_state['lon'], drone_state['alt']
            roll, pitch, yaw = drone_state['roll'], drone_state['pitch'], drone_state['yaw']
            baro_alt = drone_state['baro_alt']

            if ids is not None and None not in (lat, lon, alt, roll, pitch, yaw):
                ref_alt = baro_alt if baro_alt is not None else alt
                for i in range(len(ids)):
                    aruco.drawDetectedMarkers(frame, corners, ids)

                    position_drone, center_px = compute_marker_position(corners[i], ref_alt)
                    m_lat, m_lon, m_alt, dist, dlat, dlon = cam_to_global(position_drone, roll, pitch, yaw, lat, lon, ref_alt)
                    m_id = ids[i][0]

                    if m_id not in marker_data or dist < marker_data[m_id][3]:
                        marker_data[m_id] = (m_lat, m_lon, m_alt, dist, frame_idx, frame.copy())
                        img_name = f"marker_images/marker_{m_id}_frame_{frame_idx}.png"
                        cv2.imwrite(img_name, frame)

                        with open("global_coordinate.csv", "w", newline="") as f:
                            w = csv.writer(f)
                            w.writerow(["Marker ID", "Latitude", "Longitude", "Altitude", "Dist(m)", "Frame"])
                            for mid, data in marker_data.items():
                                w.writerow([mid, data[0], data[1], data[2], data[3], data[4]])

                    # Draw line from center of image to marker
                    image_center = (frame.shape[1]//2, frame.shape[0]//2)
                    marker_center = tuple(map(int, center_px))
                    cv2.line(frame, image_center, marker_center, (255, 0, 0), 2)
                    coord_text = f"ΔLat: {dlat:.6f}, ΔLon: {dlon:.6f}"
                    cv2.putText(frame, coord_text, (marker_center[0]+10, marker_center[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Display telemetry
            if None not in (lat, lon, alt):
                gps_text = f"GPS: {lat:.6f}, {lon:.6f}, Alt: {alt:.2f}m"
                cv2.putText(frame, gps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(frame, "NO GPS LOCK FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            def d2str(a): return f"{np.degrees(a):.1f}°" if a is not None else "—"
            cv2.putText(frame, f"Roll:  {d2str(roll)}",  (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,100), 2)
            cv2.putText(frame, f"Pitch: {d2str(pitch)}", (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
            cv2.putText(frame, f"Yaw:   {d2str(yaw)}",   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,255), 2)

            f_counter += 1
            if time.time()-t0 >= 1.0:
                fps = f_counter/(time.time()-t0)
                t0, f_counter = time.time(), 0
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            video_writer.write(frame)
            cv2.imshow("ArUco Detection", frame)
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
