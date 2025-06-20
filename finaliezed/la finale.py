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

# ==== Camera intrinsics ====
camera_matrix = np.array([[448.44050858, 0, 302.36894562],
                          [0, 450.05835973, 244.72255502],
                          [0, 0, 1]])
dist_coeffs = np.array([0.26974184, -1.56360967, -0.00950144,
                        -0.00800682, 3.5658071])
marker_size = 0.5  # metres

# ==== Load Camera‑to‑Drone calibration ====
def load_cam_to_drone_transform(path="finaliezed/cam_to_drone_transform.csv"):
    with open(path, "r") as f:
        lines = list(csv.reader(f))
    R = np.array([list(map(float, lines[1])),
                  list(map(float, lines[2])),
                  list(map(float, lines[3]))])
    T = np.array(list(map(float, lines[5])))
    return R, T

R_cam2drone, T_cam2drone = load_cam_to_drone_transform()

# ==== ArUco ====
aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()
detector     = aruco.ArucoDetector(aruco_dict, aruco_params)

# ==== Shared state ====
state = {
    'lat': None, 'lon': None, 'alt': None,
    'roll': None, 'pitch': None, 'yaw': None,
    'baro_alt': None
}
marker_data = defaultdict(list)

# ==== MAVLink ====
def connect_to_sitl():
    master = mavutil.mavlink_connection(input("Enter connection string (e.g. udp:127.0.0.1:14550) : "))
    master.wait_heartbeat(timeout=10)
    print("✔ Connected to vehicle")
    return master

def mav_listener(master):
    while True:
        msg = master.recv_match(blocking=True)
        if not msg:
            continue
        t = msg.get_type()
        if t == 'GPS_RAW_INT':
            state['lat'] = msg.lat / 1e7
            state['lon'] = msg.lon / 1e7
            state['alt'] = msg.alt / 1000.0
        elif t == 'ATTITUDE':
            state['roll']  = msg.roll
            state['pitch'] = msg.pitch
            state['yaw']   = msg.yaw
        elif t == 'VFR_HUD':
            state['baro_alt'] = msg.alt

# ==== Coordinate transform ====
def cam_to_world(tvec_cam, roll, pitch, yaw):
    tvec_drone = R_cam2drone @ tvec_cam + T_cam2drone
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    return (Rz @ Ry @ Rx) @ tvec_drone

# ==== Video ====
cap = cv2.VideoCapture('/dev/video2')
print("▶ VideoCapture started")

# 📹 ==== VideoWriter ====
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc       = cv2.VideoWriter_fourcc(*'XVID')  # or 'mp4v' for .mp4
video_writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

# ==== Output folder ====
out_dir = "marker_images"
shutil.rmtree(out_dir, ignore_errors=True)
os.makedirs(out_dir, exist_ok=True)

# ==== Start MAVLink ====
master = connect_to_sitl()
Thread(target=mav_listener, args=(master,), daemon=True).start()

# ==== FPS counters ====
t0 = time.time()
f_counter = 0
fps = 0
frame_idx = 0

# -----------------------------------------------------------
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        lat, lon, alt  = state['lat'], state['lon'], state['alt']
        roll = state['roll']; pitch = state['pitch']; yaw = state['yaw']
        baro_alt = state['baro_alt']

        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, marker_size, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvecs[i], tvecs[i], 0.1)

                if None not in (lat, lon, alt, roll, pitch, yaw):
                    t_world = cam_to_world(tvecs[i][0], roll, pitch, yaw)
                    dist = np.linalg.norm(t_world)

                    R_earth = 6371000.0
                    dlat = (t_world[1] / R_earth) * 180/np.pi
                    dlon = (t_world[0] / (R_earth*np.cos(lat*np.pi/180))) * 180/np.pi
                    m_lat, m_lon = lat + dlat, lon + dlon
                    m_alt = (baro_alt if baro_alt is not None else alt) - t_world[2]

                    m_id = ids[i][0]
                    marker_data[m_id].append(
                        (m_lat, m_lon, m_alt, dist, frame_idx, frame.copy()))

        # ---------- On-screen overlays ----------
        if None not in (lat, lon, alt):
            gps_text = f"GPS: {lat:.6f}, {lon:.6f}, Alt: {alt:.2f}m"
            cv2.putText(frame, gps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(frame, "NO GPS LOCK FOUND", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        def d2str(angle): return f"{np.degrees(angle):.1f}°" if angle is not None else "—"
        cv2.putText(frame, f"Roll:  {d2str(roll)}",  (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,100), 2)
        cv2.putText(frame, f"Pitch: {d2str(pitch)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
        cv2.putText(frame, f"Yaw:   {d2str(yaw)}",   (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,255), 2)

        if None not in (roll, pitch, yaw):
            centre = (frame.shape[1]-120, 100)
            L = 50
            Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
            Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
            Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
            R = Rz @ Ry @ Rx

            axes = {
                (0,0,255): R @ np.array([1,0,0])*L,
                (0,255,0): R @ np.array([0,1,0])*L,
                (255,0,0): R @ np.array([0,0,1])*L
            }
            for col, vec in axes.items():
                p1 = centre
                p2 = (int(centre[0]+vec[0]), int(centre[1]-vec[1]))
                cv2.arrowedLine(frame, p1, p2, col, 2, tipLength=0.2)

        # ---------- FPS ----------
        f_counter += 1
        if time.time()-t0 >= 1.0:
            fps = f_counter/(time.time()-t0)
            t0 = time.time()
            f_counter = 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # 📹 Write to video file
        video_writer.write(frame)

        cv2.imshow("ArUco Detection", frame)
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    video_writer.release()  # 📹 Save the video
    cv2.destroyAllWindows()

# ---- Save closest marker detections ----
with open("global_coordinate.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["Marker ID","Latitude","Longitude","Altitude","Dist(m)","Frame"])
    for m_id, detections in marker_data.items():
        close = min(detections, key=lambda d: d[3])
        w.writerow([m_id,close[0],close[1],close[2],close[3],close[4]])
        img_name = os.path.join(out_dir, f"marker_{m_id}_frame_{close[4]}.png")
        cv2.imwrite(img_name, close[5])
        print(f"Saved snapshot → {img_name}")
