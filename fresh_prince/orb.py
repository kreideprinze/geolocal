import cv2
import numpy as np
import math
import yaml
from pymavlink import mavutil
from pyproj import Geod

# -------- Load Calibration (Kalibr YAML) --------
with open('camchain-imucam.yaml', 'r') as f:
    data = yaml.safe_load(f)
    T = data['cam0']['T_B_C']
    R_cam2body = np.array(T['R'])
    t_cam2body = np.array(T['p'])

# -------- Intrinsics (from calibration) --------
intrinsic_matrix = np.array([[650.90417285, 0, 318.97278063],
                             [0, 651.45358764, 236.01686148],
                             [0, 0, 1]])
focal_len_px = (intrinsic_matrix[0, 0] + intrinsic_matrix[1, 1]) / 2

# -------- ArUco --------
ar_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ar_params = cv2.aruco.DetectorParameters()

# -------- MAVLink Init --------
mav = mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)
mav.wait_heartbeat()
print("MAVLink connected")

# -------- Get Initial Altitude (MSL Ref) --------
def get_mav_data():
    msg = mav.recv_match(type=['GLOBAL_POSITION_INT', 'ATTITUDE'], blocking=True)
    if msg.get_type() == 'GLOBAL_POSITION_INT':
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.alt / 1000  # MSL in meters
        rel_alt = msg.relative_alt / 1000  # Above ground in meters
        return lat, lon, alt, rel_alt
    elif msg.get_type() == 'ATTITUDE':
        yaw = math.degrees(msg.yaw) % 360
        return yaw
    return None

init_lat, init_lon, init_alt, _ = get_mav_data()
alt_ref = init_alt  # Set MSL = 0 baseline
print(f"Reference Altitude: {alt_ref} m")

# -------- Projection Helper --------
geod = Geod(ellps="WGS84")
def offset_lat_lon(lat, lon, dx, dy):
    azimuth = math.degrees(math.atan2(dx, dy))
    distance = math.hypot(dx, dy)
    lon2, lat2, _ = geod.fwd(lon, lat, azimuth, distance)
    return lat2, lon2

# -------- Start Video --------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = cv2.aruco.detectMarkers(frame, ar_dict, parameters=ar_params)

    if ids is not None:
        aruco_corner = corners[0][0]
        aruco_center = np.mean(aruco_corner, axis=0)
        image_center = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]])

        pix_d_px = np.linalg.norm(aruco_center - image_center)

        # ---- Get telemetry ----
        lat, lon, alt, rel_alt = get_mav_data()
        heading = get_mav_data()  # second call gets ATTITUDE
        alt_above_ground = alt - alt_ref

        # ---- Compute 3D direction ----
        vec = np.array([aruco_center[0] - image_center[0],
                        aruco_center[1] - image_center[1],
                        focal_len_px])
        vec = vec / np.linalg.norm(vec)
        pos_cam = vec * rel_alt  # In meters

        # ---- Transform to drone body frame ----
        pos_body = R_cam2body @ pos_cam + t_cam2body

        # ---- Estimate Global Position ----
        dx, dy = pos_body[0], pos_body[1]
        marker_lat, marker_lon = offset_lat_lon(lat, lon, dx, dy)

        # ---- Visualization ----
        cv2.circle(frame, tuple(aruco_center.astype(int)), 5, (0, 0, 255), -1)
        cv2.circle(frame, tuple(image_center.astype(int)), 5, (255, 0, 0), -1)
        cv2.line(frame, tuple(image_center.astype(int)), tuple(aruco_center.astype(int)), (0, 255, 0), 4)
        text = f"Rel Alt: {rel_alt:.2f} m | Marker: {marker_lat:.7f}, {marker_lon:.7f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
