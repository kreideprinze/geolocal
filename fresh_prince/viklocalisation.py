import cv2
import numpy as np
import csv
import threading
import time
import signal
import sys
from pymavlink import mavutil
from ultralytics import YOLO
from math import sin, cos, radians
import queue
import os

# ---------------------- CONFIG ----------------------
VIDEO_DEVICE = 0
OUTPUT_VIDEO = "output_tracked.avi"
DETECTIONS_CSV = "detections.csv"
WEIGHTED_CSV = "weighted_detections.csv"

# Detection throttle: run heavy detection every N frames (1 = every frame)
DETECT_EVERY_N_FRAMES = 2

# Maximum frames allowed queued for detection (keeps memory bounded)
MAX_DET_QUEUE = 2

# Whether to show window (set False for headless to save CPU)
SHOW_WINDOW = True

# Path order for model backends to try (engine/onnx first if you've exported)
PREFERRED_MODELS = ["model.engine", "model.onnx", "yolov5n.pt", "yolov8n.pt", "yolov5su.pt"]

# Confidence and classes
CONFIDENCE = 0.45
CLASS_IDS = [0]  # person
# ----------------------------------------------------

# Camera intrinsics (D455) - kept from your script
intrinsic_matrix = np.array([[650.90417285, 0, 318.97278063],
                             [0, 651.45358764, 236.01686148],
                             [0, 0, 1]])
image_center = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]])
focal_len_px = (intrinsic_matrix[0, 0] + intrinsic_matrix[1, 1]) / 2

# Telemetry shared state
telemetry_lock = threading.Lock()
current_lat = None
current_lon = None
current_alt = None
current_yaw_deg = None
current_roll = 0
current_pitch = 0
base_alt = None

# Running flag
running = True

# Track detections and weighted averaging
tracked_ids = {}
weighted_results = {}

# Queues for threads
det_queue = queue.Queue(maxsize=MAX_DET_QUEUE)     # frames to be detected
annot_queue = queue.Queue(maxsize=MAX_DET_QUEUE)   # annotated frames for writer / display
csv_queue = queue.Queue()                          # rows to write to csv
weighted_queue = queue.Queue()                     # weighted summary rows
video_write_queue = queue.Queue(maxsize=100)       # frames to write to video

# Video writer will be created after we know frame size/fps
out_writer = None

# Graceful shutdown
def signal_handler(sig, frame):
    global running
    print("\n[INFO] Ctrl+C detected. Exiting cleanly...")
    running = False
signal.signal(signal.SIGINT, signal_handler)

# ---------------------- TELEMETRY THREAD ----------------------
def telemetry_thread_fn(connection_str="udp:127.0.0.1:14550"):
    global current_lat, current_lon, current_alt, base_alt, current_yaw_deg, current_roll, current_pitch
    try:
        master = mavutil.mavlink_connection(connection_str)
        master.wait_heartbeat(timeout=10)
        print("[INFO] MAVLink connected.")
    except Exception as e:
        print(f"[WARN] MAVLink connect failed: {e}. Telemetry will be empty.")
        master = None

    while running:
        if master is None:
            time.sleep(0.2)
            continue
        # Non-blocking small timeout so we don't stall
        msg = master.recv_match(type=['GLOBAL_POSITION_INT', 'ATTITUDE'], blocking=False)
        if msg is None:
            # small sleep to avoid busy loop
            time.sleep(0.01)
            continue
        with telemetry_lock:
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

# ---------------------- UTIL FUNCTIONS ----------------------
def offset_gps(lat, lon, distance_m, heading_deg):
    R = 6378137
    heading_rad = radians(heading_deg)
    dlat = (distance_m * cos(heading_rad)) / R
    dlon = (distance_m * sin(heading_rad)) / (R * cos(radians(lat)))
    new_lat = lat + dlat * (180 / np.pi)
    new_lon = lon + dlon * (180 / np.pi)
    return new_lat, new_lon

# ---------------------- CSV WRITER THREAD ----------------------
def csv_writer_thread_fn(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Frame", "ID", "Altitude_m", "GroundDist_m",
            "Person_Lat", "Person_Lon", "Drone_Lat", "Drone_Lon"
        ])
        while running or not csv_queue.empty():
            try:
                row = csv_queue.get(timeout=0.2)
                writer.writerow(row)
                csv_queue.task_done()
            except queue.Empty:
                continue

# ---------------------- WEIGHTED CSV THREAD ----------------------
def weighted_writer_thread_fn(weighted_path):
    with open(weighted_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Avg_Lat", "Avg_Lon", "Min_Weighted_Dist"])
        while running or not weighted_queue.empty():
            try:
                row = weighted_queue.get(timeout=0.2)
                writer.writerow(row)
                weighted_queue.task_done()
            except queue.Empty:
                continue

# ---------------------- VIDEO WRITER THREAD ----------------------
def video_writer_thread_fn(writer_params):
    global out_writer
    out_writer = cv2.VideoWriter(*writer_params)
    while running or not video_write_queue.empty():
        try:
            frame = video_write_queue.get(timeout=0.2)
            out_writer.write(frame)
            video_write_queue.task_done()
        except queue.Empty:
            continue
    out_writer.release()

# ---------------------- DETECTION WORKER ----------------------
def detection_worker_fn(model):
    """
    Pops frames from det_queue, runs detection, and pushes annotated detection results
    (as a list of boxes+meta) for the main loop to use.
    """
    while running:
        try:
            payload = det_queue.get(timeout=0.5)  # (frame_num, frame_copy)
        except queue.Empty:
            continue
        frame_num, frame_bgr = payload
        # Run prediction on BGR frame (model expects BGR or RGB depending on backend; Ultralytics handles it)
        try:
            results = model.predict(source=frame_bgr, conf=CONFIDENCE, classes=CLASS_IDS, verbose=False)
        except Exception as e:
            print("[WARN] Detection failed:", e)
            results = []

        # Extract boxes and pass them back in a lightweight structure
        dets = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                dets.append((x1, y1, x2, y2, conf))
        # Put results into annot_queue for annotation/display by main thread
        try:
            annot_queue.put_nowait((frame_num, dets))
        except queue.Full:
            # if annot queue is full, drop results (keeps pipeline real-time)
            pass
        det_queue.task_done()

# ---------------------- MODEL LOADER ----------------------
def load_best_model(preferred_list):
    """
    Try loading an accelerated backend if present (engine/onnx), otherwise fallback to small cpu model.
    """
    for p in preferred_list:
        if not os.path.exists(p):
            continue
        try:
            print(f"[INFO] Loading model: {p}")
            model = YOLO(p)
            # Try to use half precision if supported by backend
            try:
                model.model = getattr(model, "model", model.model)
                model.to("cpu")
                # attempt to use half if supported
                try:
                    model.model.half()
                    print("[INFO] half-precision enabled (if supported).")
                except Exception:
                    pass
            except Exception:
                pass
            return model
        except Exception as e:
            print(f"[WARN] Could not load {p}: {e}")
            continue
    # If none found, fallback to ultralight default (attempt to download if not present)
    print("[INFO] No preferred model found on disk, loading 'yolov5n.pt' from ultralytics (if available).")
    try:
        model = YOLO("yolov5n.pt")
        return model
    except Exception as e:
        print(f"[ERROR] failed to load fallback model: {e}")
        raise

# ---------------------- MAIN ----------------------
def main():
    global running, out_writer

    # Load model (accelerated engine first if present)
    model = load_best_model(PREFERRED_MODELS)

    # Start telemetry thread
    threading.Thread(target=telemetry_thread_fn, daemon=True).start()

    # Start CSV writer thread
    threading.Thread(target=csv_writer_thread_fn, args=(DETECTIONS_CSV,), daemon=True).start()
    threading.Thread(target=weighted_writer_thread_fn, args=(WEIGHTED_CSV,), daemon=True).start()

    # Open camera
    cap = cv2.VideoCapture(VIDEO_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[ERROR] Could not open camera device.")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Start video writer thread
    writer_params = (OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    threading.Thread(target=video_writer_thread_fn, args=(writer_params,), daemon=True).start()

    # Start detection worker
    threading.Thread(target=detection_worker_fn, args=(model,), daemon=True).start()

    frame_num = 0
    last_detections = []  # cached last detection for annotating skipped frames
    last_detection_frame = -999

    fps_time_start = time.time()
    frame_counter = 0
    fps_value = 0.0

    while running:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed; breaking.")
            break
        frame_num += 1
        frame_counter += 1
        if frame_counter >= 30:
            now = time.time()
            elapsed = now - fps_time_start
            fps_value = frame_counter / max(1e-6, elapsed)
            fps_time_start = now
            frame_counter = 0

        # read telemetry lock-free snapshot
        with telemetry_lock:
            lat = current_lat
            lon = current_lon
            alt = current_alt
            yaw_deg = current_yaw_deg
            roll = current_roll
            pitch = current_pitch

        # Determine if we should perform detection this frame
        do_detect = (frame_num - last_detection_frame) >= DETECT_EVERY_N_FRAMES
        if do_detect:
            # push a copy to detection queue if there's room
            try:
                det_queue.put_nowait((frame_num, frame.copy()))
                last_detection_frame = frame_num
            except queue.Full:
                # queue full -> skip scheduling; will annotate with last_detections
                pass

        # Try to get latest detection results (non-blocking)
        try:
            while True:
                # empty the queue to get the latest detection (drop older)
                frame_id, dets = annot_queue.get_nowait()
                last_detections = dets
                annot_queue.task_done()
        except queue.Empty:
            pass

        # Annotate frame using last_detections
        for (x1, y1, x2, y2, conf) in last_detections:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            box_center = np.array([cx, cy])
            pix_d_px = np.linalg.norm(box_center - image_center)
            # ground distance formula preserved (plane2plane_dist_m * pixel/focal)
            if alt is None or alt < 0 or yaw_deg is None:
                ground_dist_m = -1.0
                person_lat = None
                person_lon = None
                id_str = f"nogps_{frame_num}_{int(cx)}_{int(cy)}"
            else:
                plane2plane_dist_m = alt
                ground_dist_cm = plane2plane_dist_m * (pix_d_px / focal_len_px)
                ground_dist_m = ground_dist_cm / 100.0
                person_lat, person_lon = offset_gps(lat, lon, ground_dist_m, yaw_deg)
                id_str = f"{person_lat:.6f}_{person_lon:.6f}"

            weight = 1.0 / (1.0 + abs(roll) + abs(pitch) + max(0.001, ground_dist_m if ground_dist_m>0 else 0.001))
            if id_str not in weighted_results:
                weighted_results[id_str] = []
            weighted_results[id_str].append((person_lat, person_lon, weight, ground_dist_m))
            tracked_ids[id_str] = (person_lat, person_lon)

            delta_north = ground_dist_m * cos(radians(yaw_deg)) if ground_dist_m>=0 else 0.0
            delta_east = ground_dist_m * np.sin(radians(yaw_deg)) if ground_dist_m>=0 else 0.0
            breakdown_label = f"{ground_dist_m:.2f}m | N:{delta_north:.2f} E:{delta_east:.2f}"

            label = f"{id_str}\nLat:{person_lat if person_lat else 'NaN'}, Lon:{person_lon if person_lon else 'NaN'}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(image_center[0]), int(image_center[1])), 5, (255, 0, 0), -1)
            cv2.line(frame, (int(cx), int(cy)), (int(image_center[0]), int(image_center[1])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 50), 2)
            cv2.putText(frame, breakdown_label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 2)

            # push CSV row async
            csv_queue.put([
                frame_num,
                id_str,
                round(alt, 2) if alt is not None else "NaN",
                round(ground_dist_m, 2) if ground_dist_m>=0 else "NaN",
                person_lat if person_lat is not None else "NaN",
                person_lon if person_lon is not None else "NaN",
                lat if lat is not None else "NaN",
                lon if lon is not None else "NaN"
            ])

        # Draw GPS/status and FPS
        if lat is None or lon is None:
            cv2.putText(frame, "No GPS Lock", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            gps_status = f"Lat: {lat:.6f}, Lon: {lon:.6f}, Alt: {alt:.2f}m"
            cv2.putText(frame, gps_status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps_value:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        # Offer frame to video writer queue (non-blocking)
        try:
            video_write_queue.put_nowait(frame.copy())
        except queue.Full:
            # drop if writer is falling behind
            pass

        # Show window if requested
        if SHOW_WINDOW:
            cv2.imshow("Detection Feed", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                running = False
                break

    # End main loop
    cap.release()
    cv2.destroyAllWindows()

    # Wait for queues to empty and finish
    print("[INFO] Waiting for queues to drain...")
    det_queue.join()
    annot_queue.join()
    video_write_queue.join()
    csv_queue.join()
    weighted_queue.join()

    # Write weighted results to weighted CSV asynchronously or directly
    for pid, records in weighted_results.items():
        total_weight = sum(w for _, _, w, _ in records) or 1.0
        # guard for None lat/lon in sum
        lat_sum = sum((lat * w) for lat, _, w, _ in records if lat is not None)
        lon_sum = sum((lon * w) for _, lon, w, _ in records if lon is not None)
        avg_lat = (lat_sum / total_weight) if total_weight>0 else None
        avg_lon = (lon_sum / total_weight) if total_weight>0 else None
        min_dist = min((d for _, _, _, d in records if d is not None), default=-1)
        weighted_queue.put([pid, avg_lat, avg_lon, min_dist])

    # allow writers to flush
    time.sleep(1.0)
    print("[INFO] All done.")

if __name__ == "__main__":
    main()
