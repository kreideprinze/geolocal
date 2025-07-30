import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("yolov5su.pt")  


intrinsic_matrix = np.array([[650.90417285, 0, 318.97278063],
                             [0, 651.45358764, 236.01686148],
                             [0, 0, 1]])

focal_len_px = (intrinsic_matrix[0, 0] + intrinsic_matrix[1, 1]) / 2
plane2plane_dist_cm = 386  


cap = cv2.VideoCapture('output.avi')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 inference
    results = model.predict(source=frame, classes=[0], conf=0.5, verbose=False)

    image_center = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]])

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].item()

            # Bounding box center
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            box_center = np.array([cx, cy])

            # Pixel distance from image center
            pix_d_px = np.linalg.norm(box_center - image_center)

            # Estimate real-world distance
            true_dist_cm = (plane2plane_dist_cm * (pix_d_px / focal_len_px))

            # Draw annotations
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(image_center[0]), int(image_center[1])), 5, (255, 0, 0), -1)
            cv2.line(frame, (int(cx), int(cy)), (int(image_center[0]), int(image_center[1])), (0, 255, 0), 2)

            text = f"Dist: {true_dist_cm:.2f} cm"
            cv2.putText(frame, text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)

    cv2.imshow("YOLOv5 Human Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
