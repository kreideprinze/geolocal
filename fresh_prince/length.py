import cv2
import numpy as np

intrinsic_matrix = np.array([[650.90417285, 0, 318.97278063],
                             [0, 651.45358764, 236.01686148],
                             [0, 0, 1]])

focal_len_px = (intrinsic_matrix[0, 0] + intrinsic_matrix[1, 1]) / 2
plane2plane_dist_cm = 386

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        aruco_corner = corners[0][0]
        aruco_center = np.mean(aruco_corner, axis=0)  

        image_center = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]])  

        pix_d_px = np.linalg.norm(aruco_center - image_center)
        true_dist_cm = (plane2plane_dist_cm * (pix_d_px / focal_len_px))

        
        cv2.circle(frame, tuple(aruco_center.astype(int)), 5, (0, 0, 255), -1)
        cv2.circle(frame, tuple(image_center.astype(int)), 5, (255, 0, 0), -1)
        cv2.line(frame, tuple(image_center.astype(int)), tuple(aruco_center.astype(int)), (0, 255, 0), 4)
        text = f"Distance: {true_dist_cm:.2f} cm"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

      
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
