import cv2

aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

marker_id=0
marker_size=400

marker=cv2.aruco.generateImageMarker(aruco_dict,marker_id,marker_size)

cv2.imwrite("aruco_marker_0.png",marker)

cv2.imshow("window",marker)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Marker saves as aruco_marker_0.png")