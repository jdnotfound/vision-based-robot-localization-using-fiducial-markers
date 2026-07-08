import cv2

tag_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_16h5
)

tag_id = 0
tag_size_px = 700

tag_img = cv2.aruco.generateImageMarker(
    tag_dict,
    tag_id,
    tag_size_px
)

cv2.imwrite("apriltag_16h5_id0.png", tag_img)

print("Saved apriltag_16h5_id0.png")