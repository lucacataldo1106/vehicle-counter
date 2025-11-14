from ultralytics import YOLO
import cv2

model = YOLO("insert the absoluth path of the model")
results = model("insert the absoluth path of the image", show=True)
cv2.waitKey(0)



