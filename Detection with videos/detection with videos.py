from ultralytics import YOLO
import cv2
import cvzone
import math
import time

#select the absoluth path of the video
cap = cv2.VideoCapture("insert absoluth path of video")  # For Video

#absoluth path of the model
model = YOLO("insert absoluth path of model")

classNames = ["car", "motorbike", "truck"]
#for frame rate of videos
prev_frame_time = 0
new_frame_time = 0

#Dimension for resizing the video (insert the correct width and height for the video)
#for cars video 1080 x 720
#for video1 1080 x 1520
#for video2 1920 x 1080
#for video3 1920 x 1080
#for video4 1920 x 1080
#for video5 1920 x 1080
#for video6 1920 x 1080
output_width = 1920
output_height = 1080

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    # resize video
    img = cv2.resize(img, (output_width, output_height))

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Custom bounding box with rounded corners and thicker borders
            color = (0, 255, 0)  # Green for bounding box
            thickness = 1  # Thickness of the rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            # Display class name and confidence
            text_position = (max(0, x1), max(35, y1))
            cv2.putText(img, f'{classNames[cls]} {conf}', text_position,cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
