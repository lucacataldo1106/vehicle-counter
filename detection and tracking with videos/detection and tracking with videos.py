import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
#absolute path of the video
cap = cv2.VideoCapture(r"C:\Users\Rudimental\PycharmProjects\car_detection\Detection with videos\videos\video5(1920x1080).mp4")

model = YOLO(r"C:\Users\Rudimental\PycharmProjects\PythonProject\runs\detect\train11\weights\best.pt")

classNames = ["car", "motorbike", "truck"]
#absolute path of the image mask
mask = cv2.imread(r"C:\Users\Rudimental\PycharmProjects\car_detection\detection and tracking with videos\mask5.png")
#verifying if mask is correctly loaded
if mask is None:
    print("Error: the mask is not correctly loaded")
    exit()

# Resize mask to match the target video dimensions(insert the correct width and height for the video)
#for cars 1080 x 720
#for video1 1080 x 1520
#for video2 1920 x 1080
#for video3 1920 x 1080
#for video4 1920 x 1080
#for video5 1920 x 1080
#for video6 1920 x 1080
output_width = 1920
output_height = 1080
resized_mask = cv2.resize(mask, (output_width, output_height))

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
#line for counting for cars
#limits = [400, 297, 673, 297]
#line for counting for video1
#limits = [400, 754,953, 754]
#line for counting for video2
#limits = [110, 670,1020, 670]
#line for counting for video3
#limits = [310, 750,1060, 750]
#line for counting for video4
#limits = [680, 650,1670, 650]
#line for counting for video5
limits = [650, 970,1350, 970]
#line for counting for video6
#limits = [350, 770,870, 770]

totalCount = []



while True:
    success, img = cap.read()
# verifying if mask is correctly loaded
    if not success:
        print("Error: the image is not correctly loaded")
        break

    # resize video
    img = cv2.resize(img, (output_width, output_height))

    #applying the mask
    imgRegion = cv2.bitwise_and(img, resized_mask)
    #image for the counting on top left
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)
    # detection array
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                #for only one type of vehicle detection
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                #only for detection not tracking
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    #tracker update
    resultsTracker = tracker.update(detections)
    #line for the counting
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    #for object tracking
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
#center point for the counting
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
#for the counting after the circle pass the line(x and y for the region in which happen the counting)
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            #whenever is detected a new count if the id is present in totalcount it wont be counted
            if totalCount.count(id) == 0:
                totalCount.append(id)
                #changing the color of the line when a new count occurs
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
    #printing count on the video
    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Image", img)
    #to show the mask
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(0)
