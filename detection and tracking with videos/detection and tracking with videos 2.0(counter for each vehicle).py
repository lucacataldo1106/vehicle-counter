import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
#absolute path of the video
video_capture = cv2.VideoCapture(r"insert absoluth path of video")

yolo_model = YOLO(r"insert absoluth path of model")

vehicle_classes = ["car", "motorbike", "truck"]
#absolute path of the image mask insert the right one
mask_image = cv2.imread(r"insert absoluth path of mask")
#verifying if mask is correctly loaded
if mask_image is None:
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
#insert the right dimension below
output_width = 1920
output_height = 1080
resized_mask = cv2.resize(mask_image, (output_width, output_height))

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
#line for counting for cars, insert the right lines for the video
#limits = [320, 297, 600, 297]
#line for counting for video1
#limits = [400, 754,953, 754]
#line for counting for video2
limits = [110, 670,1020, 670]
#line for counting for video3
#limits = [310, 750,1060, 750]
#line for counting for video4
#limits = [680, 650,1670, 650]
#line for counting for video5
#limits = [650, 970,1350, 970]
#line for counting for video6
#limits = [350, 770,870, 770]

totalCars = []
totalMotorbikes = []
totalTrucks = []
tracker_classes={}

while True:
    success, img = video_capture.read()
# verifying if mask is correctly loaded
    if not success:
        print("Error: the image is not correctly loaded")
        break

    # resize video
    img = cv2.resize(img, (output_width, output_height))

    #applying the mask
    masked_region = cv2.bitwise_and(img, resized_mask)
    # Load graphics for each counter
    carGraphics = cv2.imread(r"absoluth path of carcounterBlue",cv2.IMREAD_UNCHANGED)
    motorbikeGraphics = cv2.imread(r"absoluth path of motocounterRed",cv2.IMREAD_UNCHANGED)
    truckGraphics = cv2.imread(r"absoluth path of truckcounterGReen",cv2.IMREAD_UNCHANGED)

    # Resize graphics to make them smaller
    carGraphics = cv2.resize(carGraphics, (150, 60))
    motorbikeGraphics = cv2.resize(motorbikeGraphics, (150, 60))
    truckGraphics = cv2.resize(truckGraphics, (150, 60))

    # Add graphics for counters to the video frame
    img = cvzone.overlayPNG(img, carGraphics, (20, 20))
    img = cvzone.overlayPNG(img, motorbikeGraphics, (20, 100))  # Position below carGraphics
    img = cvzone.overlayPNG(img, truckGraphics, (20, 180))  # Position below motorbikeGraphics

    detection_results = yolo_model(masked_region, stream=True)
    # Detection array
    detected_objects = np.empty((0, 5))
    objects_classes = []

    for r in detection_results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            box_width, box_height = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = vehicle_classes[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "motorbike" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detected_objects = np.vstack((detected_objects, currentArray))
                objects_classes.insert(0, currentClass)

    # Tracker update
    tracking_results = tracker.update(detected_objects)
    # Line for the counting
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    # Object tracking
    for i,tracked_object in enumerate(tracking_results):
        x1, y1, x2, y2, Id = tracked_object
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        box_width, box_height = x2 - x1, y2 - y1
        if Id not in tracker_classes:
            tracker_classes[Id] = objects_classes[i]
            print("i:", i, "Id:", Id, "\n", "trak:", tracker_classes, "\n", "obj:", objects_classes)

        currentClass = tracker_classes.get(Id, "unknown")
        # Colors for the bounding boxes
        if currentClass == "car":
            bounding_box_color = (255, 0, 0)  # Blue for cars
        elif currentClass == "motorbike":
            bounding_box_color = (0, 0, 255)  # Red for motorbikes
        elif currentClass == "truck":
            bounding_box_color = (0, 255, 0)  # Green for trucks
        else:
            bounding_box_color = (255, 255, 255)
        cvzone.cornerRect(img, (x1, y1, box_width, box_height), l=9, rt=2, colorR=bounding_box_color)
        cvzone.putTextRect(img, f' {Id}', (max(0, x1), max(35, y1)),scale=2, thickness=2, offset=10)
        # Center point for counting
        cx, cy = x1 + box_width // 2, y1 + box_height // 2
        cv2.circle(img, (cx, cy), 5, bounding_box_color, cv2.FILLED)
        # Counting logic
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if Id not in totalCars and Id not in totalMotorbikes and Id not in totalTrucks:
                if currentClass == "car":
                    totalCars.append(Id)
                elif currentClass == "motorbike":
                    totalMotorbikes.append(Id)
                elif currentClass == "truck":
                    totalTrucks.append(Id)
                # Change the color of the line when a new count occurs
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Display counts for each category
    cv2.putText(img, f'Cars: {len(totalCars)}', (40, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    cv2.putText(img, f'Motorbikes: {len(totalMotorbikes)}', (40, 140), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    cv2.putText(img, f'Trucks: {len(totalTrucks)}', (40, 220), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    cv2.imshow("Image", img)
    # Show the mask in another window uncomment
    #cv2.imshow("ImageRegion", masked_region)
    cv2.waitKey(0)
