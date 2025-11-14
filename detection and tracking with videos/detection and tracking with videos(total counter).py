import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

#absolute path of the video
video_capture = cv2.VideoCapture(r"insert the absoluth path of video")

# YOLO model path
yolo_model = YOLO(r"insert the absoluth path of model")

# Classes of vehicles
vehicle_classes = ["car", "motorbike", "truck"]

#absolute path of the image mask
mask_image = cv2.imread(r"insert the absoluth path of mask")
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
output_width = 1920
output_height = 1080
resized_mask = cv2.resize(mask_image, (output_width, output_height))

# Tracking
object_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

#line for counting for cars
#limits = [400, 297, 673, 297]
#line for counting for video1
#limits = [400, 754,953, 754]
#line for counting for video2
limits = [110, 670, 1020, 670]
#line for counting for video3
#limits = [310, 750,1060, 750]
#line for counting for video4
#limits = [680, 650,1670, 650]
#line for counting for video5
#limits = [650, 970,1350, 970]
#line for counting for video6
#limits = [350, 770,870, 770]

tracked_objects = []

while True:
    success, video_frame = video_capture.read()
    # verifying if mask is correctly loaded
    if not success:
        print("Error: the image is not correctly loaded")
        break

    # resize video
    video_frame = cv2.resize(video_frame, (output_width, output_height))

    #applying the mask
    masked_region = cv2.bitwise_and(video_frame, resized_mask)

    #image for the counting on top left
    counter_graphic = cv2.imread(r"insert absoluth path of carcounterBlue.png", cv2.IMREAD_UNCHANGED)
    video_frame = cvzone.overlayPNG(video_frame, counter_graphic, (0, 0))

    # YOLO detection
    detection_results = yolo_model(masked_region, stream=True)

    # detection array
    detections_array = np.empty((0, 5))

    for result in detection_results:
        bounding_boxes = result.boxes
        for box in bounding_boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            box_width, box_height = x2 - x1, y2 - y1

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            detected_class_index = int(box.cls[0])
            detected_class = vehicle_classes[detected_class_index]

            if detected_class == "car" or detected_class == "truck" or detected_class == "motorbike" and confidence > 0.3:
                current_detection = np.array([x1, y1, x2, y2, confidence])
                detections_array = np.vstack((detections_array, current_detection))

    #tracker update
    tracked_objects_results = object_tracker.update(detections_array)

    #line for the counting
    cv2.line(video_frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    #for object tracking
    for tracked_object in tracked_objects_results:
        x1, y1, x2, y2, object_id = tracked_object
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        box_width, box_height = x2 - x1, y2 - y1

        print(tracked_object)
        cvzone.cornerRect(video_frame, (x1, y1, box_width, box_height), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(video_frame, f' {int(object_id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        #center point for the counting
        center_x, center_y = x1 + box_width // 2, y1 + box_height // 2
        cv2.circle(video_frame, (center_x, center_y), 5, (255, 0, 255), cv2.FILLED)
        #for the counting after the circle pass the line(x and y for the region in which happen the counting)
        if limits[0] < center_x < limits[2] and limits[1] - 15 < center_y < limits[1] + 15:
            #whenever is detected a new count if the id is present in tracked_objects it wont be counted
            if tracked_objects.count(object_id) == 0:
                tracked_objects.append(object_id)
                #changing the color of the line when a new count occurs
                cv2.line(video_frame, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 5)

    #printing count on the video
    cv2.putText(video_frame, str(len(tracked_objects)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Image", video_frame)
    #to show the mask on video uncomment
    #cv2.imshow("ImageRegion", masked_region)
    cv2.waitKey(0)
