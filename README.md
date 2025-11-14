# 🚗 Vehicle Counter with YOLOv8  
Real-time vehicle detection, tracking, and counting system

Developed by **Luca Cataldo** and **Valerio Di Gennaro**  
Course supervised by **Prof. Daniele Pannone**

---

## 📌 Project Overview

This project aims to build a system capable of:

- Detecting multiple vehicle categories (cars, motorcycles, trucks)
- Tracking moving vehicles along a road segment
- Accurately counting only vehicles in transit
- Ignoring irrelevant vehicles (parked, wrong-way, outside ROI)
- Applying custom masks to filter out non-relevant areas

The system is based on **Computer Vision**, **Deep Learning**, and **Object Tracking** techniques.

---

## 🧠 Detection Model: YOLOv8

We used **YOLOv8** for vehicle detection, experimenting with two main versions:

- **YOLOv8n** (nano) → lightweight and fast
- **YOLOv8l** (large) → more accurate, ideal for powerful GPUs

The model was trained on:

- A Roboflow dataset  
- A custom dataset of 50 manually annotated images created using **CVAT**

---

## 📊 Dataset and Preprocessing

### ✔️ Dataset characteristics
- Images and videos from various traffic scenarios  
- Classes: **car**, **motorcycle**, **truck**  
- Annotations in YOLO format  

### ✔️ Custom masks
Custom masks were applied to exclude:

- Parked vehicles  
- Wrong-way vehicles  
- Non-relevant lanes  

Masks were created using **Canva**.

---

## ⚙️ System Architecture

The system consists of three main components:

### 🔍 1) Vehicle Detection (YOLOv8)
Runs on static images and video frames, providing:

- Bounding boxes  
- Confidence scores  
- Vehicle class predictions  

### 🎯 2) Object Tracking (SORT)
The **SORT** tracker uses:

- **Kalman Filter** for state prediction  
- **IOU matching** for associating detections frame-to-frame  
- Unique persistent IDs for each vehicle  

### ➕ 3) Vehicle Counting
- A configurable **virtual line** counts vehicles when crossed  
- Line color changes (red → green) to confirm a successful count  
- A list of assigned IDs prevents duplicate counting  

---

## 🖥️ Technologies & Libraries

- **Ultralytics YOLOv8** – deep learning-based detection  
- **OpenCV** – image/video processing  
- **CVZone** – graphics overlay  
- **SORT Tracker** – object tracking  
- **NumPy** – array operations  
- **PyTorch / torchvision** – model training  
- **Matplotlib** – visualizations  
- **CUDA + cuDNN** – GPU acceleration  

---

## 🚦 Main Scripts

### 📌 1. Image Detection
Performs object detection on static images.

### 📌 2. Video Detection
Processes each frame of a video, showing:

- Bounding boxes  
- Class names  
- Confidence levels  

### 📌 3. Vehicle Counter
Combines detection and tracking to count vehicles:

- Customizable virtual line  
- Optional masks  
- Single counter for all vehicles  

(A 3-counter version for each vehicle class was attempted but abandoned due to SORT losing class information during tracking.)

---

## 📈 Results

The system performs well in:

- Dense traffic scenarios  
- Complex scenes with overlapping vehicles  
- Regions of interest defined by masks  
- Real-time video processing with GPU acceleration  

---

## ⚠️ Limitations

- Multi-class counting was not feasible due to SORT losing class labels  
- Some fast or partially occluded vehicles caused ID switching  
- Performance decreases with poor lighting or low-resolution videos  
- Masks require precise calibration to avoid false detections  

---

## 🚀 Future Improvements

- Expand the dataset (night, rain, fog, highways, urban scenes)  
- Replace SORT with a tracker that retains class information  
- Integrate ReID-based tracking for better identity persistence  
- Add multi-lane and multi-counter support  

---



## 📬 Contact
Author: Luca Cataldo  
Email: luca.cataldo1106@gmail.com  
GitHub: https://github.com/lucacataldo1106  
LinkedIn: https://www.linkedin.com/in/luca-cataldo1106/  


