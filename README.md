# RadPosNet – AI-powered tool for Radiographic Positioning Assessment

**RadPosNet** is an AI-powered tool designed to assist radiographers in evaluating **chest X-ray positioning and collimation** in real-time. It combines deep learning (YOLOv8) with MediaPipe pose estimation to detect common positioning issues and ensure image quality.

---

## 🧠 Overview
Proper patient positioning and collimation are crucial for accurate diagnostic imaging. RadPosNet automatically checks for:
- Shoulder tilt
- Spine rotation
- Obliquity
- PA position alignment
- Cassette placement
- Light field coverage and over-collimation

It provides real-time feedback to reduce errors, minimize retakes, and support radiography training.

---

## ⚙️ Features
- Detects key patient posture landmarks using **MediaPipe Pose**
- Identifies cassette and light field with **YOLOv8 object detection**
- Highlights positioning and collimation errors visually
- Designed for **CPU execution** (no GPU required)
- Supports radiographer training with live feedback

---

## 🛠️ Installation
```bash
pip install opencv-python mediapipe ultralytics torch numpy
