# 🧪 AI-Powered Laboratory Safety and Compliance System
AI-powered laboratory safety and compliance monitoring system using Computer Vision, Deep Learning, OCR, and Transfer Learning for PPE detection, waste classification, and chemical expiry monitoring.


<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A real-time, multi-model AI system for automated laboratory safety monitoring using Computer Vision and Deep Learning**

*BIE300 Mini Project | SASTRA Deemed to be University, School of Chemical & Biotechnology*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Models](#-models)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Future Work](#-future-work)
- [Team](#-team)
- [References](#-references)
- [License](#-license)

---

## 🔬 Overview

Laboratory environments demand strict adherence to safety protocols to prevent accidents and regulatory violations. Traditional manual supervision is inconsistent, error-prone, and incapable of continuous monitoring. This project addresses these limitations by introducing an **AI-powered, real-time safety compliance system** that integrates multiple deep learning models under a unified decision engine.

> **Key Insight:** Nearly 25–30% of laboratory accidents occur due to PPE negligence and incorrect/expired chemical usage (OSHA Lab Safety Reports). This system targets those exact failure points.

The system monitors:
- **Personal Protective Equipment (PPE)** — lab coats, gloves, goggles, masks, footwear
- **Waste Disposal Classification** — hazardous, biological, recyclable, general
- **Chemical Expiry Detection** — OCR + CNN pipeline for reading and verifying dates
- **Real-Time Alerts** — screen pop-ups, sound alarms, log generation

---

## ✨ Key Features

| Feature | Description |
|--------|-------------|
| 🥽 **PPE Detection** | Real-time multi-label detection of lab coats, gloves, goggles, masks, footwear |
| 🗑️ **Waste Classification** | GoogLeNet-based classification into 4 waste categories |
| ⏰ **Expiry Detection** | OCR + CNN hybrid pipeline for chemical label reading |
| 🚨 **Decision Engine** | Rule-based contextual safety alert system (ISO-aligned) |
| 📊 **Dashboard** | Live compliance dashboard with violation logging |
| 📁 **Report Export** | PDF/Excel reports for safety audits |
| 🎥 **Camera Support** | Compatible with USB and CCTV cameras |
| ⚡ **Real-Time** | 15–20 FPS detection on mid-range hardware |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAMERA INPUT (USB / CCTV)                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────▼───────────────┐
           │     Frame Preprocessor         │
           │  (Resize, Normalize, Augment)  │
           └───────────────┬───────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐        ┌────▼────┐       ┌────▼────┐
   │  PPE    │        │  Waste  │       │ Expiry  │
   │ Detect  │        │ Classif │       │ Detect  │
   │(Custom  │        │(Google  │       │(OCR +   │
   │  CNN)   │        │  Net)   │       │  CNN)   │
   └────┬────┘        └────┬────┘       └────┬────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
           ┌───────────────▼───────────────┐
           │    Contextual Decision Engine  │
           │  (Rule-based + ISO Standards)  │
           └───────────────┬───────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐       ┌─────▼────┐      ┌─────▼────┐
   │ Screen  │       │  Logger  │      │ Report   │
   │  Alert  │       │(Timestamp│      │Generator │
   │  + BBox │       │  + ID)   │      │(PDF/XLSX)│
   └─────────┘       └──────────┘      └──────────┘
```

---

## 🤖 Models

### Model 1: PPE Detection — Custom CNN

Detects presence/absence of 5 PPE components simultaneously.

**Architecture:**
```
Input (224×224×3)
  → Conv2D(32) → ReLU → MaxPool
  → Conv2D(64) → ReLU → MaxPool
  → Flatten
  → Dense(128) → ReLU → Dropout(0.3)
  → Softmax(5 classes)
```

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 48 |
| Epochs | 10 |
| Loss | Categorical Cross-Entropy |

---

### Model 2: Waste Classification — GoogLeNet (Transfer Learning)

Classifies waste into 4 categories: **Hazardous / Biological / Recyclable / General**

**Why GoogLeNet?**
- Pretrained on ImageNet (1.2M images)
- Lightweight with Inception modules (multi-scale feature extraction)
- 22 layers with 1×1 reduction for computational efficiency
- Excellent performance on small custom datasets

**Transfer Learning Modifications:**
- Replaced final FC + Softmax + Classification layers
- Output: 4 waste categories
- New layers trained at 20× higher learning rate

---

### Model 3: Chemical Expiry Detection — CNN + OCR Hybrid

A 5-stage pipeline for automatic expiry date reading and verification.

```
Stage 1: Bottle Detection (Object Detection)
Stage 2: Label Region Crop
Stage 3: OCR (Tesseract / EasyOCR) → "Exp. Date 03/2019"
Stage 4: CNN confirms digit sequences
Stage 5: Date comparison → Safe / Near-Expiry / Expired
```

**Model Evolution:**

| Version | Train Acc | Val Acc | Issue |
|---------|-----------|---------|-------|
| Baseline CNN | 95% | 79.5% | Overfitting |
| + Heavy Dropout + Augmentation | — | 67.5% | Underfitting |
| Balanced (Dropout=0.1, 30 epochs) | **82%** | **83%** | ✅ Stable |

---

## 💾 Installation

### Prerequisites

- Python 3.9+
- pip / conda
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-lab-safety.git
cd ai-lab-safety

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Install Tesseract OCR

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

---

## 🚀 Usage

### 1. Run Real-Time Detection (Webcam)

```bash
python src/main.py --source 0 --mode all
```

### 2. Run on Video File

```bash
python src/main.py --source path/to/video.mp4 --mode all
```

### 3. Run Individual Modules

```bash
# PPE Detection only
python src/models/ppe_detector.py --source 0

# Waste Classification
python src/models/waste_classifier.py --image path/to/image.jpg

# Expiry Detection
python src/models/expiry_detector.py --image path/to/label.jpg
```

### 4. Launch Dashboard

```bash
python src/dashboard/app.py
# Open http://localhost:5000 in your browser
```

### 5. Generate Audit Report

```bash
python src/utils/report_generator.py --start 2025-01-01 --end 2025-12-31 --format pdf
```

---

## 📁 Project Structure

```
ai-lab-safety/
│
├── README.md                    ← You are here
├── requirements.txt             ← Python dependencies
├── LICENSE                      ← MIT License
├── .gitignore
│
├── configs/
│   ├── model_config.yaml        ← Model hyperparameters
│   ├── alert_config.yaml        ← Alert thresholds and rules
│   └── camera_config.yaml       ← Camera settings
│
├── src/
│   ├── main.py                  ← Entry point
│   ├── models/
│   │   ├── ppe_detector.py      ← Custom CNN for PPE detection
│   │   ├── waste_classifier.py  ← GoogLeNet waste classifier
│   │   ├── expiry_detector.py   ← OCR + CNN expiry pipeline
│   │   └── decision_engine.py   ← Contextual safety engine
│   ├── utils/
│   │   ├── preprocessor.py      ← Frame preprocessing utilities
│   │   ├── annotator.py         ← Bounding box drawing
│   │   ├── logger.py            ← Violation logging
│   │   └── report_generator.py  ← PDF/Excel report export
│   ├── alerts/
│   │   ├── screen_alert.py      ← On-screen popup alerts
│   │   └── sound_alert.py       ← Audio alarm system
│   └── dashboard/
│       ├── app.py               ← Flask/Dash dashboard
│       └── templates/
│
├── data/
│   ├── raw/                     ← Raw collected images
│   ├── processed/               ← Preprocessed datasets
│   └── annotations/             ← LabelImg annotation XMLs/JSONs
│
├── notebooks/
│   ├── 01_ppe_training.ipynb
│   ├── 02_waste_classification.ipynb
│   ├── 03_expiry_detection.ipynb
│   └── 04_system_evaluation.ipynb
│
├── tests/
│   ├── test_ppe_detector.py
│   ├── test_waste_classifier.py
│   └── test_expiry_detector.py
│
└── docs/
    ├── FINAL_REPORT.pdf
    ├── architecture_diagram.png
    └── demo.gif
```

---

## 📊 Results

### PPE Detection
- High accuracy for clear, well-lit images with unobstructed PPE
- Consistent bounding box predictions for multi-person frames
- Minor accuracy drop under low lighting and occlusions

### Waste Classification (GoogLeNet)
- Strong generalization across custom lab bin datasets
- Stable performance under varying lighting conditions
- Misclassification only in heavily cluttered / overlapping waste scenarios

### Expiry Detection
- **99.9% CNN confidence** on digit classification
- Successfully parsed curved, slanted, and partially faded labels
- Final model: **82% train / 83% validation accuracy** (balanced, no overfitting)

### System Performance

| Metric | Value |
|--------|-------|
| Detection Speed | 15–20 FPS |
| Hardware Requirement | Mid-range CPU / GPU |
| Camera Compatibility | USB + CCTV |
| Alert Latency | < 1 second |
| Plagiarism Score | 7–13% (Paperpal) |

---

## 🔮 Future Work

1. **YOLO-Pose Integration** — Unsafe posture detection (pipetting without gloves, hazardous reach)
2. **Fire & Spill Detection** — Flame, smoke, liquid spill, and broken glass recognition
3. **Cloud Dashboard** — Multi-lab remote monitoring with historical analytics
4. **Edge Deployment** — Jetson Nano / Raspberry Pi optimized models
5. **Expanded Dataset** — Handwritten dates, unusual PPE, extreme lighting scenarios
6. **Adaptive Learning** — Continuous retraining from flagged violations
7. **IoT Integration** — Smart bins, badge scanners, environmental sensors

---

## 👥 Team

| Name | Reg. No. | Contribution |
|------|----------|--------------|
| **Adarsh R** | 126011001 | PPE Detection Model Development |
| **Lokesh S** | 126011026 | Waste Classification Model Training |
| **Sai Ganesh L** | 126011037 | Chemical Expiry Detection System |

**Supervisor:** Dr. Hemalatha Karnan, Asst. Professor–II, School of Chemical and Biotechnology, SASTRA Deemed to be University

---

## 📚 References

1. Everingham et al. (2010). *The PASCAL VOC Challenge.* IJCV, 88(2), 303–338.
2. Gonzalez & Woods (2018). *Digital Image Processing* (4th ed.). Pearson Education.
3. Goodfellow, Bengio & Courville (2016). *Deep Learning.* MIT Press.
4. Howard et al. (2017). *MobileNets: Efficient CNNs for Mobile Vision.*
5. Krizhevsky, Sutskever & Hinton (2012). *ImageNet Classification with Deep CNNs.* NIPS.
6. Redmon & Farhadi (2018). *YOLOv3: An Incremental Improvement.*
7. Szegedy et al. (2015). *Going Deeper with Convolutions* (GoogLeNet). CVPR.
8. Tesseract OCR Engine Documentation. Google Developers (2023).
9. U.S. OSHA (2021). *Laboratory Safety Guidance.*

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with 🔬 at **SASTRA Deemed to be University**  
School of Chemical & Biotechnology, Thanjavur, Tamil Nadu, India

</div>
