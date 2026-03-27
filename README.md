# Retail Surveillance CV

> Real-time intelligent retail surveillance system using computer vision — built as a Master's level project at Saint Joseph University, Lebanon.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-green?style=flat-square)

---

## Overview

This project implements a complete, modular computer vision pipeline for retail store monitoring. It detects and tracks people in video footage, analyzes their behavior across predefined store zones, triggers rule-based alerts, generates a movement heatmap, and presents all results on a live Streamlit dashboard.

The system is designed for fixed-camera scenarios and is fully runnable on a standard laptop without a GPU.

---

## Demo

### Movement Heatmap
> Red = high foot traffic | Blue = low foot traffic

![Heatmap](data/logs/heatmap.png)

---

## Key Features

| Feature | Description |
|---|---|
| Person Detection | YOLOv8n detects only people, filtering all other classes |
| Multi-Person Tracking | ByteTrack assigns stable IDs across frames, handles occlusion |
| Zone Management | 5 rectangular store zones with real-time foot-point lookup |
| Behavior Analysis | Per-person zone history, visit sequence, and time-in-zone tracking |
| Automated Alerts | 3 rule-based alert types triggered and logged in real time |
| Movement Heatmap | Gaussian-based heatmap of foot positions saved as PNG |
| People Counter | Counts unique visitors throughout the entire video |
| CSV Logging | All alerts saved with timestamp, person ID, type, and zone |
| Live Dashboard | Streamlit dashboard with charts, heatmap, video, and auto-refresh |

---

## Alert Rules

| Alert Type | Trigger Condition |
|---|---|
| `LOITERING` | Person remains in the shelf zone for more than 10 seconds |
| `RESTRICTED_ZONE` | Person enters the restricted area (fires once per person) |
| `SKIP_CHECKOUT` | Person visited shelf → exit without ever visiting checkout |

---

## System Architecture

```
Video Input (MP4)
      │
      ▼
┌─────────────────────┐
│   YOLOv8 Detection  │  ← detects persons only (class 0)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   ByteTrack         │  ← assigns stable track_id per person
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Zone Lookup       │  ← maps foot point → zone name
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Behavior Tracker  │  ← updates history, time, triggers alerts
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
Heatmap    Alerts CSV
    │         │
    └────┬────┘
         ▼
  Streamlit Dashboard
```

---

## Project Structure

```
retail_surveillance/
│
├── app/
│   ├── detect_track.py     # Main pipeline — detection, tracking, zones, alerts, heatmap
│   ├── zones.py            # Zone definitions, drawing, and point lookup
│   ├── behavior.py         # Per-person behavior tracking and 3 alert rules
│   └── dashboard.py        # Streamlit live dashboard
│
├── data/
│   ├── raw_videos/         # Input video (test.mp4)
│   ├── processed_videos/   # Annotated output video
│   └── logs/               # alerts.csv · heatmap.png · stats.json
│
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/charbelmezeraani0-star/retail-surveillance-cv.git
cd retail-surveillance-cv

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### Step 1 — Add your video
```bash
cp /path/to/your/video.mp4 data/raw_videos/test.mp4
```

### Step 2 — Run the pipeline
```bash
python app/detect_track.py
```

**Outputs:**
- Live annotated window (press `q` to stop early)
- `data/processed_videos/output_final.mp4` — annotated video
- `data/logs/alerts.csv` — all triggered alerts
- `data/logs/heatmap.png` — movement heatmap
- `data/logs/stats.json` — visitor count and total alerts

### Step 3 — Convert video for browser playback
```bash
ffmpeg -i data/processed_videos/output_final.mp4 -vcodec libx264 -crf 23 data/processed_videos/output_web.mp4
```

### Step 4 — Launch the dashboard
```bash
streamlit run app/dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## Dashboard Sections

| Section | Description |
|---|---|
| Metrics Row | Unique visitors · Total alerts · Per-type counts |
| Alerts by Type | Bar chart of alert distribution |
| Alerts Over Time | Area chart of alerts across video timeline |
| Latest Alerts Table | Color-coded table of most recent alerts |
| Per-Person Summary | Expandable breakdown of alerts per track ID |
| Movement Heatmap | Visual foot traffic map of the scene |
| Video Playback | Embedded processed video with annotations |

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8n — Ultralytics |
| Multi-Object Tracking | ByteTrack |
| Video Processing | OpenCV |
| Numerical Computing | NumPy |
| Data Handling | Pandas |
| Dashboard | Streamlit |
| Video Conversion | FFmpeg |

---

## Requirements

```
ultralytics>=8.0.0
opencv-python>=4.8.0
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
```

---

## Academic Context

This project was developed as part of a Master's degree in Computer Vision at **Saint Joseph University (USJ), Lebanon**.

It demonstrates a full end-to-end CV pipeline covering:
- Real-time object detection and tracking
- Spatial reasoning with zone mapping
- Rule-based behavior analysis
- Data visualization and dashboard design
- Modular, production-ready code architecture

---

## Author

**Charbel Mezeraani**
Master's Student — Saint Joseph University, Zahle, Lebanon
