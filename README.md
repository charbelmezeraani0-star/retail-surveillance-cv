# Real-Time Intelligent Retail Surveillance

A complete computer vision system for retail store monitoring using **YOLOv8**, **ByteTrack**, and **Streamlit**.

Detects people, tracks their movement across store zones, analyzes behavior, triggers alerts, and visualizes everything on a live dashboard.

---

## Demo

### Movement Heatmap
![Heatmap](data/logs/heatmap.png)

---

## Features

- **Person Detection** — YOLOv8 detects only people, ignoring all other objects
- **Multi-Person Tracking** — ByteTrack assigns a stable ID to each person across frames
- **Zone Detection** — 5 store zones defined as rectangles (entrance, shelf, checkout, exit, restricted)
- **Behavior Analysis** — tracks zone history and time spent per zone for each person
- **Automated Alerts** — 3 rule-based alert types triggered in real time
- **Movement Heatmap** — shows where people spent the most time
- **People Counter** — counts unique visitors throughout the video
- **CSV Logging** — all alerts saved with timestamp, person ID, type, and zone
- **Streamlit Dashboard** — live dashboard with charts, timeline, table, heatmap, and video playback

---

## Alert Rules

| Alert | Trigger |
|---|---|
| `LOITERING` | Person stays in shelf zone for more than 10 seconds |
| `RESTRICTED_ZONE` | Person enters the restricted area (fires once per person) |
| `SKIP_CHECKOUT` | Person visited shelf then exit without ever visiting checkout |

---

## Project Structure

```
retail_surveillance/
│
├── app/
│   ├── detect_track.py   # Main pipeline: detection, tracking, zones, alerts, heatmap
│   ├── zones.py          # Zone definitions, drawing, and lookup
│   ├── behavior.py       # Per-person behavior tracking and alert rules
│   └── dashboard.py      # Streamlit dashboard
│
├── data/
│   ├── raw_videos/       # Input video (test.mp4)
│   ├── processed_videos/ # Annotated output video
│   └── logs/             # alerts.csv, heatmap.png, stats.json
│
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/retail-surveillance-cv.git
cd retail-surveillance-cv

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### 1. Add your video
Place any MP4 video at:
```
data/raw_videos/test.mp4
```

### 2. Run the pipeline
```bash
python app/detect_track.py
```

This will:
- Open a live window showing detection, tracking, zones, and alerts
- Save the annotated video to `data/processed_videos/output_final.mp4`
- Save all alerts to `data/logs/alerts.csv`
- Save the movement heatmap to `data/logs/heatmap.png`
- Save session stats to `data/logs/stats.json`

Press `q` to stop early.

### 3. Convert video for browser playback
```bash
ffmpeg -i data/processed_videos/output_final.mp4 -vcodec libx264 -crf 23 data/processed_videos/output_web.mp4
```

### 4. Launch the dashboard
```bash
streamlit run app/dashboard.py
```

Open your browser at `http://localhost:8501`

---

## Dashboard

The Streamlit dashboard displays:

- **Unique visitor count** and **total alerts**
- **Alerts by type** — bar chart
- **Alerts over time** — area chart timeline
- **Latest alerts table** — color-coded by alert type
- **Per-person alert summary** — expandable table
- **Movement heatmap** — visual foot traffic map
- **Processed video** — embedded playback

Auto-refreshes every 5 seconds.

---

## Tech Stack

| Component | Technology |
|---|---|
| Person Detection | YOLOv8n (Ultralytics) |
| Multi-Object Tracking | ByteTrack |
| Video Processing | OpenCV |
| Dashboard | Streamlit |
| Data | Pandas, NumPy |
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

## Author

**Charbel Mezeraani**
Master's Student — Saint Joseph University, Lebanon

---

## License

This project is for academic and educational purposes.
