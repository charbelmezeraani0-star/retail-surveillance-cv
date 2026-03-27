"""
detect_track.py
---------------
Full Pipeline: Detection + Tracking + Zones + Behavior + Alerts + Heatmap

Pipeline per frame:
    1. YOLO tracks all persons → stable track_ids
    2. For each person → compute foot point → get current zone
    3. Feed into BehaviorTracker → updates history + time + fires alerts
    4. Accumulate foot points into heatmap
    5. Draw zones, boxes, track_id, current zone, recent alerts on frame
    6. Save new alerts to CSV
    7. Write annotated frame to output video

At the end:
    - Saves heatmap image to data/logs/heatmap.png
    - Saves stats (unique visitors, total alerts) to data/logs/stats.json
"""

import csv
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from zones    import draw_zones, get_zone
from behavior import BehaviorTracker

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

ROOT           = Path(__file__).resolve().parent.parent
INPUT_VIDEO    = ROOT / "data" / "raw_videos"       / "test.mp4"
OUTPUT_VIDEO   = ROOT / "data" / "processed_videos" / "output_final.mp4"
ALERTS_CSV     = ROOT / "data" / "logs"             / "alerts.csv"
HEATMAP_IMG    = ROOT / "data" / "logs"             / "heatmap.png"
STATS_JSON     = ROOT / "data" / "logs"             / "stats.json"

MODEL_NAME     = "yolov8n.pt"
PERSON_CLASS   = 0
CONF_THRESHOLD = 0.4

RECENT_ALERT_FRAMES = 90   # frames to keep alert visible on screen
HEATMAP_RADIUS      = 20   # gaussian blob radius per foot point


# ──────────────────────────────────────────────
# COLORS
# ──────────────────────────────────────────────

def get_track_color(track_id: int) -> tuple:
    """Deterministic unique BGR color per track_id."""
    h = hashlib.md5(str(track_id).encode()).digest()
    return (int(h[0]), int(h[1]), int(h[2]))


# ──────────────────────────────────────────────
# HEATMAP
# ──────────────────────────────────────────────

def create_heatmap_accumulator(width: int, height: int) -> np.ndarray:
    """Create a blank float32 accumulator for the heatmap."""
    return np.zeros((height, width), dtype=np.float32)


def update_heatmap(accumulator: np.ndarray, foot: tuple):
    """
    Add a gaussian blob at the foot point position.

    Args:
        accumulator : float32 heatmap array (modified in-place)
        foot        : (x, y) pixel position
    """
    x, y = foot
    h, w = accumulator.shape

    # Build gaussian kernel bounds (clamped to image edges)
    r    = HEATMAP_RADIUS
    x1   = max(0, x - r);  x2 = min(w, x + r + 1)
    y1   = max(0, y - r);  y2 = min(h, y + r + 1)

    # Meshgrid for gaussian within bounds
    xs   = np.arange(x1, x2) - x
    ys   = np.arange(y1, y2) - y
    gx, gy = np.meshgrid(xs, ys)
    blob = np.exp(-(gx**2 + gy**2) / (2 * (r / 2)**2))

    accumulator[y1:y2, x1:x2] += blob


def save_heatmap(accumulator: np.ndarray, output_path: Path):
    """
    Normalize accumulator, apply color map, and save as PNG.

    Args:
        accumulator : raw float32 heatmap data
        output_path : where to save the PNG
    """
    if accumulator.max() == 0:
        return   # no data, skip

    # Normalize to 0–255
    norm = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)

    # Apply JET colormap (blue=cold, red=hot)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), colored)
    print(f"[DONE] Heatmap → {output_path}")


# ──────────────────────────────────────────────
# STATS
# ──────────────────────────────────────────────

def save_stats(path: Path, unique_visitors: int, total_alerts: int):
    """Save session stats to a JSON file for the dashboard to read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "unique_visitors": unique_visitors,
            "total_alerts"   : total_alerts,
        }, f)
    print(f"[DONE] Stats    → {path}")


# ──────────────────────────────────────────────
# DRAW HELPERS
# ──────────────────────────────────────────────

def draw_person(frame, box, track_id: int, confidence: float, zone: str | None):
    """Draw bounding box, track_id, confidence, and current zone for one person."""
    x1, y1, x2, y2 = map(int, box)
    color = get_track_color(track_id)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    line1 = f"ID {track_id}  {confidence:.0%}"
    line2 = f"Zone: {zone}" if zone else "Zone: —"

    for i, text in enumerate([line1, line2]):
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tw, th  = size
        ty      = y1 - 28 + i * 18
        cv2.rectangle(frame, (x1, ty - th - 3), (x1 + tw + 6, ty + 3), color, -1)
        cv2.putText(frame, text, (x1 + 3, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    foot = (int((x1 + x2) / 2), y2)
    cv2.circle(frame, foot, 4, color, -1)


def draw_recent_alerts(frame, recent_alerts: list):
    """Draw recent alerts as red text in the bottom-left corner."""
    if not recent_alerts:
        return

    frame_h = frame.shape[0]
    y_start = frame_h - 20 - (len(recent_alerts) - 1) * 24

    for alert in recent_alerts:
        text = (f"[{alert['alert_type']}]  "
                f"Person {alert['track_id']}  |  "
                f"{alert['zone']}  |  {alert['timestamp']}")
        cv2.putText(frame, text, (11, y_start + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, text, (10, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        y_start += 24


def draw_frame_info(frame, frame_number: int, person_count: int,
                    alert_count: int, visitor_count: int):
    """Draw frame stats in the top-left corner."""
    lines = [
        f"Frame    : {frame_number}",
        f"People   : {person_count}",
        f"Visitors : {visitor_count}",
        f"Alerts   : {alert_count}",
    ]
    y = 26
    for line in lines:
        cv2.putText(frame, line, (11, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 26


# ──────────────────────────────────────────────
# CSV
# ──────────────────────────────────────────────

CSV_COLUMNS = ["timestamp", "track_id", "alert_type", "zone"]

def init_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def append_alerts_to_csv(path: Path, alerts: list[dict]):
    if not alerts:
        return
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        for alert in alerts:
            writer.writerow({k: alert[k] for k in CSV_COLUMNS})


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def run():
    # ── Model ─────────────────────────────────
    print(f"[INFO] Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # ── Video ─────────────────────────────────
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Video not found: {INPUT_VIDEO}\n"
            f"Place your video at: data/raw_videos/test.mp4"
        )

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {INPUT_VIDEO}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {width}x{height}  {fps:.1f} fps  {total} frames")

    # ── Output video ──────────────────────────
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))
    print(f"[INFO] Output   → {OUTPUT_VIDEO}")

    # ── CSV + behavior ────────────────────────
    init_csv(ALERTS_CSV)
    print(f"[INFO] Alerts   → {ALERTS_CSV}")

    behavior = BehaviorTracker(fps=fps)

    # ── Heatmap accumulator ───────────────────
    heatmap_acc = create_heatmap_accumulator(width, height)

    # ── People counter ────────────────────────
    unique_visitors: set[int] = set()   # set of all track_ids ever seen

    saved_alert_index              = 0
    on_screen_alerts: list[tuple]  = []
    frame_number                   = 0

    print("[INFO] Press 'q' to quit early.\n")

    # ── Frame loop ────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # 1. Zones
        draw_zones(frame)

        # 2. YOLO tracking
        results = model.track(
            frame,
            classes=[PERSON_CLASS],
            conf=CONF_THRESHOLD,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        boxes        = results[0].boxes
        person_count = len(boxes)

        # 3. Per-person processing
        for box in boxes:
            coords     = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            track_id   = int(box.id[0]) if box.id is not None else -1

            x1, y1, x2, y2 = map(int, coords)
            foot = ((x1 + x2) // 2, y2)

            # People counter — register new visitor
            if track_id != -1:
                unique_visitors.add(track_id)

            # Zone + behavior
            zone = get_zone(foot)
            behavior.update(track_id, zone, frame_number)

            # Heatmap accumulation
            update_heatmap(heatmap_acc, foot)

            # Draw
            draw_person(frame, coords, track_id, confidence, zone)

        # 4. Alerts → CSV
        all_alerts = behavior.get_alerts()
        new_alerts = all_alerts[saved_alert_index:]
        if new_alerts:
            append_alerts_to_csv(ALERTS_CSV, new_alerts)
            for alert in new_alerts:
                on_screen_alerts.append((frame_number, alert))
            saved_alert_index = len(all_alerts)

        # 5. Recent alerts filter
        on_screen_alerts = [
            (fn, a) for fn, a in on_screen_alerts
            if frame_number - fn <= RECENT_ALERT_FRAMES
        ]

        # 6. Overlay
        draw_recent_alerts(frame, [a for _, a in on_screen_alerts])
        draw_frame_info(frame, frame_number, person_count,
                        len(all_alerts), len(unique_visitors))

        # 7. Save + display
        writer.write(frame)
        cv2.imshow("Retail Surveillance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit early by user.")
            break

    # ── Cleanup ───────────────────────────────
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # ── Save heatmap + stats ──────────────────
    save_heatmap(heatmap_acc, HEATMAP_IMG)
    save_stats(STATS_JSON, len(unique_visitors), len(behavior.get_alerts()))

    print(f"\n[DONE] {frame_number} frames processed.")
    print(f"[DONE] Unique visitors : {len(unique_visitors)}")
    print(f"[DONE] Total alerts    : {len(behavior.get_alerts())}")
    print(f"[DONE] Video  → {OUTPUT_VIDEO}")
    print(f"[DONE] Alerts → {ALERTS_CSV}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    run()
