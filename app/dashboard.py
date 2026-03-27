"""
dashboard.py
------------
Streamlit dashboard for the Retail Surveillance system.

Displays:
    - Unique visitor count + total alerts (from stats.json)
    - Alert breakdown by type (bar chart)
    - Alerts over time (timeline chart)
    - Latest alerts table
    - Movement heatmap image
    - Processed video playback
    - Per-person alert summary

Auto-refreshes every 5 seconds.

Run:
    streamlit run app/dashboard.py
"""

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

ROOT          = Path(__file__).resolve().parent.parent
ALERTS_CSV    = ROOT / "data" / "logs"             / "alerts.csv"
STATS_JSON    = ROOT / "data" / "logs"             / "stats.json"
HEATMAP_IMG   = ROOT / "data" / "logs"             / "heatmap.png"
OUTPUT_VIDEO  = ROOT / "data" / "processed_videos" / "output_web.mp4"

REFRESH_INTERVAL = 5

ALERT_COLORS = {
    "LOITERING"       : "#e67e22",
    "RESTRICTED_ZONE" : "#e74c3c",
    "SKIP_CHECKOUT"   : "#8e44ad",
}


# ──────────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────────

def load_alerts() -> pd.DataFrame:
    if not ALERTS_CSV.exists():
        return pd.DataFrame(columns=["timestamp", "track_id", "alert_type", "zone"])
    try:
        df = pd.read_csv(ALERTS_CSV)
        if df.empty:
            return df
        df["track_id"] = df["track_id"].astype(int)
        return df
    except Exception:
        return pd.DataFrame(columns=["timestamp", "track_id", "alert_type", "zone"])


def load_stats() -> dict:
    if not STATS_JSON.exists():
        return {"unique_visitors": 0, "total_alerts": 0}
    try:
        with open(STATS_JSON) as f:
            return json.load(f)
    except Exception:
        return {"unique_visitors": 0, "total_alerts": 0}


# ──────────────────────────────────────────────
# UI COMPONENTS
# ──────────────────────────────────────────────

def render_header():
    st.title("Retail Surveillance Dashboard")
    st.caption(f"Auto-refreshes every {REFRESH_INTERVAL}s")
    st.divider()


def render_summary_cards(df: pd.DataFrame, stats: dict):
    """Top row: visitors + total alerts + per type counts."""
    all_types = ["LOITERING", "RESTRICTED_ZONE", "SKIP_CHECKOUT"]
    cols      = st.columns(2 + len(all_types))

    with cols[0]:
        st.metric("Unique Visitors", stats.get("unique_visitors", 0))

    with cols[1]:
        st.metric("Total Alerts", len(df))

    for col, alert_type in zip(cols[2:], all_types):
        count = len(df[df["alert_type"] == alert_type]) if not df.empty else 0
        with col:
            st.metric(alert_type.replace("_", " ").title(), count)


def render_bar_chart(df: pd.DataFrame):
    """Alert count per type as a bar chart."""
    st.subheader("Alerts by Type")
    if df.empty:
        st.info("No alerts recorded yet.")
        return

    counts = (
        df["alert_type"]
        .value_counts()
        .rename_axis("Alert Type")
        .reset_index(name="Count")
    )
    st.bar_chart(counts.set_index("Alert Type")["Count"])


def render_timeline(df: pd.DataFrame):
    """
    Alerts over time — one dot per alert plotted by timestamp.
    Converts HH:MM:SS timestamps to total seconds for the x-axis.
    """
    st.subheader("Alerts Over Time")
    if df.empty:
        st.info("No alerts recorded yet.")
        return

    def ts_to_seconds(ts: str) -> int:
        parts = str(ts).split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return 0
        return 0

    timeline = df.copy()
    timeline["seconds"] = timeline["timestamp"].apply(ts_to_seconds)
    timeline["count"]   = 1

    # Group by second bucket + alert type
    grouped = (
        timeline.groupby(["seconds", "alert_type"])["count"]
        .sum()
        .reset_index()
        .pivot(index="seconds", columns="alert_type", values="count")
        .fillna(0)
    )
    st.area_chart(grouped)


def render_latest_table(df: pd.DataFrame, n: int = 20):
    """Most recent N alerts as a styled table."""
    st.subheader(f"Latest Alerts (last {n})")
    if df.empty:
        st.info("No alerts recorded yet.")
        return

    latest = df.tail(n).iloc[::-1].reset_index(drop=True)
    latest.index += 1

    def highlight_row(row):
        color = ALERT_COLORS.get(row["alert_type"], "#2c3e50")
        return [f"color: {color}; font-weight: bold" if col == "alert_type"
                else "" for col in row.index]

    st.dataframe(
        latest.style.apply(highlight_row, axis=1),
        use_container_width=True,
    )


def render_per_person_summary(df: pd.DataFrame):
    """Expandable: alert count per person."""
    if df.empty:
        return
    with st.expander("Per-Person Alert Summary"):
        summary = (
            df.groupby("track_id")["alert_type"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
            .rename(columns={"track_id": "Person ID"})
        )
        st.dataframe(summary, use_container_width=True)


def render_heatmap():
    """Show the movement heatmap image if it exists."""
    st.subheader("Movement Heatmap")
    if not HEATMAP_IMG.exists():
        st.info("Heatmap not generated yet. Run the pipeline first.")
        return
    st.image(str(HEATMAP_IMG), caption="Foot position heatmap — red = most visited",
             use_container_width=True)


def render_video():
    """Embed the processed output video for playback."""
    st.subheader("Processed Video")
    if not OUTPUT_VIDEO.exists():
        st.info("Processed video not found. Run the pipeline first.")
        return
    st.video(str(OUTPUT_VIDEO))


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Retail Surveillance",
        page_icon="🏪",
        layout="wide",
    )

    render_header()

    df    = load_alerts()
    stats = load_stats()

    # ── Row 1: summary metrics ────────────────
    render_summary_cards(df, stats)
    st.divider()

    # ── Row 2: charts ─────────────────────────
    left, right = st.columns([1, 2])
    with left:
        render_bar_chart(df)
    with right:
        render_timeline(df)

    st.divider()

    # ── Row 3: table ──────────────────────────
    render_latest_table(df)
    render_per_person_summary(df)

    st.divider()

    # ── Row 4: heatmap + video ────────────────
    col_heat, col_vid = st.columns(2)
    with col_heat:
        render_heatmap()
    with col_vid:
        render_video()

    # ── Auto-refresh ──────────────────────────
    st.divider()
    st.caption(f"Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')}")
    time.sleep(REFRESH_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()
