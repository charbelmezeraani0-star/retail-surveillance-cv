"""
behavior.py
-----------
Per-person behavior tracking + alert rules for the retail surveillance system.

Tracks per person:
    - track_id       : unique integer assigned by the tracker
    - current_zone   : zone the person is currently in (or None)
    - zone_history   : ordered list of zones visited  e.g. ["entrance", "shelf"]
    - time_per_zone  : seconds spent in each zone     e.g. {"shelf": 14.3}

Alert rules:
    1. LOITERING       — person in "shelf" zone > LOITER_THRESHOLD seconds
    2. RESTRICTED      — person enters "restricted" zone (triggers once per person)
    3. SKIP_CHECKOUT   — person visited shelf then exit, never visited checkout

Each alert is a dict:
    {
        "timestamp"  : str   e.g. "00:01:23"
        "track_id"   : int
        "alert_type" : str   e.g. "LOITERING"
        "zone"       : str
    }
"""

from collections import defaultdict
from datetime import timedelta

# ──────────────────────────────────────────────
# ALERT CONFIG
# ──────────────────────────────────────────────

LOITER_THRESHOLD = 10.0   # seconds in shelf before loitering alert


class BehaviorTracker:
    """
    Tracks movement and time-in-zone for every person in the scene.
    One instance lives for the entire video — call update() once per frame
    for each detected person.
    """

    def __init__(self, fps: float = 30.0):
        """
        Args:
            fps : frames per second of the source video,
                  used to convert frame counts → seconds
        """
        self.fps = fps

        # ── Per-person state ──────────────────────────────────────────────
        self._current_zone:     dict[int, str | None]  = defaultdict(lambda: None)
        self._zone_history:     dict[int, list]         = defaultdict(list)
        self._time_per_zone:    dict[int, dict]         = defaultdict(lambda: defaultdict(float))
        self._zone_entry_frame: dict[int, int]          = {}

        # ── Alert state ───────────────────────────────────────────────────
        self._alerts: list[dict] = []

        # Flags to ensure each alert fires only once per person
        self._loiter_alerted:        set[int] = set()
        self._restricted_alerted:    set[int] = set()
        self._skip_checkout_alerted: set[int] = set()

    # ──────────────────────────────────────────────
    # MAIN UPDATE  (call once per frame per person)
    # ──────────────────────────────────────────────

    def update(self, track_id: int, zone: str | None, frame_number: int):
        """
        Update state for one person and check all alert rules.

        Args:
            track_id     : stable integer ID from the tracker
            zone         : zone the person is currently in, or None
            frame_number : current frame index (used for time + timestamp)
        """
        previous_zone = self._current_zone[track_id]

        # ── Zone change ───────────────────────────
        if zone != previous_zone:

            # Commit time for the zone being left
            if previous_zone is not None and track_id in self._zone_entry_frame:
                frames_in_zone = frame_number - self._zone_entry_frame[track_id]
                self._time_per_zone[track_id][previous_zone] += frames_in_zone / self.fps

            # Update current zone
            self._current_zone[track_id] = zone

            if zone is not None:
                self._zone_entry_frame[track_id] = frame_number

                history = self._zone_history[track_id]
                if not history or history[-1] != zone:
                    history.append(zone)

                print(f"[BEHAVIOR] Person {track_id} entered {zone} zone")
            else:
                print(f"[BEHAVIOR] Person {track_id} left all zones")

        # ── Check alert rules every frame ─────────
        self._check_loitering(track_id, frame_number)
        self._check_restricted(track_id, frame_number)
        self._check_skip_checkout(track_id, frame_number)

    # ──────────────────────────────────────────────
    # ALERT RULES
    # ──────────────────────────────────────────────

    def _check_loitering(self, track_id: int, frame_number: int):
        """
        Rule 1 — LOITERING
        Trigger if person has been in 'shelf' zone for more than LOITER_THRESHOLD seconds.
        Fires once per person.
        """
        if track_id in self._loiter_alerted:
            return

        if self._current_zone[track_id] != "shelf":
            return

        # Time committed to shelf + time in current ongoing shelf visit
        committed   = self._time_per_zone[track_id].get("shelf", 0.0)
        entry_frame = self._zone_entry_frame.get(track_id, frame_number)
        ongoing     = (frame_number - entry_frame) / self.fps
        total_shelf_time = committed + ongoing

        if total_shelf_time > LOITER_THRESHOLD:
            alert = self._make_alert(track_id, "LOITERING", "shelf", frame_number)
            self._alerts.append(alert)
            self._loiter_alerted.add(track_id)
            print(f"[ALERT] {alert['timestamp']} — Person {track_id} "
                  f"LOITERING in shelf ({total_shelf_time:.1f}s)")

    def _check_restricted(self, track_id: int, frame_number: int):
        """
        Rule 2 — RESTRICTED ZONE
        Trigger once when a person enters the 'restricted' zone.
        """
        if track_id in self._restricted_alerted:
            return

        if self._current_zone[track_id] == "restricted":
            alert = self._make_alert(track_id, "RESTRICTED_ZONE", "restricted", frame_number)
            self._alerts.append(alert)
            self._restricted_alerted.add(track_id)
            print(f"[ALERT] {alert['timestamp']} — Person {track_id} "
                  f"entered RESTRICTED zone")

    def _check_skip_checkout(self, track_id: int, frame_number: int):
        """
        Rule 3 — SHELF → EXIT WITHOUT CHECKOUT
        Trigger if person visited shelf, is now at exit, and never visited checkout.
        Fires once per person.
        """
        if track_id in self._skip_checkout_alerted:
            return

        if self._current_zone[track_id] != "exit":
            return

        history = self._zone_history[track_id]

        visited_shelf    = "shelf"    in history
        visited_exit     = "exit"     in history
        visited_checkout = "checkout" in history

        if visited_shelf and visited_exit and not visited_checkout:
            alert = self._make_alert(track_id, "SKIP_CHECKOUT", "exit", frame_number)
            self._alerts.append(alert)
            self._skip_checkout_alerted.add(track_id)
            print(f"[ALERT] {alert['timestamp']} — Person {track_id} "
                  f"left via exit WITHOUT visiting checkout")

    # ──────────────────────────────────────────────
    # ALERT BUILDER
    # ──────────────────────────────────────────────

    def _make_alert(self, track_id: int, alert_type: str,
                    zone: str, frame_number: int) -> dict:
        """
        Build a standardised alert dictionary.

        Returns:
            {
                "timestamp"  : "HH:MM:SS"
                "track_id"   : int
                "alert_type" : str
                "zone"       : str
            }
        """
        seconds  = int(frame_number / self.fps)
        timestamp = str(timedelta(seconds=seconds))   # "0:00:07" format

        return {
            "timestamp"  : timestamp,
            "track_id"   : track_id,
            "alert_type" : alert_type,
            "zone"       : zone,
        }

    # ──────────────────────────────────────────────
    # ACCESSORS
    # ──────────────────────────────────────────────

    def get_alerts(self) -> list[dict]:
        """Return all alerts generated so far."""
        return list(self._alerts)

    def get_new_alerts(self, since_index: int) -> list[dict]:
        """
        Return only alerts generated after a given index.
        Useful for displaying recent alerts on the video frame.

        Args:
            since_index : return alerts from this index onwards
        """
        return self._alerts[since_index:]

    def get(self, track_id: int) -> dict:
        """Return full tracking snapshot for one person."""
        return {
            "track_id"     : track_id,
            "current_zone" : self._current_zone[track_id],
            "zone_history" : list(self._zone_history[track_id]),
            "time_per_zone": dict(self._time_per_zone[track_id]),
        }

    def get_current_zone(self, track_id: int) -> str | None:
        return self._current_zone[track_id]

    def get_zone_history(self, track_id: int) -> list:
        return list(self._zone_history[track_id])

    def get_time_in_zone(self, track_id: int, zone: str) -> float:
        return self._time_per_zone[track_id].get(zone, 0.0)

    def all_ids(self) -> list:
        return list(self._current_zone.keys())

    # ──────────────────────────────────────────────
    # DEBUG
    # ──────────────────────────────────────────────

    def print_summary(self, track_id: int):
        info = self.get(track_id)
        print(f"\n── Person {track_id} ─────────────────────────")
        print(f"  Current zone  : {info['current_zone']}")
        print(f"  Zone history  : {' → '.join(info['zone_history']) or 'none'}")
        time_str = ", ".join(
            f"{z}: {t:.1f}s" for z, t in info["time_per_zone"].items()
        ) or "none"
        print(f"  Time per zone : {time_str}")

    def print_all_summaries(self):
        for tid in self.all_ids():
            self.print_summary(tid)


# ──────────────────────────────────────────────
# QUICK TEST  (no video needed)
# ──────────────────────────────────────────────

if __name__ == "__main__":

    tracker = BehaviorTracker(fps=30.0)

    # Person 1: normal path — entrance → shelf (3s) → checkout → exit  (no alerts)
    events = [
        (1, "entrance",   1),
        (1, "shelf",      30),   # enters shelf at frame 30
        (1, "checkout",   120),  # leaves shelf after 3s — under threshold
        (1, "exit",       180),
    ]

    # Person 2: loiterer + skip checkout — shelf 16s, never goes to checkout
    events += [
        (2, "entrance",   1),
        (2, "shelf",      20),   # enters shelf
        (2, "shelf",      500),  # still in shelf at frame 500 (~16s) → LOITERING
        (2, "exit",       501),  # exits without checkout → SKIP_CHECKOUT
    ]

    # Person 3: enters restricted zone → RESTRICTED_ZONE alert
    events += [
        (3, "entrance",   5),
        (3, "restricted", 40),
    ]

    print("=" * 50)
    print("  Simulating behavior events")
    print("=" * 50)

    for track_id, zone, frame in events:
        tracker.update(track_id, zone, frame)

    print("\n" + "=" * 50)
    print("  Final Summaries")
    print("=" * 50)
    tracker.print_all_summaries()

    print("\n" + "=" * 50)
    print("  All Alerts")
    print("=" * 50)
    for alert in tracker.get_alerts():
        print(f"  [{alert['alert_type']}] "
              f"Person {alert['track_id']} | "
              f"Zone: {alert['zone']} | "
              f"Time: {alert['timestamp']}")
