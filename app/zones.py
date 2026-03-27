"""
zones.py
--------
Zone definitions for the retail surveillance system.

Each zone is a named rectangle: (x1, y1, x2, y2)
Coordinates are based on a 1280x720 reference frame.
Adjust values to match your actual video resolution and camera layout.

Zones:
    entrance   - door area where people enter
    shelf      - product shelf area to monitor for loitering
    checkout   - cashier / payment area
    exit       - door area where people leave
    restricted - staff-only or off-limits area
"""

# ──────────────────────────────────────────────
# ZONE DEFINITIONS  (x1, y1, x2, y2)
# ──────────────────────────────────────────────
# Tweak these rectangles to match your video layout.

ZONES = {
    "entrance"   : (  0,   0, 200, 720),   # left strip
    "shelf"      : (200, 100, 750, 620),   # large central area
    "checkout"   : (750, 300, 980, 620),   # right-center
    "exit"       : (980,   0, 1280, 720),  # right strip
    "restricted" : (200,   0, 750,  100),  # top band above shelf
}

# Display color per zone (BGR)
ZONE_COLORS = {
    "entrance"   : (0,   255,   0),    # green
    "shelf"      : (255, 165,   0),    # orange
    "checkout"   : (255, 255,   0),    # yellow
    "exit"       : (0,   200, 255),    # cyan
    "restricted" : (0,     0, 255),    # red
}

ZONE_ALPHA = 0.15   # transparency of the filled zone overlay


# ──────────────────────────────────────────────
# PUBLIC FUNCTIONS
# ──────────────────────────────────────────────

def draw_zones(frame):
    """
    Draw all zones on the frame as semi-transparent filled rectangles
    with a solid border and zone name label.

    Args:
        frame : video frame (numpy array, modified in-place)
    """
    import cv2
    import numpy as np

    overlay = frame.copy()

    for name, (x1, y1, x2, y2) in ZONES.items():
        color = ZONE_COLORS[name]

        # Semi-transparent fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Solid border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Zone name label (top-left corner of zone)
        label = name.upper()
        cv2.putText(
            frame, label,
            (x1 + 6, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, color, 2
        )

    # Blend overlay onto frame for the transparent fill effect
    cv2.addWeighted(overlay, ZONE_ALPHA, frame, 1 - ZONE_ALPHA, 0, frame)


def get_zone(point: tuple) -> str | None:
    """
    Return the name of the zone that contains the given point.
    If the point is in multiple zones, the first match in ZONES is returned.
    Returns None if the point is not inside any zone.

    Args:
        point : (x, y) pixel coordinate — typically the bottom-center
                of a person's bounding box (feet position)

    Returns:
        Zone name string (e.g. "shelf") or None
    """
    x, y = point

    for name, (x1, y1, x2, y2) in ZONES.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name

    return None


# ──────────────────────────────────────────────
# QUICK VISUAL TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import cv2
    import numpy as np

    # Create a blank 1280x720 canvas to preview zones
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_zones(canvas)

    # Test get_zone with a few sample points
    test_points = [
        (100, 360),   # should be: entrance
        (480, 350),   # should be: shelf
        (860, 450),   # should be: checkout
        (1100, 360),  # should be: exit
        (400,  50),   # should be: restricted
        (1000, 50),   # should be: None (exit strip top)
    ]

    for pt in test_points:
        zone = get_zone(pt)
        color = ZONE_COLORS.get(zone, (200, 200, 200)) if zone else (200, 200, 200)
        cv2.circle(canvas, pt, 8, color, -1)
        cv2.putText(canvas, str(zone), (pt[0] + 10, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        print(f"  Point {pt} → zone: {zone}")

    cv2.imshow("Zone Preview — press any key to close", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
