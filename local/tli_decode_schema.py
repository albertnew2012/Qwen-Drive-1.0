"""Work out the traffic-light label column layout by matching it to label_statistics.json.

The KITTI rows end in a block of integers with no header. label_statistics.json gives the
per-class totals for the same sequence, so summing each candidate column alignment and
comparing against those totals identifies the layout without guessing.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/perception_data/gravity_tli_data/batch_2083")
SEQ = "a4QC0c-gravity-USA-vin0054-20250518_174641"
CLASSES = ["Red_Solid", "Red_Left", "Red_Right", "Red_Straight", "Red_LeftDiagonal",
           "Red_RightDiagonal", "Yellow_Solid", "Yellow_Left", "Yellow_Right",
           "Yellow_Straight", "Yellow_LeftDiagonal", "Yellow_RightDiagonal",
           "Green_Solid", "Green_Left", "Green_Right", "Green_Straight",
           "Green_LeftDiagonal", "Green_RightDiagonal", "Bulb_Off"]

stats = json.loads((ROOT / "metadata" / SEQ / "label_statistics.json").read_text())
want = stats["objects"]["overall"]["traffic_lights"]
print("  published totals:", {k: v for k, v in want.items() if v})

sf = ROOT / "labels/batch_2083/traffic_lights/KITTI_SENSORFUSION" / f"{SEQ}.txt"
rows = [l.split() for l in sf.read_text().splitlines() if l.strip()]
print(f"\n  {len(rows)} rows, {len(rows[0])} fields each")
print(f"  example: {' '.join(rows[0])}")

ints = [[int(float(v)) for v in r[14:]] for r in rows]     # tail block after the pose
width = len(ints[0])
print(f"  tail block width {width}")

for off in range(0, width - len(CLASSES) + 1):
    got = Counter()
    for row in ints:
        for i, c in enumerate(CLASSES):
            v = row[off + i]
            if v:
                got[c] += v
    match = all(got.get(c, 0) == want.get(c, 0) for c in CLASSES)
    tag = "  <== MATCH" if match else ""
    if match or off < 3:
        print(f"    offset {off}: " + str({k: v for k, v in got.items() if v}) + tag)
