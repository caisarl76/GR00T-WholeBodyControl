"""Overlay D435i stream-profile FOVs on the widest view, in tangent-plane coordinates.

x = tan(horizontal angle), y = tan(vertical angle, +down): pinhole image rectangles map to rectangles.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

cap_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
meta = json.loads((cap_dir / "capture_metadata.json").read_text())["profiles"]


def from_intrinsics(p):
    """Exact image bounds from calibrated intrinsics (zero distortion on this device)."""
    return (
        -p["ppx"] / p["fx"],
        (p["width"] - p["ppx"]) / p["fx"],
        -p["ppy"] / p["fy"],
        (p["height"] - p["ppy"]) / p["fy"],
    )


def from_fov(h_deg, v_deg):
    """Centered bounds when only the calibrated FOV is known."""
    h, v = math.tan(math.radians(h_deg / 2)), math.tan(math.radians(v_deg / 2))
    return (-h, h, -v, v)


hd = from_intrinsics(meta["1280x720"])
vga = from_intrinsics(meta["640x480"])


def vla_view(p, edge=256, crop=230 / 256, factor=32):
    """GR00T N1.7 eval transform: SmallestMaxSize(256) -> center crop 230/256 -> SmallestMaxSize(256),
    then Qwen3-VL rounds each side to a multiple of 32 (patch 16 x merge 2). Returns (bounds, input size)."""
    w, h = p["width"], p["height"]
    s = edge / min(w, h)
    w1, h1 = round(w * s), round(h * s)
    cw, ch = int(w1 * crop), int(h1 * crop)
    s2 = edge / min(cw, ch)
    qw, qh = round(cw * s2 / factor) * factor, round(ch * s2 / factor) * factor
    # Crop is centered on the image center, expressed back in source pixels.
    x0, x1 = (w - w * cw / w1) / 2, (w + w * cw / w1) / 2
    y0, y1 = (h - h * ch / h1) / 2, (h + h * ch / h1) / 2
    bounds = ((x0 - p["ppx"]) / p["fx"], (x1 - p["ppx"]) / p["fx"], (y0 - p["ppy"]) / p["fy"], (y1 - p["ppy"]) / p["fy"])
    return bounds, (qw, qh)


def fov_deg(b):
    return (math.degrees(math.atan(b[1]) - math.atan(b[0])), math.degrees(math.atan(b[3]) - math.atan(b[2])))


# 640x360 is the 1280x720 mode downscaled, so its view shares the HD intrinsics (scaled by 0.5).
hd360 = {k: (v / 2 if k in ("fx", "fy", "ppx", "ppy", "width", "height") else v) for k, v in meta["1280x720"].items()}
vla_vga, vla_vga_in = vla_view(meta["640x480"])
vla_169, vla_169_in = vla_view(hd360)

# (bounds, label, color, linestyle, linewidth, label anchor)
rects = [
    (from_fov(89.072, 57.923), "Depth/IR 1280×720 · 848×480 · 640×360 · 480×270\n89.1° × 57.9°  (widest D435i view)", "#555555", "--", 2.0, "tl"),
    (from_fov(78.699, 63.177), "Depth/IR 640×480\n78.7° × 63.2°", "#8a8a8a", ":", 2.0, "bl"),
    (hd, "RGB 1280×720 (= 1920×1080) — widest RGB\n70.2° × 43.1°", "#1f9bd1", "-", 3.0, "above_r"),
    (from_fov(69.899, 43.135), "RGB 424×240  69.9° × 43.1°", "#2ca02c", "-.", 2.0, "below_r"),
    (vga, "RGB 640×480 — GR00T-WBC server & LeRobot head-camera\nserver defaults   55.6° × 43.1°", "#e4572e", "-", 3.5, "in_b"),
    (vla_169, "VLA sees · 16:9 ≥256 rows (e.g. 640×360@60)\n"
     f"{vla_169_in[0]}×{vla_169_in[1]} · {(vla_169_in[0] // 32) * (vla_169_in[1] // 32)} tok · "
     "{:.1f}°×{:.1f}°".format(*fov_deg(vla_169)), "#7b2cbf", (0, (6, 3)), 2.5, "tl"),
    (vla_vga, "VLA sees · current 640×480\n"
     f"{vla_vga_in[0]}×{vla_vga_in[1]} · {(vla_vga_in[0] // 32) * (vla_vga_in[1] // 32)} tok · "
     "{:.1f}°×{:.1f}°".format(*fov_deg(vla_vga)), "#ffd23f", (0, (6, 3)), 2.5, "tr"),
]

fig, ax = plt.subplots(figsize=(16, 10.5))
ax.set_facecolor("#ececec")

# Real 1280x720 RGB capture, placed exactly in its own bounds.
img = Image.open(cap_dir / "rgb_1280x720.png").convert("RGB")
ax.imshow(img, extent=(hd[0], hd[1], hd[3], hd[2]), zorder=1)

# Dim the RGB pixels that 640x480 throws away.
for x0, x1 in ((hd[0], vga[0]), (vga[1], hd[1])):
    ax.add_patch(patches.Rectangle((x0, hd[2]), x1 - x0, hd[3] - hd[2], color="#e4572e", alpha=0.28, zorder=2, lw=0))
    ax.text((x0 + x1) / 2, (hd[2] + hd[3]) / 2, "lost at\n640×480", ha="center", va="center", color="white",
            fontsize=12, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", fc="#b23a1d", ec="none", alpha=0.9))

ax.text(0, -0.58, "depth / IR only — no RGB pixels here", ha="center", va="center", fontsize=11, color="#555555", zorder=6)

for (l, r, t, b), label, color, ls, lw, anchor in rects:
    ax.add_patch(patches.Rectangle((l, t), r - l, b - t, fill=False, ec=color, ls=ls, lw=lw, zorder=5))
    box = dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.92)
    pad = 0.015
    pos = {
        "tl": (l + pad, t + pad, "left", "top"),
        "bl": (l + pad, b - pad, "left", "bottom"),
        "tr": (r - pad, t + pad, "right", "top"),
        "br": (r - pad, b - pad, "right", "bottom"),
        "in_b": ((l + r) / 2, b - pad, "center", "bottom"),
        "above_r": (r, t - pad, "right", "bottom"),
        "below_r": (r, b + pad, "right", "top"),
    }[anchor]
    ax.text(pos[0], pos[1], label, ha=pos[2], va=pos[3], fontsize=10.5, color="black", zorder=7, bbox=box)

# Optical axis
ax.plot([0], [0], marker="+", ms=18, mew=2, color="white", zorder=8)

lim_x = math.tan(math.radians(48))
lim_y = math.tan(math.radians(34))
ax.set_xlim(-lim_x, lim_x)
ax.set_ylim(lim_y, -lim_y)
ax.set_aspect("equal")
deg_x = range(-40, 41, 10)
deg_y = range(-30, 31, 10)
ax.set_xticks([math.tan(math.radians(d)) for d in deg_x], [f"{d}°" for d in deg_x])
ax.set_yticks([math.tan(math.radians(d)) for d in deg_y], [f"{d}°" for d in deg_y])
ax.set_xlabel("horizontal angle from optical axis (tangent scale)")
ax.set_ylabel("vertical angle (tangent scale)")
ax.set_title(
    "D435i (serial 254322071593) — field of view per stream profile vs. the widest view\n"
    "Photo: real 1280×720 RGB capture (2026-10-23). RGB 1280×720 and 640×480 bounds from calibrated intrinsics; "
    "other profiles centered from calibrated FOV.",
    fontsize=12,
)
fig.text(
    0.5, 0.015,
    "Depth/IR and RGB are different cameras ~15 mm apart; their small relative rotation and near-range parallax are ignored "
    "(angles at infinity). FOV depends on resolution, not FPS.\n"
    "Dashed 'VLA sees': GR00T N1.7 eval transform (short edge 256 → center crop 230/256 → short edge 256 → Qwen3-VL 32-px grid). "
    "Detail is capped at ~230 rows whatever the camera resolution.",
    ha="center", fontsize=10, color="#444444",
)
fig.tight_layout(rect=(0, 0.065, 1, 0.985))
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110)

# Self-check: VGA must sit inside HD horizontally with ~equal pixel strips, and share its vertical extent.
fx = meta["1280x720"]["fx"]
left_px = (vga[0] - hd[0]) * fx
right_px = (hd[1] - vga[1]) * fx
assert abs(left_px - 160) < 1 and abs(right_px - 160) < 1, (left_px, right_px)
assert abs(vga[2] - hd[2]) < 1e-3 and abs(vga[3] - hd[3]) < 1e-3
assert vla_vga_in == (352, 256) and vla_169_in == (448, 256), (vla_vga_in, vla_169_in)
assert abs(fov_deg(vla_169)[0] - 64.5) < 0.5 and abs(fov_deg(vla_vga)[0] - 50.7) < 0.5, (fov_deg(vla_169), fov_deg(vla_vga))
print("VLA views (deg):", [round(v, 1) for v in fov_deg(vla_vga)], [round(v, 1) for v in fov_deg(vla_169)])
print(f"saved {out}; VGA strips: left {left_px:.2f}px right {right_px:.2f}px of 1280")
print("angles (deg) HD  L/R:", [round(math.degrees(math.atan(v)), 2) for v in hd[:2]],
      "VGA L/R:", [round(math.degrees(math.atan(v)), 2) for v in vga[:2]])
