"""Capture one left-IR and one depth frame per depth profile, with intrinsics and depth->color extrinsics.

Run on the robot while nothing else holds the camera (stop the camera server first):
    .venv_camera/bin/python capture_depth_ir.py OUT_DIR
Writes ir_WxH.png (8-bit), depth_WxH.png (16-bit, raw z16), depth_profiles.json, and rgb_WxH.png +
capture_metadata.json from the same camera (sequential captures; keep the scene still).
"""

import json
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

PROFILES = [(1280, 720), (640, 480)]
WARMUP_FRAMES = 30  # let auto-exposure settle

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

dev = rs.context().query_devices()[0]
serial = dev.get_info(rs.camera_info.serial_number)
depth_sensor = dev.first_depth_sensor()
color_profile = next(
    p.as_video_stream_profile()
    for s in dev.query_sensors()
    for p in s.get_stream_profiles()
    if p.stream_type() == rs.stream.color and p.format() == rs.format.rgb8
    and p.as_video_stream_profile().width() == 1280 and p.as_video_stream_profile().height() == 720
)

meta = {
    "serial": serial,
    "firmware": dev.get_info(rs.camera_info.firmware_version),
    "usb_type": dev.get_info(rs.camera_info.usb_type_descriptor),
    "depth_scale": depth_sensor.get_depth_scale(),
    "emitter_enabled": float(depth_sensor.get_option(rs.option.emitter_enabled)),
    "profiles": {},
}

for w, h in PROFILES:
    # Lowest fps the link offers for this size keeps USB2 within bandwidth.
    fps = min(
        p.fps()
        for p in depth_sensor.get_stream_profiles()
        if p.stream_type() == rs.stream.depth and p.format() == rs.format.z16
        and p.as_video_stream_profile().width() == w and p.as_video_stream_profile().height() == h
    )
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, fps)
    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    try:
        for _ in range(WARMUP_FRAMES):
            frames = pipe.wait_for_frames(10000)
        depth, ir = frames.get_depth_frame(), frames.get_infrared_frame(1)
        cv2.imwrite(str(out / f"ir_{w}x{h}.png"), np.asanyarray(ir.get_data()))
        cv2.imwrite(str(out / f"depth_{w}x{h}.png"), np.asanyarray(depth.get_data()))
        dprof = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        i = dprof.get_intrinsics()
        try:
            ext = dprof.get_extrinsics_to(color_profile)
        except RuntimeError:  # some firmware only reports it while color streams too
            ext = None
        meta["profiles"][f"{w}x{h}"] = {
            "width": i.width, "height": i.height, "fps": fps,
            "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
            "distortion_model": str(i.model), "distortion_coeffs": list(i.coeffs),
            "fov_degrees": list(rs.rs2_fov(i)),
            "depth_to_color_rotation": list(ext.rotation) if ext else None,
            "depth_to_color_translation_m": list(ext.translation) if ext else None,
            "frame_number": depth.get_frame_number(),
        }
        print(f"{w}x{h}@{fps}: fov {meta['profiles'][f'{w}x{h}']['fov_degrees']}")
    finally:
        pipe.stop()

(out / "depth_profiles.json").write_text(json.dumps(meta, indent=2))

# RGB from the same camera, so fov_overlay.py can use OUT_DIR for both the photo and the depth/IR frames.
color_sensor = dev.first_color_sensor()
rgb = {"capture_date": time.strftime("%Y-%m-%d"), "serial": serial, "usb_type": meta["usb_type"], "profiles": {}}
for w, h in PROFILES:
    fps = min(
        p.fps()
        for p in color_sensor.get_stream_profiles()
        if p.stream_type() == rs.stream.color and p.format() == rs.format.rgb8
        and p.as_video_stream_profile().width() == w and p.as_video_stream_profile().height() == h
    )
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    try:
        for _ in range(WARMUP_FRAMES):
            frames = pipe.wait_for_frames(10000)
        color = frames.get_color_frame()
        cv2.imwrite(str(out / f"rgb_{w}x{h}.png"), cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_RGB2BGR))
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        rgb["profiles"][f"{w}x{h}"] = {
            "width": i.width, "height": i.height, "fps": fps,
            "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
            "fov_degrees": list(rs.rs2_fov(i)), "frame_number": color.get_frame_number(),
        }
        print(f"rgb {w}x{h}@{fps}: fov {rgb['profiles'][f'{w}x{h}']['fov_degrees']}")
    finally:
        pipe.stop()
(out / "capture_metadata.json").write_text(json.dumps(rgb, indent=2))
print(f"saved to {out}")
