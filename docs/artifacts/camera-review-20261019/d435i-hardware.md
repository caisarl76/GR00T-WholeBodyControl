# D435i hardware limits — official-source review

Reviewed 2026-10-19. Documentation evidence only; this note does not establish the connected camera's negotiated profiles or measured throughput. Datasheet references below use revision 017, September 2023, with printed page numbers.

The D435i product specification gives **depth 87°×58°** and **RGB 69°×42°** (horizontal×vertical). Depth uses global shutters; RGB uses a rolling shutter. These are different cameras with different fields of view. [Official D435i specifications](https://www.realsenseai.com/products/depth-camera-d435i/)

RGB **640×480 and 320×240 crop horizontal coverage to approximately 55°**, according to RealSense Customer Support. Therefore increasing a VGA image's display size cannot recover the scene outside that crop. [Official engineering reply, issue 7170](https://github.com/realsenseai/librealsense/issues/7170)

Depth VGA: **75°×62°**; native stereo-imager optics: **90°×64°**, each axis ±1°. Output FOV depends on aspect ratio and stereo overlap. [Datasheet Tables 3-13/3-49, pp.43/71; §4.3, p.88](https://www.realsenseai.com/wp-content/uploads/2023/10/Intel-RealSense-D400-Series-Datasheet-September-2023.pdf#page=43)

**Standard USB 3.1 Gen 1 profiles.** Formats are datasheet transport names. [Table 4-2, p.81](https://www.realsenseai.com/wp-content/uploads/2023/10/Intel-RealSense-D400-Series-Datasheet-September-2023.pdf#page=81)

| Stream/format | Resolution(s) | FPS |
|---|---|---|
| Depth Z16 / IR Y8 | 1280×720 | 6,15,30 |
| Depth Z16 / IR Y8 | 848×480; 640×480; 640×360; 480×270; 424×240 | 6,15,30,60,90 |
| RGB YUY2 | 1920×1080; 1280×720 | 6,15,30 |
| RGB YUY2 | 960×540; 848×480; 640×480; 640×360; 424×240 | 6,15,30,60 |
| RGB YUY2 | 320×240; 320×180 | 6,30,60 |

USB3 supports simultaneous depth/RGB combinations on a dedicated port; hardware synchronization requires equal FPS. Shared-hub bandwidth matters. [Table 4-5, p.84, notes 1–3](https://www.realsenseai.com/wp-content/uploads/2023/10/Intel-RealSense-D400-Series-Datasheet-September-2023.pdf#page=84)

**USB 2.0 individual profiles.** [Table 4-7, pp.85–86](https://www.realsenseai.com/wp-content/uploads/2023/10/Intel-RealSense-D400-Series-Datasheet-September-2023.pdf#page=85)

| Stream/format | Resolution | FPS |
|---|---|---|
| Depth Z16 | 1280×720 | 6 |
| Depth Z16 | 848×480 | 6,10 |
| Depth Z16 | 640×480 | 6,15,30 |
| Depth Z16 | 640×360 | 30 |
| Depth Z16 | 480×270 | 6,15,30,60 |
| RGB YUY2 | 1280×720 | 6,15 |
| RGB YUY2 | 640×480 | 6,15,30 |
| RGB YUY2 | 424×240 | 6,15,30,60 |

USB2 simultaneous examples: depth+IR 640×480@15 with RGB640×480@30; depth+IR480×270@60 with RGB424×240@30. [Table 4-10, p.87](https://www.realsenseai.com/wp-content/uploads/2023/10/Intel-RealSense-D400-Series-Datasheet-September-2023.pdf#page=87)

These individual USB2 profiles **do not establish that every pair works simultaneously**. In particular, treat depth+RGB both VGA@30 and a depth+RGB60 pair as unverified until the actual device resolves and streams that combination. This is an interpretation of the distinction between the individual and simultaneous tables, not a claim that those pairs are universally impossible. RealSense also documents burst-readout and bandwidth constraints on USB2. [Official USB2 integration article](https://www.realsenseai.com/news-insights/usb2-support-for-intel-realsense-technology/)

**Special high-speed depth mode is separate from normal image capture.** The official whitepaper describes **848×100 at 300 FPS**, cropping a thin horizontal strip from the vertical center, and a **256×144 at 300 FPS** mode introduced for self-calibration. It recommends avoiding simultaneous RGB in the high-speed configuration. RealSense support explicitly reports D435i operation at 848×100@300. These modes do not justify promising normal-size RGB-D capture at 300 FPS. [High-speed whitepaper §1, §2.1–2.2](https://dev.realsenseai.com/docs/high-speed-capture-mode-of-intel-realsense-depth-camera-d435/), [official D435i support confirmation](https://support.realsenseai.com/hc/en-us/community/posts/360043522273-Intel-publish-white-paper-on-high-speed-300-FPS-depth-capture-for-D435?sort_by=votes)

**Alignment can discard the wider depth image.** `align` to color projects depth into the color camera's image bounds; the implementation skips projected points outside those bounds. Keep the original depth frame if wide depth coverage is required. Aligning color to depth cannot create color observations outside the RGB camera's optical view. This last statement follows from the projection operation rather than assuming alignment adds new observations. [librealsense align.cpp, `align_images` and `align_z_to_other`](https://github.com/realsenseai/librealsense/blob/master/src/proc/align.cpp), [RealSense D435i alignment explanation](https://support.realsenseai.com/hc/en-us/community/posts/39239529290259-Depth-image-alignment-on-the-D435i-camera)

**Requested FPS is not measured FPS.** RGB auto-exposure priority permits exposure-driven frame-rate reduction. Keeping auto exposure enabled while disabling auto-exposure priority is the documented way to request constant sensor cadence; it does not cure USB/controller losses or slow processing. [Official librealsense engineering discussion, issue 1957](https://github.com/realsenseai/librealsense/issues/1957)

For profile selection, use actual stream intrinsics and `rs2_fov` to check coverage, then verify unique frame numbers/timestamps and sustained delivery. The SDK documents access to intrinsics, FOV, frame timestamps and frame numbers. [librealsense API How-To](https://github.com/realsenseai/librealsense/wiki/API-How-To)

**Review implications:** On a confirmed USB3 link, RGB960×540@60 with depth848×480@60 is a candidate for increased cadence while avoiding the known RGB VGA crop; the exact FOV and stable combined throughput still require device verification. On USB2, choose from the separate USB2 table and verify the complete combination. The common product-page “90 FPS depth” and “30 FPS RGB” labels alone do not describe all resolution-dependent modes. [USB3 profile/simultaneous tables, pp.81/84](https://www.realsenseai.com/wp-content/uploads/2023/10/Intel-RealSense-D400-Series-Datasheet-September-2023.pdf#page=81), [D435i product specifications](https://www.realsenseai.com/products/depth-camera-d435i/)
