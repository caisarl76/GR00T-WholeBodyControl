# D435i frame comparison and humanoid dataset FPS

Checked 2026-10-23. This follows the [camera audit](../camera-review-20261019/review.md) and [VLA FPS research](../vla-camera-fps-20261022/review.md).

## Actual camera frames

![Actual D435i frames with calibrated crop overlay](fov_comparison.png)

Unannotated captures: [640×480 RGB](rgb_640x480.png), [1280×720 RGB](rgb_1280x720.png). Both were acquired from D435I serial `254322071593` on `unitree@192.168.0.223`, with RGB8 at 15 FPS. The camera was fixed in place; the two captures were 4.1 seconds apart, so they are not simultaneous. The 40th frameset after startup was selected from each profile. No sensor options were manually changed, and camera-server configuration and robot-control settings were unchanged. The camera pipelines were stopped after acquisition, and a subsequent device-owner check found no process holding the video devices.

| Profile | Calibrated horizontal FOV | Calibrated vertical FOV |
|---|---:|---:|
| 1280×720 RGB | 70.238° | 43.135° |
| 640×480 RGB | 55.621° | 43.135° |

The VGA image covers approximately columns **160 through 1120** of the wide image: the middle **960×720** rectangle, scaled to **640×480**. Each removed side strip is **160 columns**, or **12.5%** of the wide image width. Together they account for **25% of image width**, rather than 25% of the angular FOV. Vertical coverage is preserved.

The crop was calculated from the two active stream profiles' intrinsics:

```text
u_wide = (u_vga - cx_vga) * fx_wide / fx_vga + cx_wide
v_wide = (v_vga - cy_vga) * fy_wide / fy_vga + cy_wide
```

Both profiles report zero distortion coefficients. Mapping the VGA bounds gives `(160.00005, 0.00002)` to `(1119.99998, 719.999996)` in wide-image coordinates. The bottom-right panel is a deterministic center crop of the wide capture, resized for comparison; the bottom-left panel is the independently captured native VGA image. Neither is an AI-generated image. Full intrinsics, sensor frame timestamps, active profiles and USB descriptor are in [capture_metadata.json](capture_metadata.json). The descriptor is still `2.1` for this capture.

## Dataset rates

### UnitreeRobotics

The official `unitreerobotics` Hub organization contains multiple datasets. All **10 sampled repositories** declare **30 FPS** at the dataset level and for every listed camera video. Samples span G1 Dex1/Dex3, Brainco, whole-body teleoperation and R1. Eight sampled repositories use 640×480 images; the two DiverseManip variants use 128×128 and 256×256. This is a sampled metadata finding, not a claim that every dataset or physical camera in the organization uses the same native acquisition rate. The exact repository names, revisions, metadata URLs, resolutions, and converter details are in [unitree-fps.md](unitree-fps.md). Example: [commit-pinned ToastedBread metadata](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset/blob/02cb29161f5adb2981a072aa35b408d82a556d07/meta/info.json).

### Humanoid Everyday

The official release is `USC-PSI-Lab/humanoid-everyday`, from the Humanoid Everyday authors. Its metadata declares dataset **`fps: 30`**, RGB **`video.fps: 30.0`**, H.264 video, and **640×480** egocentric RGB. The authors also describe 30 Hz recording. These identify the released/stored rate; native motor/sensor rates and actual frame uniqueness were not measured here. [Official metadata](https://huggingface.co/datasets/USC-PSI-Lab/humanoid-everyday/raw/main/meta/info.json), [authors' project](https://humanoideveryday.github.io/).

The paper supplies a useful training distinction: its OpenVLA experiment downsampled data from **30 Hz to 2 Hz**, alleviating poor motion generation while producing less-smooth behavior. That is a reported model-specific training configuration, not a 2 FPS camera-recording requirement or a universal minimum. [Paper v3, §V-A](https://arxiv.org/html/2510.08807v3#S5.SS1). Source details and conversion checks are in [humanoid-everyday-fps.md](humanoid-everyday-fps.md).

The authors' [D435 calibration](https://github.com/physical-superintelligence-lab/Humanoid-Everyday#camera-intrinsics-realsense-d435) also explicitly uses 640×480 RGB for G1 and H1. Its documented RGB focal lengths are close to this camera's VGA calibration; changing resolution/FOV when adapting a policy should therefore be treated as an observation change, rather than assuming a training benefit from a wider image alone.

No model was trained and no dataset media was audited frame by frame in this task. The camera comparison uses actual fresh captures; dataset FPS findings use official metadata, author documentation and conversion code.
