# Humanoid Everyday camera and dataset rate

Checked 2026-10-23 against the authors' project, paper, release metadata, and conversion code. Only small text/metadata sources were retrieved; no dataset media was downloaded.

**Answer: the official Humanoid Everyday dataset uses 30 FPS egocentric RGB, at 640 × 480.** This is the USC/Toyota Research Institute dataset by Zhao, Jing et al., collected with Unitree G1 and H1; it is distinct from UnitreeRobotics demonstration repositories. The [official project](https://humanoideveryday.github.io/) identifies the authors, institutions, robots, and 30 Hz collection. The [official release metadata](https://huggingface.co/datasets/USC-PSI-Lab/humanoid-everyday/raw/main/meta/info.json) independently declares both `fps: 30` and `features.observation.images.egocentric.video_info.video.fps: 30.0`.

| Item | Evidence and interpretation |
| --- | --- |
| Recorded multimodal dataset | 30 Hz for sensor/action records, including RGB, depth, LiDAR, tactile, IMU and joint poses/actions. The paper states this in Section I, PDF page 2; the dataset card explicitly lists the modalities at 30 Hz. |
| Released RGB video | Metadata specifies H.264, 640 × 480, 30 FPS; the recorded RGB is the egocentric camera. |
| Legacy/raw representation | README describes 640 × 480 PNG RGB frames, uint16 depth and PCD LiDAR, with 30 Hz recording. |
| LeRobot conversion | `scripts/he2lerobot.py` sets `FPS = 30` (line 21), assigns timestamps using frame index / FPS (line 256), encodes RGB with `fps=FPS` (line 291), and writes 30 FPS metadata (lines 405, 447). |
| Actual motor/control loop or sensor-native rates | These are separate from the recorded 30 Hz dataset. The paper describes separate robot-control, IK and I/O processes; it does not establish that the underlying motor loop or every physical sensor itself runs at 30 Hz. No different native RGB/depth/IMU rate was verified here. |
| Training downsampling | Section V-A, PDF page 6, describes training on 30 Hz data and specifically downsampling OpenVLA to 2 Hz. That experiment does not change the dataset's acquisition/release rate. |

Sources: [paper v3, Section I](https://arxiv.org/html/2510.08807v3#S1), [paper Section III-B](https://arxiv.org/html/2510.08807v3#S3.SS2), [paper Section V-A](https://arxiv.org/html/2510.08807v3#S5.SS1), [authors' PDF](https://humanoideveryday.github.io/file/humanoid_everyday_arxiv.pdf), [official dataset card](https://huggingface.co/datasets/USC-PSI-Lab/humanoid-everyday), [official README](https://github.com/physical-superintelligence-lab/Humanoid-Everyday#overview), [conversion code](https://github.com/physical-superintelligence-lab/Humanoid-Everyday/blob/master/scripts/he2lerobot.py).

The strongest evidence for VLA loading is the released `meta/info.json`: `codebase_version` is `2.1`, `robot_type` is `mixed`, and the current release reports 8,949 episodes, 3,436,171 frames, and 246 tasks. Those release counts differ from the paper's headline collection counts; the rate is consistent. The card's older example uses `USC-GVL/humanoid-everyday`, while the inspected repository is `USC-PSI-Lab/humanoid-everyday`.

Confidence is high for the documented capture/recording rate and declared encoded rate. This was not a per-video ffprobe audit or an empirical unique-frame/exposure timestamp measurement. The converter's synthetic timestamps alone cannot prove exact physical sensor timing or absence of duplicates/dropped frames.
