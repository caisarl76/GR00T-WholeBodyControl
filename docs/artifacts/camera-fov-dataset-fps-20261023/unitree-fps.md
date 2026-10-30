# Unitree official manipulation dataset FPS

Checked 2026-10-23. All **10 sampled official datasets** declare dataset `fps: 30` and `video.fps: 30` for every listed camera stream in their pinned `meta/info.json`. This is the published dataset/video metadata rate (33.33 ms per frame), not a measurement of sensor acquisition, dropped/duplicated frames, teleoperation timing, or the robot low-level controller.

The [official organization](https://huggingface.co/unitreerobotics) includes G1 Dex1/Dex3, Brainco, whole-body Inspire/Brainco, DiverseManip, and R1 manipulation families. The public [dataset API](https://huggingface.co/api/datasets?author=unitreerobotics&limit=300) returned 205 repositories; this is a representative sample, not an exhaustive FPS claim about all repositories.

| Exact dataset ID | Dataset FPS | All listed video FPS | Video size W×H | Streams | Episodes | Pinned primary metadata |
|---|---:|---:|---|---:|---:|---|
| unitreerobotics/G1_Dex3_ToastedBread_Dataset | 30 | 30 | 640×480 | 4 | 418 | [02cb29161f5a](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset/blob/02cb29161f5adb2981a072aa35b408d82a556d07/meta/info.json) |
| unitreerobotics/G1_Dex3_BlockStacking_Dataset | 30 | 30 | 640×480 | 4 | 301 | [57faa2cf516e](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset/blob/57faa2cf516e008f96d91fe3b67ad53a74f012e6/meta/info.json) |
| unitreerobotics/G1_Dex1_MountCamera_Dataset | 30 | 30 | 640×480 | 4 | 351 | [20c864f95542](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_MountCamera_Dataset/blob/20c864f9554292194ec34030100ff48858959db6/meta/info.json) |
| unitreerobotics/G1_Dex1_DiverseManip_DualArm_128x128 | 30 | 30 | 128×128 | 2 | 525 | [7d8da3fb89b9](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_DiverseManip_DualArm_128x128/blob/7d8da3fb89b910ae53ccedd262a39128e3444447/meta/info.json) |
| unitreerobotics/G1_Brainco_GraspOreo_Dataset | 30 | 30 | 640×480 | 4 | 201 | [ad16b54daac9](https://huggingface.co/datasets/unitreerobotics/G1_Brainco_GraspOreo_Dataset/blob/ad16b54daac9b12d815934723059048d4358ca1b/meta/info.json) |
| unitreerobotics/G1_WBT_Brainco_Pickup_Pillow | 30 | 30 | 640×480 | 4 | 300 | [6849d285d713](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Brainco_Pickup_Pillow/blob/6849d285d713b57176fac8f717d073a8c3025290/meta/info.json) |
| unitreerobotics/R1_Dex1_Fold_Towel | 30 | 30 | 640×480 | 4 | 29 | [0611a1fc08ea](https://huggingface.co/datasets/unitreerobotics/R1_Dex1_Fold_Towel/blob/0611a1fc08ea3c2ad293988614c766c5b4418b60/meta/info.json) |
| unitreerobotics/G1_Dex1_Put_Pens_In_Box | 30 | 30 | 640×480 | 8 | 250 | [490ee5237d97](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Put_Pens_In_Box/blob/490ee5237d9705c1e506af2c56a512955011288a/meta/info.json) |
| unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine | 30 | 30 | 640×480 | 4 | 300 | [ff9d6b380b11](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine/blob/ff9d6b380b1191be3f90d96400f64c83d63e1a86/meta/info.json) |
| unitreerobotics/G1_Dex1_DiverseManip_SingleArm_256x256 | 30 | 30 | 256×256 | 2 | 468 | [adfe712e2ac8](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_DiverseManip_SingleArm_256x256/blob/adfe712e2ac801ca7ba18c0da79e39483975cc1f/meta/info.json) |

All sampled revisions identify LeRobot `codebase_version: v3.0`. The four-stream datasets contain left/right head and left/right wrist views. DiverseManip samples contain two head views. Put_Pens_In_Box additionally includes wrist IR and rectified head streams. Each row's pinned metadata supports its stream count, resolution, and declared FPS; RGB video fields generally use AV1, while the sampled R1 metadata declares `mp4v`.

The [official Unitree conversion code](https://github.com/unitreerobotics/unitree_lerobot/blob/main/unitree_lerobot/utils/convert_unitree_json_to_lerobot.py#L247-L257) explicitly passes `fps=30` to `LeRobotDataset.create`. Its frame loop adds indexed state/action/image samples without passing original capture timestamps ([source](https://github.com/unitreerobotics/unitree_lerobot/blob/main/unitree_lerobot/utils/convert_unitree_json_to_lerobot.py#L259-L286)). Thus the conversion setting corroborates the dataset convention, but cannot establish that a physical camera captured a fresh frame every 33.33 ms. The code link follows main as observed on the check date; dataset links above are commit-pinned.

For VLA temporal alignment with these samples, 30 Hz is the metadata reference for observations and actions. A camera capable of nominal 30 FPS can target that reference, but actual unique-frame rate, timestamps, exposure latency, and synchronization must be measured separately. This is an engineering inference from the declared dataset rate, not a claim that any 30 FPS camera matches the training distribution.

Access/method: read-only public Hugging Face API and small JSON files over HTTPS, plus official source. No authentication was needed. Sandboxed DNS failed initially; the approved network retry succeeded. No videos or parquet files were downloaded, no encoded MP4 headers were independently probed, and no sensor acquisition timestamps or teleoperation/controller logs were inspected. Consequently 'video FPS' in this note means metadata-declared encoded video FPS, not independent ffprobe verification.

