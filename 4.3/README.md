# Workshop 4.3 — Sensor Fusion, Estimation, and Control

Workshop 4.2 gave you separate, per-frame **detections**. Workshop 4.3 turns noisy
sensor streams into one confident **estimate** of the world — and then *acts* on it.
We build a self-driving perception-to-control stack one notebook at a time, ending
on an interactive closed-loop simulator.

```
Sense  →  Fuse  →  Estimate  →  Track  →  Act
 (4.2)     N1        N2          N3      N4 · N5
```

## Structure: three presentations, five notebooks

The workshop is delivered as **three paired presentations**:

| Presentation | Notebook(s) | What you build |
|---|---|---|
| **1** | **N1** | Pinhole camera model; project LiDAR ↔ image; back-project the road to a bird's-eye view (BEV); SegFormer road segmentation. **Downloads the KITTI drive the rest of the workshop reuses.** |
| **2** | **N2**, **N3** | N2: the Kalman filter from first principles (predict/update, gain, Q/R tuning, coasting). N3: LiDAR 3D detection **and** multi-object tracking — ground removal, clustering, 3D boxes, Hungarian association, track lifecycle. |
| **3** | **N4**, **N5** | N4: path-tracking control — kinematic bicycle model, pure-pursuit and Stanley controllers. N5: the capstone — a closed-loop BEV simulator with a time-to-collision safety layer and a live tune-the-knobs dashboard. |

| # | Notebook | Stage |
|---|----------|-------|
| **N1** | [`N1_Camera_Lidar_Projection.ipynb`](N1_Camera_Lidar_Projection.ipynb) | Sense / Fuse |
| **N2** | [`N2_Kalman_From_Scratch.ipynb`](N2_Kalman_From_Scratch.ipynb) | Estimate |
| **N3** | [`N3_LiDAR_3D_Tracking.ipynb`](N3_LiDAR_3D_Tracking.ipynb) | Track |
| **N4** | [`N4_Path_Tracking_and_Control.ipynb`](N4_Path_Tracking_and_Control.ipynb) | Act |
| **N5** | [`N5_Capstone_BEV_Simulator.ipynb`](N5_Capstone_BEV_Simulator.ipynb) | Act / capstone |

## Slides

All decks live in [`slides/`](slides/):

| Deck | Covers |
|---|---|
| `P0_Workshop_Intro.pptx` | Workshop intro |
| `P1_N1_Camera_Lidar_Projection.pptx` | Presentation 1 — N1 |
| `P2_N2_Kalman.pptx` | Presentation 2 — N2 |
| `P2_N3_LiDAR_Object_Detection.pptx` | Presentation 2 — N3 |
| `P3_N4-N5_Acting_on_the_World.pptx` | Presentation 3 — N4 + N5 |

## Run order and the KITTI data dependency

Run the notebooks **in order, N1 → N5**. The key reason is data:

- **N1 is the sole downloader** of the KITTI raw drive `2011_09_26_drive_0005`
  (calibration + synced camera / LiDAR / OxTS data) into `kitti_data/`.
- **N3 and N5 reuse that same drive.** N5 falls back to a synthetic ego path if
  KITTI is absent, so it always runs.
- **N2 and N4 are self-contained** (synthetic data) and can be run anytime.

## Dependencies

```bash
pip install numpy scipy matplotlib ipywidgets opencv-python-headless pykitti \
            scikit-learn transformers imageio[ffmpeg] plotly
```

> **Widgets:** N5 uses `ipywidgets`. Run the install cell once, then **restart the
> kernel** so the widget front-end loads — otherwise the live dashboard can render
> blank.

## Running on the JupyterHub

The hub image is built from [`../Dockerfile/4.3/`](../Dockerfile/4.3/) on top of
`quay.io/jupyter/pytorch-notebook` (CUDA torch is already included — do **not**
`pip install torch`, which would replace the GPU build with a CPU one). The build
workflow is [`docker-publish-4.3.yml`](../.github/workflows/docker-publish-4.3.yml).

To test locally against the same environment:

```bash
docker build -t nairr-4.3 ../Dockerfile/4.3
docker run --rm -p 8888:8888 -v "$PWD":/home/jovyan/work nairr-4.3
```
