# NAIRR-Workshops-OU

Oakland University's public-facing materials for the NAIRR (National AI Research
Resource) Workshop Series — autonomous systems track.

## Workshops

| # | Workshop | Folder | Status |
|---|----------|--------|--------|
| 4.3 | Sensor Fusion, Estimation, and Control | [`4.3/`](4.3/) | In progress |

## Repository layout

```
NAIRR-Workshops-OU/
├── 4.3/                              # workshop notebooks, slides, media
│   ├── N1_Camera_Lidar_Projection.ipynb
│   ├── N2_Kalman_From_Scratch.ipynb
│   ├── N3_LiDAR_3D_Tracking.ipynb
│   ├── N4_Path_Tracking_and_Control.ipynb
│   ├── N5_Capstone_BEV_Simulator.ipynb
│   ├── slides/                       # all decks (intro, N1–N5)
│   ├── lidar_camera_fusion.mp4
│   └── README.md
├── Dockerfile/4.3/                   # hub image (Dockerfile + requirements.txt)
└── .github/workflows/               # docker-publish-4.3.yml
```

See [`4.3/README.md`](4.3/README.md) for the workshop's structure, run order, and
setup.
