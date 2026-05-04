# mpcp-onboarding-final-delib

A proof-of-concept demo showing biomechanical stroke-cycle-aware windowing for freestyle swimming analysis.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the MediaPipe pose model
```bash
wget -O pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

### 3. Download the demo video
```bash
mkdir -p data output
yt-dlp "https://www.youtube.com/shorts/xzkFh4OY4no" -f 136 -o data/freestyle-video.mp4
```

## Run

```bash
python demo.py
```

The live video will play with landmarks overlaid. When a stroke cycle boundary is detected "BOUNDARY DETECTED" will flash on screen. After the video ends a plot will be saved to `output/boundary_plot.png` showing the wrist and shoulder x positions over time with green vertical lines marking each detected boundary.
