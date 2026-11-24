# Spider-Man AR Frame 🕷️

A real-time augmented reality webcam application built with Python, OpenCV and MediaPipe.

Hold up both hands and use your **four fingertips** (left thumb, left index, right thumb, right index) to frame a quadrilateral portal — inside it, a Spider-Man-themed New York cityscape appears with a Spider-Man overlay that follows your face.

---

## Project Structure

```
SpiderMan_AR_Frame/
├── main.py                  # Entry point
├── requirements.txt         # Pinned dependencies
├── README.md                # This file
├── assets/
│   ├── cityscape.jpg        # Spider-Man NYC background  ← add your own
│   └── spiderman.png        # Transparent Spider-Man PNG ← add your own
└── modules/
    ├── __init__.py          # Package init
    ├── hand_tracking.py     # MediaPipe Hands wrapper
    ├── face_tracking.py     # MediaPipe Face Detection wrapper
    └── portal_renderer.py   # AR portal compositor
```

---

## Quick Start

### 1. Create & activate a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your assets *(optional — fallback graphics are built-in)*

| File | Description |
|------|-------------|
| `assets/cityscape.jpg` | Any Spider-Man NYC skyline JPEG |
| `assets/spiderman.png` | A transparent-background Spider-Man PNG |

If the files are absent the app generates placeholder artwork automatically.

### 4. Run the webcam test (Stage 1)

```bash
python main.py
```

Press **Q** or **Escape** to quit.

---

## Dependency Compatibility

| Package | Version | Why pinned |
|---------|---------|-----------|
| `opencv-python` | 4.9.0.80 | Stable with Python 3.10; provides `cv2.warpPerspective` |
| `mediapipe` | 0.10.9 | Last version with the **legacy Solutions API** (`mediapipe.solutions.*`) |
| `numpy` | 1.26.4 | Last 1.x release; fully compatible with mediapipe 0.10.x |

> ⚠️ Do **not** upgrade MediaPipe beyond 0.10.x — the `mediapipe.solutions` namespace was removed in 0.11+.

---

## Development Stages

| Stage | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Project setup & webcam test |
| 2 | ⏳ Pending | Hand tracking + corner detection |
| 3 | ⏳ Pending | Face detection + AR portal rendering |
| 4 | ⏳ Pending | Polish, gesture smoothing, final demo |

---

## Controls

| Key | Action |
|-----|--------|
| `Q` / `Esc` | Quit |
| `D` | Toggle debug landmarks *(Stage 2+)* |

---

## Troubleshooting

**Wrong camera opens** → Change `WEBCAM_INDEX = 0` to `1` or `2` in `main.py`.

**MediaPipe import error** → Ensure you installed exactly `mediapipe==0.10.9`.

**Low FPS** → Lower `FRAME_WIDTH`/`FRAME_HEIGHT` in `main.py` to `640 × 480`.
