# Scene Change Detection Using Classic Image Processing (Python)

## Overview
This project detects the **scene change point** in a video containing **exactly two scenes**. The approach uses **classic image processing** (not OpenCV) and relies on **histogram comparison** between consecutive frames.

---

## 1. Python Packages

| Package | Purpose |
|---------|---------|
| mediapy | Read video frames efficiently without OpenCV. Requires `ffmpeg` on macOS. |
| numpy   | Compute histograms, cumulative distributions, and distances. |
| matplotlib (optional) | Visualize frames, histograms, and distance curves. |
| scikit-image (optional) | Convert frames to grayscale reliably. |

**Installation:**
```bash
pip install mediapy numpy matplotlib scikit-image
brew install ffmpeg
```

---

## 2. Processing Pipeline (Per Frame)

### 2.1 Convert to Grayscale
Convert RGB frames to grayscale for simplicity:
```python
from skimage.color import rgb2gray
gray = rgb2gray(frame)
```

### 2.2 Compute Normalized Histogram
```python
import numpy as np
hist, _ = np.histogram(gray, bins=256, range=(0,1))
hist = hist / hist.sum()  # normalize to sum = 1
```
- Normalization ensures comparison is independent of image size.

### 2.3 Convert Histogram to Cumulative Histogram (CDF)
```python
cdf = np.cumsum(hist)
```
- Smooths noise, captures global intensity distribution.
- Makes L∞ comparison highly effective.

---

## 3. Distance Between Frames

For two consecutive CDFs `cdf1` and `cdf2`, compute **L∞ norm** (Kolmogorov-Smirnov distance):
```python
distance = np.max(np.abs(cdf1 - cdf2))
```
- Very sensitive to scene changes.
- Robust to small pixel fluctuations.

**Alternative:**
- L¹ norm works well for raw histograms.
- Bhattacharyya distance is for probability distributions (not cumulative).

---

## 4. Scene Change Detection

1. Compute CDF for each frame.
2. Compute L∞ distance between consecutive CDFs.
3. Scene change occurs at the frame with **maximum distance**:
```python
scene_change_frame = np.argmax(distance_array)
```

---

## 5. Summary of Steps

**Per Frame:**
1. Load frame
2. Convert to grayscale
3. Compute normalized histogram
4. Compute cumulative histogram (CDF)

**Per Consecutive Pair:**
5. Compute L∞ distance between CDFs
6. Select frame with maximum distance → **scene change**

This pipeline works robustly on videos with two scenes, even if there are small lighting variations or noise.

---

## 6. Optional Enhancements
- Visualize distance curve with matplotlib to confirm detection.
- Precompute histograms for efficiency in long videos.
- Extend to multiple scene changes by thresholding peaks.

---

# End of Document
