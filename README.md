[README.md](https://github.com/user-attachments/files/28295004/README.md)
# Real-Time Object Detection and Tracking

This project implements robust **Object Detection and Tracking** using state-of-the-art YOLOv8 and advanced tracking algorithms (ByteTrack/BoT-SORT).

## Features
- **Real-time processing**: Supports both webcam input and video files seamlessly via OpenCV.
- **State-of-the-art detection**: Uses YOLOv8 for high-speed, accurate object detection (Pre-trained on COCO dataset).
- **Robust tracking**: Integrates ByteTrack and BoT-SORT algorithms to maintain object identities (Track IDs) across frames.
- **Customizable**: Command-line interface allows fine-tuning confidence thresholds, models, and tracking algorithms.

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the tracking script using your webcam (default):
```bash
python tracker.py
```
*(Press `q` to quit the video stream window)*

### Advanced Usage

You can customize the execution using command-line arguments:

```bash
# Use a specific video file instead of webcam
python tracker.py --source path/to/video.mp4

# Use a larger, more accurate model (will automatically download yolov8s.pt)
python tracker.py --model yolov8s.pt

# Change the tracking algorithm to BoT-SORT
python tracker.py --tracker botsort.yaml

# Adjust confidence threshold to 50%
python tracker.py --conf 0.5
```
