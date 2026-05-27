import argparse
import cv2
import sys
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Professional Object & Pose Tracking using YOLOv8")
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 object detection model (e.g., yolov8n.pt)')
    parser.add_argument('--pose-model', type=str, default='yolov8n-pose.pt', help='YOLOv8 pose estimation model for body parts')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', choices=['bytetrack.yaml', 'botsort.yaml'], help='Tracking algorithm')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--no-pose', action='store_true', help='Disable body part tracking')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size (pixels)')
    parser.add_argument('--half', action='store_true', help='Use half precision (if supported)')
    parser.add_argument('--device', type=str, default='', help='Device to run on, e.g. 0 or cpu (default: auto)')
    parser.add_argument('--fps', type=int, default=60, help='Target FPS for display loop (best-effort)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize the YOLOv8 models
    try:
        print(f"Loading object detection model: {args.model} (for cars, electronics, etc.)")
        obj_model = YOLO(args.model)
        
        if not args.no_pose:
            print(f"Loading pose estimation model: {args.pose_model} (for humans and body parts)")
            pose_model = YOLO(args.pose_model)
        else:
            pose_model = None
            
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)
    
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        sys.exit(1)
        
    print(f"Successfully opened video source: {args.source}")
    print("Starting professional object & pose tracking... Press 'q' to quit.")

    # All COCO classes except 0 (person), so we don't draw duplicate boxes for people
    # We leave people detection and body parts to the pose model
    obj_classes = [i for i in range(1, 80)] if pose_model is not None else None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break
            
        annotated_frame = frame.copy()
        
        try:
            # 1. Track general objects (Cars, Electronics, etc.)
            obj_results = obj_model.track(
                frame,
                imgsz=args.imgsz,
                half=args.half,
                device=args.device,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                iou=args.iou,
                classes=obj_classes,
                verbose=False,
            )

            if obj_results and len(obj_results) > 0:
                # Plot object detection results on the frame
                annotated_frame = obj_results[0].plot(img=annotated_frame)

            # 2. Track humans and body parts (Pose Estimation)
            if pose_model is not None:
                pose_results = pose_model.track(
                    frame,
                    imgsz=args.imgsz,
                    half=args.half,
                    device=args.device,
                    persist=True,
                    tracker=args.tracker,
                    conf=args.conf,
                    iou=args.iou,
                    verbose=False,
                )

                if pose_results and len(pose_results) > 0:
                    # Plot pose estimation results on the same frame
                    annotated_frame = pose_results[0].plot(img=annotated_frame)

        except Exception as e:
            print(f"Error during tracking: {e}")
            break
            
        # Display the frame
        cv2.imshow('YOLOv8 Advanced Object & Pose Tracking', annotated_frame)

        # Press 'q' to exit
        # Target FPS (best-effort). If inference is slower than target, this won’t increase FPS.
        delay_ms = max(int(1000 / max(args.fps, 1)), 1)
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            print("Quitting...")
            break
            
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
