from ultralytics import YOLO
import numpy as np, cv2, torch
from sort import Sort
from sort import convert_x_to_bbox   
import supervision as sv
from trackers import SORTTracker


VIDEO   = "KALMAN_FILTER\gatos_yolo.mp4"
model   = YOLO("yolov8m.pt")
tracker = SORTTracker()
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)


def callback(frame, _):
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update(detections)
    return annotator.annotate(frame, detections, labels=detections.tracker_id)

sv.process_video(
    source_path=VIDEO,
    target_path="output.mp4",
    callback=callback,
)

