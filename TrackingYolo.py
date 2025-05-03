from ultralytics import YOLO
import numpy as np, cv2, torch
from sort import Sort
from sort import convert_x_to_bbox   
import supervision as sv
from trackers import SORTTracker

#torch.backends.cudnn.enabled = False

VIDEO   = "KALMAN_FILTER\gatos_yolo.mp4"
model   = YOLO("yolov8m.pt")
tracker = SORTTracker()
#tracker = Sort(max_age=30, min_hits=15, iou_threshold=0.3)
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

"""
cap = cv2.VideoCapture(VIDEO)
while True:
    ok, frame = cap.read()
    if not ok:
        print("Fin del video"); break

    # detección
    res  = model(frame, conf=0.65, verbose=False)[0]
    dets = res.boxes.data.cpu().numpy()[:, :5]if res.boxes else np.empty((0, 5))

    # tracking
    for x1, y1, x2, y2, _, tid in tracker.update(dets).astype(int):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'ID {tid}', (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 2)

    # pred Kalman (caja amarilla)
    for trk in tracker.trackers:
        px1, py1, px2, py2 = convert_x_to_bbox(trk.kf.x)[0].astype(int)

        cv2.rectangle(frame, (px1, py1), (px2, py2),
                    (0, 255, 255), 1, cv2.LINE_4)

        cv2.putText(frame, f'ID {trk.id + 1} (pred)',
                    (px1, py1 - 6),               
                    cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (0, 255, 255), 1)


    cv2.imshow("YOLOv8 + Kalman SORT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()
"""