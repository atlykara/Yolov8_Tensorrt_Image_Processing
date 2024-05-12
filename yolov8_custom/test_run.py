from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

model = YOLO("yolov8n_custom.engine", task="detect" )

video_path = "fire2.MOV"
capture = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda: [])
#boyutlar
#frame_width = 360
#frame_height = 640

#videoyu isteğe göre boyutlandırma
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

while(capture.isOpened()):
    success, frame = capture.read() #read 2 değişken döndürüyor
                                    #true, false ; frames
    frame = cv2.resize(frame, (360, 640))
    if success:
        # Run YOLOv8 inference on the frame
        results = model.track(frame,  persist=True, conf=0.1, imgsz=640)
        
        #boxes = results[0].boxes.xywh.cpu()
        if results[0].boxes.id != None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            # Check if results[0].boxes.id is not None before accessing int() method
            track_ids = results[0].boxes.id.int().cpu().tolist() #if results[0].boxes.id is not None else []
            #track_ids = results[0].boxes.id.cpu().tolist()
            
            annotated_frame = results[0].plot()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]

                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
        else:
            print("No box detected in this frame.")
            cv2.imshow("YOLOv8 Tracking", frame)
       
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

capture.release()
cv2.destroyAllWindows()