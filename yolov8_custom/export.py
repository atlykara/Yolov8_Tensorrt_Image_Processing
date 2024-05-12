from ultralytics import YOLO

model = YOLO("yolov8n_custom.pt")
model.export(
    format="engine",
    dynamic=True, 
    batch=8, 
    workspace=4, 
    #int8=True,
    data="data_custom.yaml", 
)