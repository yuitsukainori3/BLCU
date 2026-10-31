from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"yolo11s.pt")
    model.train(
        data=r"data.yaml",
        epochs=30,
        batch=-1,
        imgsz=640,
        cache=False,
        workers=0,
    )