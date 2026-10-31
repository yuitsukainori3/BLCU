from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"yolo11s.pt")
    model.train(
        data=r"smalldata.yaml",
        epochs=100,
        batch=-1,
        imgsz=960,
        cache="ram",
        workers=1,
    )