from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"yolo11s.pt")
    model.train(
        data=r"SHWD.yaml",
        epochs=50,
        batch=-1,
        imgsz=960,
        cache="ram",
        workers=0,
        name="imgsz960"
    )