from ultralytics import YOLO

model = YOLO(r"F:\Competition\ultralytics-8.3.163\ultralytics-8.3.163\runs\detect\train2\weights\best.pt")
model.predict(
    source= r"F:\Competition\0make_dataset\images",
    save = True,
    show = False,
    save_txt = True,
)