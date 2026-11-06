from ultralytics import YOLO

image_path = "test.png"
model = YOLO("best.pt")

results = model.predict(source=image_path, imgsz=640)

if results:
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("nothing in the image.")
    else:
        print(result.boxes.xyxy)

    result.save(filename="test_annotated.jpg")
    print("done")
else:
    print("nothing found")
