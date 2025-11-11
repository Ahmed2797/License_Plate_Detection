from project.utils import paddle_ocr_plate, save_json
from datetime import datetime
import cv2
from ultralytics import YOLO

model = YOLO('project/weights/best.pt')
cap = cv2.VideoCapture('project/data/licence_plate_detect.mp4')

startTime = datetime.now()
license_plate = set()
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    print(f"Frame_count: {count}")
    results = model.predict(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # [[x1, y1, x2, y2], ...]
        confs = result.boxes.conf.cpu().numpy()  # [conf1, conf2, ...]

        for (x1, y1, x2, y2), conf in zip(boxes, confs):
            if conf < 0.5:  # optional threshold
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, f"License_plate: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

            label = paddle_ocr_plate(frame, x1, y1, x2, y2,0.5)
            print("label------",label)
            if label:
                license_plate.add(label)
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5,
                            [0, 0,255 ], thickness=1, lineType=cv2.LINE_AA)

    # Save every 5 seconds
    currentTime = datetime.now()
    if (currentTime - startTime).total_seconds() >= 5:
        save_json(license_plate, startTime, currentTime)
        startTime = currentTime
        #license_plate.clear()

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
