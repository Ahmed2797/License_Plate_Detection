from project.utils import paddle_ocr_plate, save_json
from datetime import datetime
import cv2
from ultralytics import YOLO


class LicensePlateDetector:
    def __init__(self, video_path, model_path, conf_threshold=0.5, save_interval=5):
        """
        Initialize the License Plate Detector
        """
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.save_interval = save_interval  # in seconds
        self.license_plate_set = set()
        self.start_time = datetime.now()
        self.frame_count = 0

    def process_frame(self, frame):
        """
        Detect license plates in a single frame and annotate it
        """
        results = self.model.predict(frame)
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for (x1, y1, x2, y2), conf in zip(boxes, confs):
                if conf < self.conf_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"License_plate: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
                
                label = paddle_ocr_plate(frame, x1, y1, x2, y2, self.conf_threshold)
                if label:
                    self.license_plate_set.add(label)
                    text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + text_size[0], y1 - text_size[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [0, 0, 255], thickness=1, lineType=cv2.LINE_AA)
        
        return frame

    def save_if_needed(self):
        """
        Save license plates to JSON every `save_interval` seconds
        """
        current_time = datetime.now()
        if (current_time - self.start_time).total_seconds() >= self.save_interval:
            save_json(self.license_plate_set, self.start_time, current_time)
            self.start_time = current_time
            self.license_plate_set.clear()  # Uncomment if you want to clear after saving

    def run(self):
        """
        Run the license plate detection on video
        """
        cap = cv2.VideoCapture(self.video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            print(f"Frame count: {self.frame_count}")

            annotated_frame = self.process_frame(frame)
            self.save_if_needed()

            cv2.imshow("Video", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# if __name__ == "__main__":
#     detector = LicensePlateDetector(
#         video_path='project/data/licence_plate_detect.mp4',
#         model_path='project/weights/best.pt',
#         conf_threshold=0.5,
#         save_interval=5
#     )
#     detector.run()
