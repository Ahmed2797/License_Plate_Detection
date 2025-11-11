from project.plate_detect import LicensePlateDetector

if __name__ == "__main__":
    detector = LicensePlateDetector(
        video_path='project/data/licence_plate_detect.mp4',
        model_path='project/weights/best.pt',
        conf_threshold=0.5,
        save_interval=5
    )
    detector.run()
