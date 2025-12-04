from project.plate_detect import LicensePlateDetector

if __name__ == "__main__":
    detector = LicensePlateDetector(
        video_path='project/data/LicensePlateDetectionTest_1080p.mp4',
        model_path='project/weights/best_licence_patedt15.pt',
        conf_threshold=0.5,
        save_interval=5
    )
    detector.run()


