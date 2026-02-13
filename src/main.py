import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from draw import draw_landmarks_on_image

MODEL_PATH = "data/hand_landmarker.task"


class HandTracker:
    def __init__(self, model_path):
        self.detection_result = None
        self.detector = self._init_detector(model_path)

    def _result_callback(self, result, output_image: mp.Image, timestamp_ms: int):
        self.detection_result = result

    def _init_detector(self, model_path):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=model_path,
                delegate=mp.tasks.BaseOptions.Delegate.GPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self._result_callback,
            num_hands=2,
        )
        return vision.HandLandmarker.create_from_options(options)

    def run(self):
        cap = cv2.VideoCapture(0)
        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue
                mp_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=mp_image)
                frame_count += 1
                self.detector.detect_async(mp_image, frame_count)
                if self.detection_result:
                    frame = draw_landmarks_on_image(frame, self.detection_result, cv2)
                cv2.imshow("Hand Gesture Recognition", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    tracker = HandTracker(MODEL_PATH)
    tracker.run()


if __name__ == "__main__":
    main()
