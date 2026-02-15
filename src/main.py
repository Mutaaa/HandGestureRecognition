import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

from draw import draw_landmarks_on_image
from overlay import overlay_thumb_image, overlay_dab_image


MODEL_PATH = "data/hand_landmarker.task"
THUMB_IMAGE_PATH = "img/cat_thumb.png"
DAB_IMAGE_PATH = "img/dab.png"

class HandTracker:
    def __init__(self, model_path):
        self.detection_result = None
        self.detector = self._init_detector(model_path)
        self.gesture = None
        if os.path.exists(THUMB_IMAGE_PATH):
            self.thumb_img = cv2.imread(THUMB_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        else:
            self.thumb_img = None
        if os.path.exists(DAB_IMAGE_PATH):
            self.dab_img = cv2.imread(DAB_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        else:
            self.dab_img = None

    def _result_callback(self, result, output_image: mp.Image, timestamp_ms: int):
        self.detection_result = result
        if result.hand_landmarks:
            if len(result.hand_landmarks) == 2:
                self.gesture = self.detect_dab(result.hand_landmarks)
            else:
                self.gesture = self.detect_gesture(result.hand_landmarks[0])
        else:
            self.gesture = None

    def detect_gesture(self, landmarks):
        # Simple thumbs up detection: thumb tip above other finger tips
        # Indexes: 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        if (thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and
            thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y):
            return "Thumbs Up"
        return "Unknown"

    def detect_dab(self, hands_landmarks):
        # Use wrist landmarks (0) for both hands
        hand1 = hands_landmarks[0][0]
        hand2 = hands_landmarks[1][0]
        y_diff = abs(hand1.y - hand2.y)
        x_diff = abs(hand1.x - hand2.x)
        if y_diff > 0.3 and x_diff > 0.3:
            # If hand1 is above hand2
            if hand1.y < hand2.y and hand1.x > hand2.x:
                return "Dab Left"
            # If hand2 is above hand1
            elif hand2.y < hand1.y and hand2.x > hand1.x:
                return "Dab Right"
        return "Unknown"

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
                    if self.gesture == "Thumbs Up":
                        # Clone effect: tile thumbs up images as background, then draw landmarks on top
                        bg = overlay_thumb_image(frame, self.thumb_img)
                        frame = draw_landmarks_on_image(bg, self.detection_result, cv2)
                    elif self.gesture in ["Dab Left", "Dab Right"]:
                        frame = overlay_dab_image(frame, self.dab_img, self.gesture)
                    else:
                        frame = draw_landmarks_on_image(frame, self.detection_result, cv2)
                        if self.gesture:
                            cv2.putText(frame, f"Gesture: {self.gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
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
