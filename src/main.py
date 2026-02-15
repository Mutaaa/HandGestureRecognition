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
        self.prev_gesture = None
        if os.path.exists(THUMB_IMAGE_PATH):
            self.thumb_img = cv2.imread(THUMB_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        else:
            self.thumb_img = None
        if os.path.exists(DAB_IMAGE_PATH):
            self.dab_img = cv2.imread(DAB_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        else:
            self.dab_img = None
        self.heart_img = cv2.imread("img/pendant.png", cv2.IMREAD_UNCHANGED) if os.path.exists("img/pendant.png") else None

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

    def detect_heart_hands(self, hands_landmarks):
        # hands_landmarks: [hand1, hand2], each is a list of 21 landmarks
        # Indexes: 4=thumb_tip, 8=index_tip
        hand1, hand2 = hands_landmarks[0], hands_landmarks[1]
        thumb1 = hand1[4]
        thumb2 = hand2[4]
        index1 = hand1[8]
        index2 = hand2[8]
        # Calculate Euclidean distances between thumb tips and index tips
        thumb_dist = ((thumb1.x - thumb2.x) ** 2 + (thumb1.y - thumb2.y) ** 2) ** 0.5
        index_dist = ((index1.x - index2.x) ** 2 + (index1.y - index2.y) ** 2) ** 0.5
        # Heuristic: both distances should be small (fingers close together)
        if thumb_dist < 0.13 and index_dist < 0.13:
            return True
        return False

    def detect_dab(self, hands_landmarks):
        # Heart hands detection first
        if self.detect_heart_hands(hands_landmarks):
            return "Heart Hands"
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

    def overlay_heart_effect(self, frame, hands_landmarks):
        if self.heart_img is None:
            return frame
        # Get thumb and index tips for both hands
        hand1, hand2 = hands_landmarks[0], hands_landmarks[1]
        l_thumb = hand1[4]
        l_index = hand1[8]
        r_thumb = hand2[4]
        r_index = hand2[8]
        fh, fw = frame.shape[:2]
        # Convert normalized coordinates to pixel
        pts = [
            (int(l_thumb.x * fw), int(l_thumb.y * fh)),
            (int(l_index.x * fw), int(l_index.y * fh)),
            (int(r_index.x * fw), int(r_index.y * fh)),
            (int(r_thumb.x * fw), int(r_thumb.y * fh)),
        ]
        # Center of the quadrilateral (average of 4 points)
        cx = sum([p[0] for p in pts]) // 4
        cy = sum([p[1] for p in pts]) // 4
        # Estimate size: average of distances between opposite points
        import math
        d1 = math.hypot(pts[0][0] - pts[2][0], pts[0][1] - pts[2][1])
        d2 = math.hypot(pts[1][0] - pts[3][0], pts[1][1] - pts[3][1])
        pendant_w = pendant_h = max(30, int((d1 + d2) / 2 * 0.9))
        pendant = cv2.resize(self.heart_img, (pendant_w, pendant_h))
        h, w = pendant.shape[:2]
        px1 = cx - w // 2
        py1 = cy - h // 2
        px2 = px1 + w
        py2 = py1 + h
        # Bounds check
        if px1 < 0 or py1 < 0 or px2 > fw or py2 > fh:
            return frame
        # Overlay with alpha
        if pendant.shape[2] == 4:
            alpha_s = pendant[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                frame[py1:py2, px1:px2, c] = (alpha_s * pendant[:, :, c] +
                                              alpha_l * frame[py1:py2, px1:px2, c])
        else:
            frame[py1:py2, px1:px2] = pendant[:, :, :3]
        return frame

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
                    self.prev_gesture = self.gesture
                    if self.gesture == "Thumbs Up":
                        bg = overlay_thumb_image(frame, self.thumb_img)
                        frame = draw_landmarks_on_image(bg, self.detection_result, cv2)
                    elif self.gesture in ["Dab Left", "Dab Right"]:
                        frame = overlay_dab_image(frame, self.dab_img, self.gesture)
                    elif self.gesture == "Heart Hands":
                        frame = draw_landmarks_on_image(frame, self.detection_result, cv2)
                        frame = self.overlay_heart_effect(frame, self.detection_result.hand_landmarks)
                        cv2.putText(frame, f"Gesture: {self.gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
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
