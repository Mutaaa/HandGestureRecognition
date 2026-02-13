# Origin: https://url.faunris.com/w2DCQb


import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result, cv2):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw landmarks and connections
        for i, landmark in enumerate(hand_landmarks):
            x = int(landmark.x * annotated_image.shape[1])
            y = int(landmark.y * annotated_image.shape[0])
            cv2.circle(annotated_image, (x, y), 4, (0, 255, 0), -1)
        # Draw connections (simple version: connect consecutive points)
        for i in range(1, len(hand_landmarks)):
            x1 = int(hand_landmarks[i-1].x * annotated_image.shape[1])
            y1 = int(hand_landmarks[i-1].y * annotated_image.shape[0])
            x2 = int(hand_landmarks[i].x * annotated_image.shape[1])
            y2 = int(hand_landmarks[i].y * annotated_image.shape[0])
            cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Draw handedness (left or right hand) on the image.
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * annotated_image.shape[1])
        text_y = int(min(y_coordinates) * annotated_image.shape[0]) - MARGIN
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image
