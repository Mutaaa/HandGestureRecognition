import cv2
import numpy as np
import mediapipe as mp


def overlay_thumb_image(frame, thumb_img):
    if thumb_img is None:
        return frame
    # Resize overlay to be larger in the center
    overlay = cv2.resize(thumb_img, (350, 350))
    h, w = overlay.shape[:2]
    fh, fw = frame.shape[:2]
    # Center position
    y1 = fh // 2 - h // 2
    y2 = y1 + h
    x1 = fw // 2 - w // 2
    x2 = x1 + w

    if overlay.shape[2] == 4:
        alpha_s = overlay[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (alpha_s * overlay[:, :, c] +
                                      alpha_l * frame[y1:y2, x1:x2, c])
    else:
        frame[y1:y2, x1:x2] = overlay[:, :, :3]
    return frame


def overlay_dab_image(frame, dab_img, gesture):
    if dab_img is None:
        return frame
    overlay = dab_img.copy()
    h, w = overlay.shape[:2]
    fh, fw = frame.shape[:2]
    if gesture == "Dab Right":
        overlay = cv2.flip(overlay, 1)
    y1 = fh // 2 - h // 2
    y2 = y1 + h
    x1 = fw // 2 - w // 2
    x2 = x1 + w
    if y1 < 0 or x1 < 0 or y2 > fh or x2 > fw:
        return frame
    if overlay.shape[2] == 4:
        alpha_s = overlay[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (alpha_s * overlay[:, :, c] +
                                      alpha_l * frame[y1:y2, x1:x2, c])
    else:
        frame[y1:y2, x1:x2] = overlay[:, :, :3]
    return frame
