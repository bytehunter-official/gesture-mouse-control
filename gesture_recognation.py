import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

screen_width, screen_height = pyautogui.size()
screen_aspect_ratio = screen_width / screen_height

if frame_width / frame_height > screen_aspect_ratio:
    rect_h = frame_height
    rect_w = screen_aspect_ratio * rect_h
else:
    rect_w = frame_width
    rect_h = rect_w / screen_aspect_ratio

rect_w, rect_h = int(rect_w*2/3), int(rect_h*2/3)
rect_x = int((frame_width - rect_w) // 2)
rect_y = int((frame_height - rect_h) // 2)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

click_threshold = 0.2
double_click_interval = 0.5
long_press_duration = 5.0

click_state = "idle"
press_start_time = 0
last_click_time = 0


def is_index_finger_extended(hand_landmarks, w, h, distance_0_17):
    tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    wrost = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    distance = (np.sqrt((tip.x - wrost.x) ** 2 + 
                        (tip.y - wrost.y) ** 2 + 
                        (tip.z - wrost.z) ** 2)) / distance_0_17

    threshold = 2

    return distance > threshold


def is_win_tab(hand_landmarks, w, h, distance_0_17):
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    wrost = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    distence_middle = (np.sqrt((middle_tip.x - wrost.x) ** 2 + 
                              (middle_tip.y - wrost.y) ** 2 + 
                              (middle_tip.z - wrost.z) ** 2)) / distance_0_17
    distence_ring = (np.sqrt((ring_tip.x - wrost.x) ** 2 + 
                            (ring_tip.y - wrost.y) ** 2 + 
                            (ring_tip.z - wrost.z) ** 2)) / distance_0_17
    distence_pinky_thumb = (np.sqrt((pinky_tip.x - thumb_tip.x) ** 2 + 
                                   (pinky_tip.y - thumb_tip.y) ** 2 + 
                                   (pinky_tip.z - thumb_tip.z) ** 2)) / distance_0_17

    return distence_middle > 2 and distence_ring > 2 and distence_pinky_thumb < 0.3


def is_mouse_wheel(hand_landmarks, w, h, distance_0_17):
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

    distence_middle = (np.sqrt((middle_tip.x - middle_mcp.x) ** 2 + 
                              (middle_tip.y - middle_mcp.y) ** 2 + 
                              (middle_tip.z - middle_mcp.z) ** 2)) / distance_0_17
    distence_ring_thumb = (np.sqrt((ring_tip.x - thumb_tip.x) ** 2 + 
                                  (ring_tip.y - thumb_tip.y) ** 2 + 
                                  (ring_tip.z - thumb_tip.z) ** 2)) / distance_0_17
    distence_ring_thumb_pip = (np.sqrt((ring_pip.x - thumb_tip.x) ** 2 + 
                                      (ring_pip.y - thumb_tip.y) ** 2 + 
                                      (ring_pip.z - thumb_tip.z) ** 2)) / distance_0_17

    return (distence_middle > 0.85 and 
            (distence_ring_thumb < 0.3 or distence_ring_thumb_pip < 0.3), 
            distence_ring_thumb_pip, 
            distence_ring_thumb)


def is_finger_extended(tip, pip):
    return tip.y < pip.y


def is_win_h(hand_landmarks, w, h, distance_0_17):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_pip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    wrost = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    thumb_extended = is_finger_extended(thumb_tip, thumb_pip)
    pinky_extended = is_finger_extended(pinky_tip, pinky_pip)
    
    distence_middle = (np.sqrt((middle_tip.x - wrost.x) ** 2 + 
                             (middle_tip.y - wrost.y) ** 2 + 
                             (middle_tip.z - wrost.z) ** 2)) / distance_0_17
    distence_ring = (np.sqrt((ring_tip.x - wrost.x) ** 2 + 
                            (ring_tip.y - wrost.y) ** 2 + 
                            (ring_tip.z - wrost.z) ** 2)) / distance_0_17

    return (thumb_extended and
            distence_middle < 1.65 and 
            distence_ring < 1.65 and
            pinky_extended)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        h, w, _ = frame.shape

        hand_0 = hand_landmarks.landmark[0]
        hand_17 = hand_landmarks.landmark[17]
        distance_0_17 = np.sqrt((hand_0.x - hand_17.x) ** 2 + 
                               (hand_0.y - hand_17.y) ** 2 + 
                               (hand_0.z - hand_17.z) ** 2)

        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        wrost = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

        distance = (np.sqrt((middle_pip.x - thumb_tip.x) ** 2 + 
                           (middle_pip.y - thumb_tip.y) ** 2 + 
                           (middle_pip.z - thumb_tip.z) ** 2)) / distance_0_17
        distence_wrost_middle = (np.sqrt((wrost.x - middle_pip.x) ** 2 + 
                                       (wrost.y - middle_pip.y) ** 2 + 
                                       (wrost.z - middle_pip.z) ** 2)) / distance_0_17

        if is_index_finger_extended(hand_landmarks, w, h, distance_0_17):
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)
            
            norm_x = np.clip((x_index - rect_x) / rect_w, 0, 1)
            norm_y = np.clip((y_index - rect_y) / rect_h, 0, 1)
            pyautogui.moveTo(int(norm_x * screen_width), int(norm_y * screen_height))

            current_time = time.time()

            if is_win_tab(hand_landmarks, w, h, distance_0_17):
                pyautogui.hotkey('win', 'tab')
                print(">> 手势感应：召唤任务视图")
                time.sleep(1)

            if is_win_h(hand_landmarks, w, h, distance_0_17):
                pyautogui.hotkey('win', 'h')
                print(">> 手势感应：开启语音输入")
                time.sleep(0.5)

            t_f, distence_ring_thumb_pip, distence_ring_thumb = is_mouse_wheel(hand_landmarks, w, h, distance_0_17)
            if t_f:
                if distence_ring_thumb_pip - distence_ring_thumb < 0:
                    pyautogui.scroll(80)
                else:
                    pyautogui.scroll(-80)

            if distance < click_threshold and distence_wrost_middle < 1.7:
                if click_state == "idle":
                    press_start_time = current_time
                    click_state = "pressed"
                elif click_state == "pressed":
                    if current_time - press_start_time >= long_press_duration:
                        pyautogui.mouseDown()
                        print(">> 手势感应：长按开始")
                        click_state = "long_press"
            else:
                if click_state == "pressed":
                    if current_time - last_click_time < double_click_interval:
                        pyautogui.doubleClick()
                        print(">> 手势感应：双击")
                    else:
                        pyautogui.click()
                        print(">> 手势感应：单击")
                    last_click_time = current_time
                elif click_state == "long_press":
                    pyautogui.mouseUp()
                    print(">> 手势感应：长按释放")
                click_state = "idle"

    cv2.rectangle(frame, (rect_x, rect_y), 
                 (rect_x + rect_w, rect_y + rect_h), 
                 (255, 0, 0), 2)

    cv2.imshow('Gesture Mouse Control', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()