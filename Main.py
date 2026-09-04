import sys
import cv2
import time
import threading
from PIL import Image, ImageTk
import tkinter as tk
import pyttsx3
import numpy as np
import os
import mediapipe as mp
import pickle
import queue

video_queue = queue.Queue()

root = tk.Tk()
root.title("Gestura")
root.geometry("1280x720")
root.resizable(True, True)
#================================================Frame=================================================
left_frame = tk.Frame(root, width=220)
left_frame.pack(side="left", fill="y")

header_label = tk.Label(left_frame, text="Menu", font=("Helvetica", 16, "bold"))
header_label.pack(pady=(20, 10))

translation_btn = tk.Button(left_frame, text="Translation", font=("Helvetica", 30))
settings_btn = tk.Button(left_frame, text="Settings", font=("Helvetica", 30))

translation_btn.pack(pady=15, padx=15, fill="x")
settings_btn.pack(pady=15, padx=15, fill="x")

main_frame = tk.Frame(root)
main_frame.pack(side="right", fill="both", expand=True)

translation_frame = tk.Frame(main_frame)
settings_frame = tk.Frame(main_frame)
translation_frame.pack(fill="both", expand=True)

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

try:
    root.iconbitmap(resource_path("gestura_icon.ico"))
except Exception:
    pass

model_path = resource_path('model.p')
print("loading model from:", model_path)
try:
    model_dict = pickle.load(open(model_path, 'rb'))
    gesture_model_1h = model_dict.get('gesture_model_1h')
    gesture_model_2h = model_dict.get('gesture_model_2h')
except Exception:
    print("Failed to load model")
    sys.exit(1)

Gestures_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: '', 10: 'K',
               11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: '', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
               21: 'V', 22: 'W', 23: '', 24: 'Y', 25: '', 26: ' ', 27: 'rest', 28: 'NG', 29: '', 30: 'TATAY ',
               31: 'GALIT ', 32: '', 33: 'MABILIS ', 34: 'KAMUSTA ', 35: 'NANAY ', 36: 'NASAKTAN ', 37: 'MAHAL KITA ',
               38: 'PAKIUSAP ', 39: 'KALIKASAN ', 40: 'PATAWAD ',
               41: 'MALI ', 42: 'MAIKSI ', 43: 'KA ', 44: 'ONE ', 45: 'TWO ', 46: 'THREE ', 47: 'FOUR ', 48: 'FIVE ',
               49: 'SIX ', 50: 'SEVEN ',
               51: 'EIGHT ', 52: 'NINE ', 53: 'TEN ', 54: '', 55: 'KAMUSTA ', 56: 'MABUTI ', 57: 'MAGANDANG ',
               58: 'UMAGA ', 59: 'TANGHALI ', 60: 'GABI ',
               61: 'MAHAL ', 62: 'KILIG ', 63: 'PIKON ', 64: 'ALAM KO ', 65: 'TEKA ', 66: 'AKO ', 67: 'SI ', 68: 'OO ',
               69: 'HINDI ', 70: 'KAIN ', 71: 'INOM '}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

custom_landmark_style = mp_drawing.DrawingSpec(color=(80, 80, 80), thickness=2, circle_radius=2)
custom_connection_style = mp_drawing.DrawingSpec(color=(192, 192, 192), thickness=2)

sentence = ""
last_character = ""
last_prediction_time = 0
prediction_delay = 3.0
running = True
sentence_changed = False
min_conf = 0.60

engine = None

#================================================Button Logic=================================================
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
except Exception:
    engine = None

def show_translation():
    translation_frame.pack(fill="both", expand=True)
    settings_frame.pack_forget()

def show_settings():
    settings_frame.pack(fill="both", expand=True)
    translation_frame.pack_forget()

def update_video_label(img, w, h):
    try:
        tk_img = ImageTk.PhotoImage(img.resize((w, h)))
        video_label.configure(image=tk_img)
        video_label.image = tk_img
    except Exception as e:
        print(f"Video update error: {e}")

#================================================UI Config=================================================
translation_btn.configure(command=show_translation)
settings_btn.configure(command=show_settings)

translation_label = tk.Label(translation_frame, text="Sign Language Translation", font=("Helvetica", 20, "bold"))
translation_label.pack(pady=(20, 10))

video_frame = tk.Frame(translation_frame)
video_frame.pack(fill="both", expand=True, padx=20, pady=10)

video_label = tk.Label(video_frame, text="")
video_label.pack(fill="both", expand=True)
video_label.image = None

translated_label = tk.Label(translation_frame, text="Translated Text:", font=("Helvetica", 25))
translated_label.pack(pady=(10, 5))

sentence_textbox = tk.Text(translation_frame, wrap="word", font=("Helvetica", 25), height=3)
sentence_textbox.pack(fill="x", padx=20, pady=(5, 20))
sentence_textbox.insert("1.0", "Translated sentence will appear here...")

control_frame = tk.Frame(translation_frame)
control_frame.pack(pady=(0, 20))

clear_btn = tk.Button(control_frame, text="Clear (R)", font=("Helvetica", 25), command=lambda: clear_sentence())
clear_btn.grid(row=0, column=0, padx=10)

delete_btn = tk.Button(control_frame, text="Delete Last (Backspace)", font=("Helvetica", 25), command=lambda: delete_last())
delete_btn.grid(row=0, column=1, padx=10)

quit_btn = tk.Button(control_frame, text="Quit (Q)", font=("Helvetica", 25), command=lambda: quit_app())
quit_btn.grid(row=0, column=2, padx=10)

settings_label = tk.Label(settings_frame, text="Settings", font=("Helvetica", 20, "bold"))
settings_label.pack(pady=(20, 10))

sensitivity_label = tk.Label(settings_frame, text="Detection Sensitivity:", font=("Helvetica", 14))
sensitivity_label.pack(pady=(10, 5))

sensitivity_slider = tk.Scale(settings_frame, from_=0.3, to=0.9, resolution=0.01, orient="horizontal", command=lambda val: update_sensitivity(val))
sensitivity_slider.set(min_conf)
sensitivity_slider.pack(pady=(5, 5), fill="x", padx=20)

sensitivity_value_label = tk.Label(settings_frame, text=f"Current: {min_conf:.2f}", font=("Helvetica", 12))
sensitivity_value_label.pack(pady=(0, 15))

delay_label = tk.Label(settings_frame, text="Detection Delay (seconds):", font=("Helvetica", 14))
delay_label.pack(pady=(10, 5))

delay_slider = tk.Scale(settings_frame, from_=0.5, to=5.0, resolution=0.1, orient="horizontal", command=lambda val: update_delay(val))
delay_slider.set(prediction_delay)
delay_slider.pack(pady=(5, 5), fill="x", padx=20)

delay_value_label = tk.Label(settings_frame, text=f"Current: {prediction_delay:.2f}s", font=("Helvetica", 12))
delay_value_label.pack(pady=(0, 15))

#================================================Functions and Logic=================================================

def update_sensitivity(val):
    global min_conf
    min_conf = float(val)
    sensitivity_value_label.configure(text=f"Current: {min_conf:.2f}")
    print(f"Detection sensitivity updated to {min_conf:.2f}")

def update_delay(val):
    global prediction_delay
    prediction_delay = float(val)
    delay_value_label.configure(text=f"Current: {prediction_delay:.2f}s")
    print(f"Detection delay updated to {prediction_delay:.2f} seconds")

def speak(text):
    if engine and text.strip():
        def speak_thread():
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
        threading.Thread(target=speak_thread, daemon=True).start()

def update_sentence_display():
    sentence_textbox.delete("1.0", "end")
    sentence_textbox.insert("1.0", sentence if sentence else "Translated sentence will appear here...")

def check_sentence():
    global sentence_changed
    if sentence_changed:
        update_sentence_display()
        sentence_changed = False
    root.after(100, check_sentence)

def clear_sentence():
    global sentence, last_character, sentence_changed
    sentence = ""
    last_character = ""
    sentence_changed = True

def delete_last():
    global sentence, last_character, sentence_changed
    if sentence:
        sentence = sentence[:-1]
        last_character = ""
        sentence_changed = True

def quit_app():
    global running
    running = False
    if 'cap' in globals() and cap.isOpened():
        cap.release()
    root.quit()

def on_key_press(event):
    if event.char and event.char.lower() == 'r':
        clear_sentence()
    elif event.keysym == 'BackSpace':
        delete_last()
    elif event.char and event.char.lower() == 'q':
        quit_app()
    elif event.keysym == 'Return':
        if sentence.strip():
            speak(sentence)

root.bind('<Key>', on_key_press)
root.focus_set()

#================================================Video Feed=================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    running = False

def check_video_queue():
    try:
        while True:
            img, w, h = video_queue.get_nowait()
            update_video_label(img, w, h)
    except queue.Empty:
        pass
    except Exception as e:
        pass
    root.after(100, check_video_queue)

def video_loop():
    global sentence, last_character, last_prediction_time, running, sentence_changed, min_conf

    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=min_conf, max_num_hands=2)

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = hands.process(frame_rgb)
        except Exception:
            continue

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    custom_landmark_style,
                    custom_connection_style
                )

            x_, y_, data_aux = [], [], []
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                min_x, min_y = min(x_), min(y_)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min_x)
                    data_aux.append(y - min_y)

            predicted_character = None
            if len(results.multi_hand_landmarks) == 1 and gesture_model_1h is not None and len(data_aux) >= 42:
                try:
                    prediction = gesture_model_1h.predict([np.asarray(data_aux[:42])])
                    predicted_character = Gestures_map[int(prediction[0])]
                except Exception:
                    pass
            elif len(results.multi_hand_landmarks) == 2 and gesture_model_2h is not None and len(data_aux) >= 84:
                try:
                    prediction = gesture_model_2h.predict([np.asarray(data_aux[:84])])
                    predicted_character = Gestures_map[int(prediction[0])]
                except Exception:
                    pass

            if predicted_character and predicted_character.strip():
                current_time = time.time()
                if predicted_character != last_character and predicted_character != 'rest' and (
                        current_time - last_prediction_time) > prediction_delay:
                    sentence += predicted_character
                    last_character = predicted_character
                    last_prediction_time = current_time
                    sentence_changed = True

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video_queue.put((img, w, h))
        time.sleep(0.03)

    hands.close()

video_thread = threading.Thread(target=video_loop, daemon=True)
video_thread.start()

root.after(100, check_video_queue)
root.after(100, check_sentence)
root.protocol("WM_DELETE_WINDOW", quit_app)
root.mainloop()
