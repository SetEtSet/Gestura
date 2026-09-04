import os
import cv2
import time
import threading

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 66
dataset_size = 200
timer = 3
countdown_running = False
countdown_value = 0
countdown_started = False

Gestures_map = { 0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G' ,7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N' ,13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
    20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: '', 25: 'NG' ,26: 'TATAY', 27: 'GALIT', 28: 'MABILIS',29: 'KAMUSTA',
    30: 'NANAY', 31: 'NASAKTAN', 32: 'MAHAL KITA', 33: 'PAKIUSAP', 34: 'KALIKASAN', 35: 'PATAWAD', 36: 'MALI', 37: 'MAIKSI', 38: 'KA', 39: 'ONE',
    40: 'TWO', 41: 'THREE', 42: 'FOUR', 43: 'FIVE', 44: 'SIX', 45: 'SEVEN', 46: 'EIGHT', 47: 'NINE', 48: 'TEN', 49: 'KAMUSTA',
    50: 'MABUTI', 51: 'MAGANDANG', 52: 'UMAGA', 53: 'TANGHALI', 54: 'GABI', 55: 'MAHAL', 56: 'KILIG', 57: 'PIKON', 58: 'ALAM KO', 59: 'TEKA',
    60: 'AKO', 61: 'SI', 62: 'OO', 63: 'HINDI', 64: 'KAIN', 65: 'INOM' }

def countdown(seconds):
    global countdown_value, countdown_running
    countdown_running = True
    for t in range(seconds, 0, -1):
        countdown_value = t
        print("Starting in", t)
        time.sleep(1)
    countdown_value = 0
    countdown_running = False

cap = cv2.VideoCapture(0)
exit_program = False

for j in range(number_of_classes):
    if exit_program:
        break

    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for gesture {}'.format(Gestures_map[j]))

    countdown_started = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'Collecting Data for Gesture: {Gestures_map[j]}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, f'Collecting Data for Gesture: {Gestures_map[j]}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, 'Press "esc" to close', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, 'Press "esc" to close', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if countdown_value > 0 and countdown_running:
            cv2.putText(frame, f'Starting in {countdown_value}...', (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, f'Starting in {countdown_value}...', (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            exit_program = True
            break
        key = cv2.waitKey(1)

        if key == ord('q') and not countdown_running:
            countdown_started = True
            thread = threading.Thread(target=countdown, args=(timer,), daemon=True)
            thread.start()
        elif key == 27:
            exit_program = True
            break
        if countdown_started and not countdown_running and countdown_value == 0:
            break

    if exit_program:
        break

    existing_files = os.listdir(class_dir)
    start_index = len(existing_files)

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'Class: {j}  Image: {start_index+counter+1}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, f'Class: {j}  Image: {start_index+counter+1}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, 'Press "S" to stop collection for this gesture', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, 'Press "S" to stop collection for this gesture', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            exit_program = True
            break
        key = cv2.waitKey(25)

        if key == ord('s'):
            print(f'Stopped collection for gesture {Gestures_map[j]} at image {start_index+counter+1}')
            break
        elif key == 27:
            exit_program = True
            break

        filename = os.path.join(class_dir, f'{start_index+counter}.jpg')
        cv2.imwrite(filename, frame)

        counter += 1

    if exit_program:
        break

cap.release()
cv2.destroyAllWindows()