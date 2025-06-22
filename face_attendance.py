import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
import random

# --- 1. Setup and Initialization ---

KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_FILE = 'attendance.csv'

EAR_THRESH = 0.25# Eye Aspect Ratio threshold for blink detection
EAR_CONSEC_FRAMES = 2# Number of consecutive frames the eye must be below the threshold
HEAD_TURN_PIXEL_THRESH = 10# Horizontal pixel distance for head turn
LIVENESS_CHALLENGE_TIMEOUT = 70  # ~2.3s at 30fps

# --- 2. Load Known Faces ---

def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(known_faces_dir):
        print(f"Creating directory: {known_faces_dir}. Please add images of known people here.")
        os.makedirs(known_faces_dir)
    
    print("Loading known faces...")
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                name = os.path.splitext(filename)[0].replace('_', ' ').title()
                image_path = os.path.join(known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                face_encodings_list = face_recognition.face_encodings(image)
                
                if face_encodings_list:
                    known_face_encodings.append(face_encodings_list[0])
                    known_face_names.append(name)
                    print(f"Loaded face for {name}")
                else:
                    print(f"Warning: No face found in {filename}. Skipping.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if not known_face_names:
        print("\nWarning: No known faces loaded. The 'known_faces' directory is empty or no faces were detected.")
        
    return known_face_encodings, known_face_names

# --- 3. Liveness Detection ---

def calculate_ear(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# --- 4. Attendance Management ---

def initialize_attendance_log(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
        df.to_csv(file_path, index=False)
        return df

def mark_attendance(name, attendance_df, file_path):
    now = datetime.now()
    date_string = now.strftime('%d-%m-%y')
    time_string = now.strftime('%H:%M:%S')
    is_already_marked = not attendance_df[(attendance_df['Name'] == name) & (attendance_df['Date'] == date_string)].empty
    
    if not is_already_marked:
        new_row = pd.DataFrame([{'Name': name, 'Date': date_string, 'Time': time_string}])
        updated_df = pd.concat([attendance_df, new_row], ignore_index=True)
        updated_df.to_csv(file_path, index=False)
        print(f"Attendance marked for {name} at {time_string}")
        return updated_df
    return attendance_df

# --- 5. Main Application Logic ---

def main():
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
    attendance_df = initialize_attendance_log(ATTENDANCE_FILE)
    liveness_status = {}
    liveness_trackers = {}

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nStarting real-time face recognition... Press 'q' to quit.")

    #  Force show the initial frame so the window appears
    cv2.namedWindow('Face Recognition Attendance System', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Face Recognition Attendance System', 100, 100)
    ret, frame = video_capture.read()
    if ret:
        cv2.imshow('Face Recognition Attendance System', frame)
        cv2.waitKey(1)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)

        for i, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unauthorized"
            status_text = ""

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    is_already_marked_today = not attendance_df[(attendance_df['Name'] == name) & 
                                                                (attendance_df['Date'] == datetime.now().strftime('%d-%m-%y'))].empty

                    if name not in liveness_status:
                        liveness_status[name] = 'awaiting_blink'
                        liveness_trackers[name] = {'blink_counter': 0}

                    current_state = liveness_status[name]

                    if is_already_marked_today:
                        status_text = "Attendance Marked"
                        liveness_status[name] = 'verified'

                    elif current_state == 'awaiting_blink':
                        status_text = "Blink to verify"
                        face_landmarks = face_landmarks_list[i]
                        left_ear = calculate_ear(face_landmarks['left_eye'])
                        right_ear = calculate_ear(face_landmarks['right_eye'])
                        ear = (left_ear + right_ear) / 2.0

                        if ear < EAR_THRESH:
                            liveness_trackers[name]['blink_counter'] += 1
                        else:
                            if liveness_trackers[name]['blink_counter'] >= EAR_CONSEC_FRAMES:
                                print(f"Blink confirmed for {name}.")
                                challenge = random.choice(['left', 'right'])
                                liveness_status[name] = f'awaiting_head_turn_{challenge}'
                                nose_tip = face_landmarks['nose_tip'][0]
                                liveness_trackers[name]['initial_nose_x'] = nose_tip[0]
                                liveness_trackers[name]['challenge_counter'] = 0
                            liveness_trackers[name]['blink_counter'] = 0

                    elif current_state.startswith('awaiting_head_turn'):
                        challenge_type = current_state.split('_')[-1]
                        status_text = f"Turn head to the {challenge_type}"

                        liveness_trackers[name]['challenge_counter'] += 1
                        if liveness_trackers[name]['challenge_counter'] > LIVENESS_CHALLENGE_TIMEOUT:
                            print(f"Challenge timed out for {name}. Resetting.")
                            liveness_status[name] = 'awaiting_blink'
                            continue

                        nose_tip = face_landmarks_list[i]['nose_tip'][0]
                        current_nose_x = nose_tip[0]
                        initial_nose_x = liveness_trackers[name]['initial_nose_x']
                        delta_x = current_nose_x - initial_nose_x

                        turn_detected = (
                            (challenge_type == 'right' and delta_x > HEAD_TURN_PIXEL_THRESH) or
                            (challenge_type == 'left' and delta_x < -HEAD_TURN_PIXEL_THRESH)
                        )

                        if turn_detected:
                            print(f"Head turn confirmed for {name}.")
                            liveness_status[name] = 'verified'
                            attendance_df = mark_attendance(name, attendance_df, ATTENDANCE_FILE)
                            status_text = "Liveness Verified"

                    elif current_state == 'verified':
                        status_text = "Liveness Verified" if not is_already_marked_today else "Attendance Marked"

            top, right, bottom, left = (coord * 4 for coord in face_locations[i])
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 12), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            if status_text:
                cv2.putText(frame, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Face Recognition Attendance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("Application stopped.")

if __name__ == "__main__":
    main()
