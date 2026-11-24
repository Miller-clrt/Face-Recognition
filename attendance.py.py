import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Configuration
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "Attendance.csv"
TOLERANCE = 0.50  # Lower = stricter matching (0.6 is default, 0.50 reduces false positives)

def load_known_faces(directory):
    """Loads images from the directory and extracts face encodings and names."""
    images = []
    class_names = []
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created '{directory}' folder. Please add reference images and restart.")
        return [], []

    file_list = os.listdir(directory)
    print(f"Loading {len(file_list)} reference images...")

    for filename in file_list:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                # Use the filename (without extension) as the person's name
                name = os.path.splitext(filename)[0].replace("_", " ")
                class_names.append(name)

    encode_list = []
    for img, name in zip(images, class_names):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)
        if len(encodings) > 0:
            encode_list.append(encodings[0])
        else:
            print(f"Warning: No face detected in {name}'s reference image. Skipping.")

    print("Encoding complete.")
    return encode_list, class_names

def mark_attendance(name):
    """Logs the person's name, date, and time to a CSV file once per day."""
    today_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')

    # Create file with headers if it doesn't exist
    if not os.path.isfile(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w') as f:
            f.write('Name,Date,Time\n')

    with open(ATTENDANCE_FILE, 'r+') as f:
        existing_lines = f.readlines()
        recorded_today = set()
        
        for line in existing_lines[1:]:  # Skip header
            entry = line.strip().split(',')
            if len(entry) >= 2 and entry[1] == today_date:
                recorded_today.add(entry[0])

        if name not in recorded_today:
            f.writelines(f'{name},{today_date},{current_time}\n')
            print(f"[LOGGED] {name} marked present at {current_time}")

def main():
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
    
    if not known_encodings:
        print("No valid faces loaded. Exiting.")
        return

    # Initialize webcam (0 is usually the built-in camera)
    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from webcam.")
            break

        # Resize frame to 1/4 size for faster real-time processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces and compute encodings in the current webcam frame
        faces_cur_frame = face_recognition.face_locations(rgb_small_frame)
        encodes_cur_frame = face_recognition.face_encodings(rgb_small_frame, faces_cur_frame)

        for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
            matches = face_recognition.compare_faces(known_encodings, encode_face, tolerance=TOLERANCE)
            face_dis = face_recognition.face_distance(known_encodings, encode_face)

            name = "UNKNOWN"
            box_color = (0, 0, 255)  # Red for unknown

            if len(face_dis) > 0:
                best_match_index = np.argmin(face_dis)
                if matches[best_match_index]:
                    name = known_names[best_match_index].upper()
                    box_color = (0, 255, 0)  # Green for recognized
                    mark_attendance(name)

            # Scale face locations back up by 4 since we resized the frame to 1/4
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # Draw bounding box and name label
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), box_color, cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Face Recognition Attendance System', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()