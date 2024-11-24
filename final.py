import os
import pickle
import numpy as np
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from ultralytics import YOLO
import cvzone
import tkinter as tk
from PIL import Image, ImageTk
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Initialize Firebase Admin

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facerecognition-84e51-default-rtdb.firebaseio.com/",
    'storageBucket': "facerecognition-84e51.appspot.com"
})

bucket = storage.bucket()

# Initialize YOLO model
model = YOLO("v8.pt")

# Required items sets
# required_items_set_1 = {'Coat', 'Slacks', 'Tie', 'Shoes'}
# required_items_set_2 = {'Coat', 'Id', 'Ribbon', 'Shoes', 'Skirt'}
required_items_set_1 = {'Coat', 'Slacks',"Long-Sleeves", 'Id', 'Tie', 'Shoes'}
required_items_set_2 = {'Coat', 'Id', 'Ribbon',"Long-Sleeves", 'Shoes', 'Skirt'}
classNames = ["Coat", "Id", "Long-Sleeves", "Ribbon", "Shoes", "Skirt", "Slacks", "Tie"]

def setup_uniform_detector():
    root = tk.Tk()
    root.title("Uniform Detector")
    root.geometry("1280x720")
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=640, height=480)
    canvas.pack(side=tk.LEFT, padx=(160,0))

    info_frame = tk.Frame(root, bg="black")
    info_frame.pack(side=tk.RIGHT, fill=tk.Y)

    timer_label = tk.Label(info_frame, text="", font=("Helvetica", 16), fg="red", bg="black")
    timer_label.pack(pady=(10, 0))

    detected_label = tk.Label(info_frame, text="Detected Items:", font=("Helvetica", 16), bg="black", fg="white")
    detected_label.pack(pady=(10, 0))

    detected_items_list = tk.Listbox(info_frame, font=("Helvetica", 14), bg="black", fg="white", selectbackground="gray", height=10)
    detected_items_list.pack(padx=10, pady=10)

    return root, canvas, timer_label, detected_items_list

def get_missing_items(detected_items, required_items_set):
    return required_items_set - detected_items

def send_email(subject, body, student_info=None, missing_items=None):
    SMTP_SERVER = 'smtp.gmail.com'
    SMTP_PORT = 587
    SENDER_EMAIL = 'itsfaithyolo@gmail.com'
    SENDER_PASSWORD = 'wler kgio rvou ujex'
    RECEIVER_EMAIL = 'kurt.palomo.lauwrence@gmail.com'

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject

    # Enhanced email body with missing items
    if student_info:
        body = f"""
Student Information:
ID: {student_info['id']}
Name: {student_info['name']}

"""
    if missing_items:
        body += f"""
Missing Uniform Items:
{', '.join(missing_items)}

"""

    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def process_student(cap, img, encodeListKnown, studentIds):
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            id = studentIds[matchIndex]
            student_data = db.reference(f'Students/{id}').get()

            if student_data:
                student_info = {
                    'id': id,
                    'name': student_data['name'],
                    'gender': student_data['Gender']  # Make sure to fetch gender
                }
                # Display recognition success in full screen
                info_text = f"Student ID: {id}"
                name_text = f"Name: {student_data['name']}"
                cv2.putText(img, info_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(img, name_text, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(img, "Starting Uniform Detection...", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.namedWindow("Face Recognition", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Face Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Face Recognition", img)
                cv2.waitKey(2000)
                return True, student_info

    # If no match is found, you can clear or reset your previous data here.
    # Return False if the face is not recognized.
    return False, None


def run_system():
    print("Loading Encode File ...")
    with open('EncodeFile.p', 'rb') as file:
        encodeListKnownWithIds = pickle.load(file)
    encodeListKnown, studentIds = encodeListKnownWithIds
    print("Encode File Loaded")

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height

    # Add a delay after initializing the camera
    print("Camera initializing. Please wait...")
    time.sleep(2)  # Wait for 2 seconds

    while True:  # Main loop for the system
        face_recognized = False
        student_info = None

        # Face Recognition Phase
        while not face_recognized:
            success, img = cap.read()
            if not success:
                print("Failed to capture image from webcam.")
                break

            # Start scanning for faces after the delay
            faceCurFrame = face_recognition.face_locations(img)
            encodeCurFrame = face_recognition.face_encodings(img, faceCurFrame)

            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    id = studentIds[matchIndex]
                    student_data = db.reference(f'Students/{id}').get()

                    if student_data:
                        student_info = {
                            'id': id,
                            'name': student_data['name'],
                            'gender': student_data.get('Gender', 'unknown')  # Access gender directly
                        }
                        
                        # Display recognition success in full screen
                        info_text = f"Student ID: {id}"
                        name_text = f"Name: {student_data['name']}"
                        cv2.putText(img, info_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        cv2.putText(img, name_text, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(img, "Starting Uniform Detection...", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        cv2.namedWindow("Face Recognition", cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty("Face Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.imshow("Face Recognition", img)
                        cv2.waitKey(2000)
                        face_recognized = True  # Mark face as recognized

                else:
                    cv2.putText(img, "Unknown Person", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imshow("Face Recognition", img)
                    cv2.waitKey(1000)

            cv2.imshow("Face Recognition", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return  # Exit the main function

        if face_recognized:
            # Transition to Uniform Detection
            cv2.destroyAllWindows()
            
            # Setup and run uniform detection
            root, canvas, timer_label, detected_items_list = setup_uniform_detector()
            timer_duration = 15
            message_display_duration = 5

            detected_items = set()
            start_time = time.time()

            # Select the required items set based on gender
            if student_info['gender'].lower() == 'female':
                required_items_set = required_items_set_2
            else:
                required_items_set = required_items_set_1

            while True:
                success, img = cap.read()
                if not success:
                    print("Failed to capture image from webcam.")
                    break  # If capture fails, break out of the loop

                img = cv2.resize(img, (640, 480))

                elapsed_time = time.time() - start_time
                remaining_time = max(0, timer_duration - int(elapsed_time))
                timer_label.config(text=f"Time Remaining: {remaining_time} seconds")

                if remaining_time == 0:
                    if required_items_set.issubset(detected_items):
                        canvas.delete("all")
                        canvas.create_text(320, 240, text="Complete Uniform, Have a good day!", font=("Helvetica", 24), fill="green")
                    else:
                        canvas.delete("all")
                        canvas.create_text(320, 240, text="Incomplete Uniform, Don't have a Good day!", font=("Helvetica", 24), fill="red")
                        missing_items = get_missing_items(detected_items, required_items_set)
                        send_email(
                            subject="Uniform Detection Alert",
                            body="An incomplete uniform was detected.",
                            student_info=student_info,
                            missing_items=missing_items
                        )

                    root.update()
                    time.sleep(message_display_duration)
                    break  # Exit the uniform detection loop

                results = model(img, stream=True)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        cvzone.cornerRect(img, (x1, y1, w, h))

                        conf = round(float(box.conf[0]), 2)
                        cls = int(box.cls[0])
                        currentClass = classNames[cls]

                        if conf > 0.2 and currentClass in required_items_set:
                            detected_items.add(currentClass)
                            myColor = (0, 0, 255) if conf > 0.5 else (255, 0, 0)
                            cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), 
                            scale=1, thickness=1, colorB=myColor, colorT=(255, 255, 255), 
                            colorR=myColor, offset=5)
                            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

                detected_items_list.delete(0, tk.END)
                for item in detected_items:
                    detected_items_list.insert(tk.END, item)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(image=img)
                canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)

                root.update_idletasks()
                root.update()

            # Clean up the GUI after uniform detection
            root.destroy()

        # Check if the user wants to exit the main loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()

