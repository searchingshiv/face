from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

# MongoDB connection
client = MongoClient("mongodb+srv://wwww:wwww@cluster0.u2hhi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['test']
collection = db['students']

# Load all student encodings into memory at the start
students = list(collection.find())
loaded_student_encodings = {student['name']: np.array(student['encoding']) for student in students}


def detect_and_draw_faces(frame, recognized_students):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        
        # Skip already recognized students
        for student_name, stored_encoding in loaded_student_encodings.items():
            if student_name not in recognized_students:
                match = face_recognition.compare_faces([stored_encoding], encoding, tolerance=0.6)
                if match[0]:
                    name = student_name
                    recognized_students.append(name)
                    break
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame, recognized_students


@app.route('/capture', methods=['POST'])
def capture_face_image():
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        return jsonify({"error": "Could not open webcam."}), 500

    print("Press 's' to capture a face image. Press 'q' to exit.")

    while True:
        ret, frame = video_capture.read()

        if not ret:
            video_capture.release()
            cv2.destroyAllWindows()
            return jsonify({"error": "Could not capture image."}), 500
        
        # Detect and draw faces in the frame
        frame_with_faces, face_encodings = detect_and_draw_faces(frame, [])

        # Display the video feed
        cv2.imshow('Webcam Feed', frame_with_faces)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # Capture image when 's' is pressed
        if key == ord('s'):
            video_capture.release()
            cv2.destroyAllWindows()
            if len(face_encodings) > 0:
                encoding = face_encodings[0].tolist()
                return jsonify({"encoding": encoding}), 200
            else:
                return jsonify({"error": "No face detected."}), 400
        
        # Exit when 'q' is pressed
        elif key == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            return jsonify({"error": "Operation canceled by user."}), 400


@app.route('/save_student', methods=['POST'])
def save_student_to_mongodb():
    data = request.json
    student_name = data.get("name")
    student_id = data.get("id")
    batch = data.get("batch")
    encoding = data.get("encoding")

    # Save to MongoDB
    student_data = {
        "name": student_name,
        "id": student_id,
        "batch": batch,
        "encoding": encoding  # Already a list from JSON
    }

    collection.insert_one(student_data)
    return jsonify({"message": "Student data saved to MongoDB."}), 200


def generate_video_feed():
    video_capture = cv2.VideoCapture(0)
    recognized_students = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect and draw faces in the frame
        frame_with_faces, recognized_students = detect_and_draw_faces(frame, recognized_students)
        _, buffer = cv2.imencode('.jpg', frame_with_faces)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start_attendance', methods=['GET'])
def start_attendance():
    video_capture = cv2.VideoCapture(0)
    recognized_students = []

    if not video_capture.isOpened():
        return jsonify({"error": "Could not open webcam."}), 500

    while True:
        ret, frame = video_capture.read()
        if not ret:
            video_capture.release()
            return jsonify({"error": "Could not capture image."}), 500
        
        # Detect and encode faces in the frame
        frame_with_faces, recognized_students = detect_and_draw_faces(frame, recognized_students)

        # Update recognized students in real-time
        if len(recognized_students) > 0:
            video_capture.release()
            return jsonify({"recognized_students": recognized_students}), 200

        # Add a small delay to limit the number of frames processed
        cv2.waitKey(1)

    video_capture.release()
    return jsonify({"recognized_students": recognized_students}), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
