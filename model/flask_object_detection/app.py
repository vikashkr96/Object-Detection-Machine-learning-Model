# app.py
from flask import Flask, render_template, Response ,jsonify
import cv2
from model import ObjectDetector
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

# Initialize the object detector
detector = ObjectDetector("yolov3.weights", "yolov3.cfg", "coco.names")

# Open the camera
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect objects
        detected_labels = detector.detect_objects(frame)
 
        # Draw boxes and labels on the frame
        for label, count in detected_labels.items():
            cv2.putText(frame, f"{label}: {count}", (10, 30 + 30 * list(detected_labels.keys()).index(label)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze')
def analyze():
    csv_file_path = 'detected_objects.csv'
    
    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        return jsonify(result="Error: The file was not found.")
    except pd.errors.EmptyDataError:
        return jsonify(result="Error: The file is empty.")
    
    # Check if the DataFrame is empty
    if df.empty:
        return jsonify(result="No data to analyze.")
    
    # Perform analysis (you can customize this part)
    analysis_result = df.describe().to_string()  # Example analysis result
    
    # Optionally save the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Object'], df['Count'], color='skyblue')
    plt.title('Detected Objects vs Count')
    plt.xlabel('Objects')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('detected_objects_plot.png')
    
    return jsonify(result=analysis_result)

if __name__ == '__main__':
    app.run(debug=True)