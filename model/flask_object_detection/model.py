import cv2
import numpy as np
import pandas as pd
import os

class ObjectDetector:
    def __init__(self, weights_path, config_path, names_path):
        self.net = cv2.dnn.readNet("flask_object_detection\yolov3.weights", "flask_object_detection\yolov3.cfg")
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

        with open("flask_object_detection\coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect_objects(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detected_labels = {}
        for i in range(len(boxes)):
            if i in indexes:
                label = str(self.classes[class_ids[i]])
                if label in detected_labels:
                    detected_labels[label] += 1
                else:
                    detected_labels[label] = 1

        # Save detected labels to CSV
        self.save_to_csv(detected_labels)

        return detected_labels

    def save_to_csv(self, detected_labels):
        # Create a DataFrame from the detected labels
        df = pd.DataFrame(list(detected_labels.items()), columns=['Object', 'Count'])
        
        # Specify the CSV file path
        csv_file_path = 'detected_objects.csv'
        
        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        print(f"Detected objects and their counts have been saved to '{csv_file_path}'.")
        