import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from scipy.spatial import distance as dist

# Load YOLO model
yolo_model = YOLO('/Users/sushanthkesamreddy/Documents/ImgSegmentation/LaysDataset/best.pt')

# Class labels to keep
class_labels = ['Bar.B.Q', 'Barbecue', 'Cheddar - Sour Cream', 'Cheddar Jalapeno', 'Classic', 'Dill Pickle', 'Flamin Hot', 'French Cheese', 'Honey Barbecue', 'Lays', 'Masala', 'PAPRIKA', 'Poppables', 'Salt - Vinegar', 'Salted', 'Sour Cream - Onion', 'Sweet Southern Heat Barbecue', 'Wavy', 'Yogurt - Herb']

# Price mapping for specific items
price_mapping = {
    'Pantene': 180,
    'coca cola': 40,
    'ariel-powder-machine': 300,
    'Classic-Salted': 30,
    'Sour Cream Onion': 30,
    'silk': 80,
    'Amul-Milk-Full-Cream-500ml': 30
}

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Variables for tracking detected objects and their classes
object_centroids = {}
object_classes = {}
next_object_id = 0
total_money = 0  # Initialize total money

# Function to match new detections with existing tracked objects
def match_detections(new_centroids, object_centroids, threshold=30):
    if len(new_centroids) == 0 or len(object_centroids) == 0:
        return {}

    object_ids = list(object_centroids.keys())
    object_centroid_values = list(object_centroids.values())

    new_centroids = np.array(new_centroids)
    object_centroid_values = np.array(object_centroid_values)

    D = dist.cdist(new_centroids, object_centroid_values)

    matched_pairs = {}
    for i, row in enumerate(D):
        matched_index = row.argmin()
        if row[matched_index] < threshold:
            matched_pairs[i] = object_ids[matched_index]

    return matched_pairs

# Initialize FPS calculation
prev_time = 0
CONFIDENCE_THRESHOLD = 0.6  # Adjusted confidence threshold

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Get YOLO model predictions
    results = yolo_model(frame)

    # List to hold new object centroids and their class labels
    new_centroids = []
    new_classes = []

    # Process YOLO results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0]  # Box coordinates
            conf = box.conf[0]  # Confidence score
            cls = box.cls[0]    # Class index

            if conf > CONFIDENCE_THRESHOLD:  # Confidence threshold
                label = class_labels[int(cls)]

                # Only process the specified classes
                if label in class_labels:
                    x1, y1, x2, y2 = map(int, xyxy)
                    centroid = ((x1 + x2) // 2, (y1 + y2) // 2)  # Calculate centroid
                    new_centroids.append(centroid)
                    new_classes.append(int(cls))

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw class label
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Match new detections to existing objects
    matched = match_detections(new_centroids, object_centroids)

    # Update existing objects or add new ones
    for i, (centroid, cls) in enumerate(zip(new_centroids, new_classes)):
        if i in matched.values():
            obj_id = matched[i]
            # Update the position of the existing object
            object_centroids[obj_id] = centroid
        else:
            object_centroids[next_object_id] = centroid
            object_classes[next_object_id] = cls
            next_object_id += 1

            # Update total money based on detected item
            item_label = class_labels[int(cls)]
            if item_label in price_mapping:
                total_money += price_mapping[item_label]

    # Count the number of detected objects per class
    class_counts = {label: 0 for label in class_labels}
    for obj_id in object_classes.values():
        class_counts[class_labels[obj_id]] += 1

    # Get current time for FPS calculation
    current_time = time.time()

    # Calculate FPS
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS on the frame
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display object counts per class
    y_offset = 60
    for class_label, count in class_counts.items():
        if count > 0:  # Display only classes with detected objects
            object_count_text = f"{class_label}: {count}"
            cv2.putText(frame, object_count_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            y_offset += 30

    # Display total money on the frame
    total_money_text = f"Total Money: ${total_money}"
    cv2.putText(frame, total_money_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show the live frame
    cv2.imshow('Dynamic Object Detection with YOLO', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()