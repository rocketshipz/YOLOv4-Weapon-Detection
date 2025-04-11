import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


weights_path = 'yolov4.weights'  # Path to YOLO weights
config_path = 'yolov4.cfg'  # Path to YOLO config
class_file = 'coco.names'  # Path to COCO class labels

net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

fig, ax = plt.subplots()
img_plot = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
plt.axis('off')

# Load class labels
with open(class_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define weapon-related classes
weapon_classes = ["knife", "gun"]

# Initialize video capture
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open video device")
    exit()

def detect_objects(frame):
    height, width, _ = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Process detections
    class_ids = []
    confidences = []
    boxes = []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected, get the box and draw it
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to eliminate weak boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))

            # Check if detected class is a weapon
            global robot_status
            if label in weapon_classes:
                # Use red color for weapons
                color = (0, 0, 255)
                alert_text = f"ALERT! Weapon Detected: {label}"
                cv2.putText(frame, alert_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print(alert_text)  # Print alert to the console
                
            else:
                # Use green color for other objects
                
                color = (0, 255, 0)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def update_plot(i):
    ret, frame = video_capture.read()
    if not ret:
        return [img_plot]
    frame = detect_objects(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_plot.set_data(frame_rgb)
    return [img_plot]   

ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=True)
plt.show()

video_capture.release()