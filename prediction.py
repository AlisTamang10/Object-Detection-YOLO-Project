import cv2  # Import OpenCV library for computer vision tasks.
import numpy as np  # Import NumPy library for numerical operations.
import os  # Import os module for interacting with the operating system.
import yaml  # Import YAML library for reading YAML files.
from yaml.loader import SafeLoader  # Import SafeLoader from YAML for safe loading of YAML files.

# Load YAML file containing class labels.
with open(r'C:\Users\alist\OneDrive\Desktop\YOLO project\1_datapreparation\data.yaml', mode='r') as f:
    data_yaml = yaml.load(f, Loader=SafeLoader)  # Load YAML data.
labels = data_yaml['names']  # Extract class labels from YAML data.

# Load YOLO model from ONNX format.
yolo = cv2.dnn.readNetFromONNX(r'C:\Users\alist\OneDrive\Desktop\YOLO project\prediction\Model13\weights\best.onnx')
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Set backend for YOLO.
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Set target for YOLO.

# Open video file for object detection.
video_path = r'C:\Users\alist\OneDrive\Desktop\YOLO project\prediction\video.mp4'  # Video file path.
cap = cv2.VideoCapture(video_path)  # Open video capture object.

while True:
    ret, frame = cap.read()  # Read a frame from the video.
    
    if not ret:  # If no frame is grabbed (end of video), break out of the loop.
        break
    
    # Preprocess frame and perform YOLO object detection.
    max_rc = max(frame.shape[:2])  # Get maximum dimension of the frame.
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)  # Create a black canvas.
    input_image[0:frame.shape[0], 0:frame.shape[1]] = frame  # Paste frame onto the canvas.
    INPUT_WH_YOLO = 640  # Define input size for YOLO model.
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)  # Preprocess image for YOLO.
    yolo.setInput(blob)  # Set input for YOLO model.
    preds = yolo.forward()  # Perform forward pass and get predictions.

    # Process detections and draw bounding boxes.
    detections = preds[0]  # Extract detections from predictions.
    boxes = []  # Initialize list to store bounding boxes.
    confidences = []  # Initialize list to store confidences.
    classes = []  # Initialize list to store class indices.

    image_w, image_h = input_image.shape[:2]  # Get width and height of input image.
    x_factor = image_w / INPUT_WH_YOLO  # Calculate scaling factor for width.
    y_factor = image_h / INPUT_WH_YOLO  # Calculate scaling factor for height.

    for i in range(len(detections)):  # Iterate through detections.
        row = detections[i]  # Extract detection information.
        confidence = row[4]  # Extract confidence score.
        if confidence > 0.4:  # Filter detections based on confidence threshold.
            class_score = row[5:].max()  # Get maximum class score.
            class_id = row[5:].argmax()  # Get index of the class with maximum score.
            if class_score > 0.25:  # Filter detections based on class score threshold.
                cx, cy, w, h = row[0:4]  # Extract bounding box coordinates.
                left = int((cx - 0.5 * w) * x_factor)  # Calculate left coordinate of bounding box.
                top = int((cy - 0.5 * h) * y_factor)  # Calculate top coordinate of bounding box.
                width = int(w * x_factor)  # Calculate width of bounding box.
                height = int(h * y_factor)  # Calculate height of bounding box.
                box = np.array([left, top, left + width, top + height])  # Create bounding box array.

                confidences.append(class_score)  # Append confidence score to list.
                boxes.append(box)  # Append bounding box coordinates to list.
                classes.append(class_id)  # Append class index to list.

    boxes_np = np.array(boxes)  # Convert bounding box list to NumPy array.
    confidences_np = np.array(confidences)  # Convert confidence score list to NumPy array.

    # Perform non-maximum suppression to remove redundant bounding boxes.
    output = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
    if len(output) > 0:  # If there are detections after NMS.
        index = output.flatten()  # Flatten the output array.
    else:
        index = np.empty((0,), dtype=int)  # If no detections after NMS, create an empty array.

    # Draw bounding boxes and labels on the frame.
    for ind in index:  # Iterate through indices of selected bounding boxes.
        x, y, w, h = boxes_np[ind]  # Extract bounding box coordinates.
        bb_conf = int(confidences_np[ind] * 100)  # Calculate bounding box confidence.
        class_id = classes[ind]  # Get class index.
        class_name = labels[class_id]  # Get class name corresponding to index.

        text = f'{class_name}: {bb_conf}%'  # Create label text.

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box.
        cv2.rectangle(frame, (x, y - 30), (x + w, y), (255, 255, 255), -1)  # Draw rectangle for label background.
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)  # Add label text.

    # Display the frame with the bounding boxes.
    cv2.imshow('YOLO Object Detection', frame)
    
    # Check for 'q' key press to exit the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close windows.
cap.release()  # Release the video capture object.
cv2.destroyAllWindows()  # Close all OpenCV windows.