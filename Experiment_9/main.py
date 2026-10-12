import cv2
from ultralytics import YOLO
import os
# 1. Load YOLOv8 pre-trained mode
model = YOLO("yolov8n.pt")
# 2. Input imag
image_path = "imag9.jpg"

if not os.path.exists(image_path):
    print("Error: image.jpg not found!")
    exit()

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to read image.jpg")
    exit()

# 3. Perform object detectio
results = model(image)
# 4. Draw detection
output = image.copy()

for result in results:

    boxes = result.boxes

    for box in boxes:

        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Confidence score
        confidence = float(box.conf[0])

        # Class ID
        class_id = int(box.cls[0])

        # Class name
        class_name = model.names[class_id]

        # Draw bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"{class_name} {confidence:.2f}"

        # Text background
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        cv2.rectangle(
            output,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            (0, 255, 0),
            -1
        )

        # Text
        cv2.putText(
            output,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
    
# 5. Save output imag
output_path = "output_detection.jpg"

cv2.imwrite(output_path, output)
# 6. Display resul
cv2.imshow("YOLOv8 Object Detection", output)

print("-----------------------------------")
print("Object detection completed!")
print("Input image  :", image_path)
print("Output image :", output_path)
print("-----------------------------------")

cv2.waitKey(0)
cv2.destroyAllWindows()