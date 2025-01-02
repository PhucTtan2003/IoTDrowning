from ultralytics import YOLO
import cv2
import numpy as np
import pygame
import time

# Load YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO("D:/KhungAnh/best.pt")  # Replace with your custom trained YOLOv8 model path

# Khởi tạo pygame mixer để chơi âm thanh
pygame.mixer.init()

# Bộ đếm thời gian cho các đối tượng đứng yên
motion_timers = {}  # Lưu thời gian mỗi đối tượng đứng yên

def detectDrowning(source):  # Default video source
    isDrowning = False
    cap = cv2.VideoCapture("D:/KhungAnh/KhungAnh/videos/" + source)  # Load the video

    if not cap.isOpened():
        print("Error: Cannot open video source")
        return

    frame_width = int(cap.get(3))  # Width of the video frame
    frame_height = int(cap.get(4))  # Height of the video frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLOv8 inference
        results = model.predict(source=frame, imgsz=640, conf=0.5)  # Get predictions
        detections = results[0].boxes  # Get detected bounding boxes

        # Extract bounding boxes, labels, and confidences
        bbox = []
        label = []
        conf = []
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
            cls = int(box.cls[0])  # Class ID
            confidence = float(box.conf[0])  # Confidence score
            bbox.append([int(x1), int(y1), int(x2), int(y2)])
            label.append(model.names[cls])  # Class name
            conf.append(confidence)

        # Draw bounding boxes on the frame
        for i in range(len(bbox)):
            x1, y1, x2, y2 = bbox[i]
            color = (0, 255, 0) if label[i] != 'person' else (0, 0, 255)  # Red for person
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw rectangle
            text = f"{label[i]}: {conf[i]:.2f}"  # Display label with confidence
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Kiểm tra chuyển động của đối tượng (đứng yên lâu quá)
        for i, (x1, y1, x2, y2) in enumerate(bbox):
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            obj_id = f"{x1}_{y1}_{x2}_{y2}"  # Tạo ID dựa trên bounding box

            # Kiểm tra chuyển động
            if obj_id in motion_timers:
                prev_center = motion_timers[obj_id]["center"]
                dist = np.linalg.norm(np.array(center) - np.array(prev_center))

                # Nếu khoảng cách di chuyển nhỏ hơn ngưỡng (người đứng yên)
                if dist < 10:  # Ngưỡng đứng yên (pixel)
                    motion_timers[obj_id]["time"] += 1
                else:
                    motion_timers[obj_id]["time"] = 0

                # Cảnh báo nếu thời gian đứng yên vượt quá ngưỡng
                if motion_timers[obj_id]["time"] > 50:  # Ngưỡng thời gian (khung hình)
                    isDrowning = True
            else:
                motion_timers[obj_id] = {"center": center, "time": 0}  # Lưu trữ thông tin ban đầu

        # Logic-based drowning detection: check if people are too close
        if label.count('person') > 1:
            centres = [
                [(bbox[i][0] + bbox[i][2]) / 2, (bbox[i][1] + bbox[i][3]) / 2]
                for i in range(len(bbox)) if label[i] == 'person'
            ]
            distances = [
                np.linalg.norm(np.array(centres[i]) - np.array(centres[j]))  # Calculate Euclidean distance
                for i in range(len(centres))
                for j in range(i + 1, len(centres))
            ]
            if len(distances) > 0 and min(distances) < 50:  # Threshold distance for potential drowning
                isDrowning = True

        # Hiển thị thông báo trạng thái phát hiện đuối nước
        status_text = "Drowning detected!" if isDrowning else "No drowning detected"
        color = (0, 0, 255) if isDrowning else (0, 255, 0)  # Red if drowning detected
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Phát âm thanh khi phát hiện đuối nước
        if isDrowning:
            pygame.mixer.music.load('D:/KhungAnh/KhungAnh/sound/alarm.mp3')  # Load your alarm sound file
            pygame.mixer.music.play()  # Play the sound

        # Show output frame
        cv2.imshow("YOLOv8 Drowning Detection", frame)

        # Break on "Q" key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run detection with default video
detectDrowning("drowning_002.mp4")  # Replace with the video filename you want to test
