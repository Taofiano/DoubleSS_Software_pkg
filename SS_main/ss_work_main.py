import cv2
import serial
import requests
import time
import tkinter as tk
from tkinter import Label, Canvas, Frame
from PIL import Image, ImageTk
import threading

# Serial port setup (for conveyor belt control)
ser = serial.Serial("/dev/ttyACM0", 9600)

# Vision AI API URL and settings
VISION_API_URL = "####"
params = {
    "min_confidence": 0.4,
    "base_model": "YOLOv6-L"
}

# Class mapping and colors
class_mapping = {
    5: "RASBERRY PICO",
    3: "HOLE",
    1: "BOOTSEL",
    4: "OSCILLATOR",
    6: "USB",
    2: "CHIPSET"
}

class_colors = {
    "RASBERRY PICO": (255, 102, 102),  # light red
    "HOLE": (255, 178, 102),           # orange
    "BOOTSEL": (178, 255, 102),        # light green
    "OSCILLATOR": (102, 178, 255),     # light blue
    "USB": (204, 153, 255),            # light purple
    "CHIPSET": (192, 192, 192)         # grey
}

# API call to process frame
def process_frame(frame):
    if frame is None or frame.size == 0:
        return None
    _, img_encoded = cv2.imencode(".jpg", frame)
    img_bytes = img_encoded.tobytes()
    try:
        response = requests.post(
            url=VISION_API_URL,
            params=params,
            files={"file": ("image.jpg", img_bytes, "image/jpeg")}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

# Function to draw bounding boxes and count detected classes
def draw_predictions(frame, predictions):
    counts = {class_name: 0 for class_name in class_mapping.values()}
    for obj in predictions.get("objects", []):
        class_number = obj["class_number"]
        class_name = class_mapping.get(class_number, "Unknown")
        bbox = obj["bbox"]
        confidence = obj["confidence"]
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        color = class_colors.get(class_name, (255, 255, 255))  # Default to white
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Label the box
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Increment count for this class
        counts[class_name] += 1
    return counts

# Function to show missing parts for defective products
def show_missing_parts(counts):
    missing_parts = []
    required_parts = {
        "RASBERRY PICO": 1,
        "HOLE": 4,
        "BOOTSEL": 1,
        "OSCILLATOR": 1,
        "USB": 1,
        "CHIPSET": 1
    }
    for part, required_count in required_parts.items():
        detected_count = counts.get(part, 0)
        if detected_count < required_count:
            missing_parts.append(f"{part}: {required_count - detected_count} missing")
    return missing_parts

# UI Setup
root = tk.Tk()
root.title("Conveyor Belt Vision AI")
root.geometry("900x600")

# Green Bulb for good products
good_bulb = Label(root, width=20, height=3, bg="gray", text="Good Product\nCount: 0", font=("Arial", 12))
good_bulb.grid(row=0, column=0, padx=10, pady=10)

# Orange Bulb for defective products
defective_bulb = Label(root, width=20, height=3, bg="gray", text="Defective Product\nCount: 0", font=("Arial", 12))
defective_bulb.grid(row=1, column=0, padx=10, pady=10)

# Emergency Stop Button
def emergency_stop():
    ser.write(b"0")  # Stop the conveyor belt
    print("Emergency Stop Activated.")
    root.quit()  # Quit the Tkinter UI

stop_button = tk.Button(
    root,
    text="Emergency Stop",
    command=emergency_stop,
    bg="red",
    fg="white",
    width=10,
    height=2,
    font=("Arial", 12),
    relief="flat"
)
stop_button.grid(row=2, column=0, padx=10, pady=10)
stop_button.config(borderwidth=2, highlightbackground="black", highlightthickness=2)

# Log Area (Right side)
log_frame = tk.Frame(root, width=400, height=600, bg="white")
log_frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="n")
log_frame.grid_propagate(False)

# List to hold log entries
log_entries = []

# Update Log
def update_log(img, message):
    global log_entries
    if len(log_entries) >= 5:
        oldest_entry = log_entries.pop(0)
        oldest_entry[0].destroy()
        oldest_entry[1].destroy()

    img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    img_label = Label(log_frame, image=img_tk)
    img_label.image = img_tk
    img_label.pack(side="top", padx=5, pady=5)

    msg_label = Label(log_frame, text=message, bg="white", font=("Arial", 10), anchor="w", justify="left")
    msg_label.pack(side="top", padx=5, pady=5)

    log_entries.append((img_label, msg_label))

# Video Feed Processing
def start_processing():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera Error")
        return
    time.sleep(2)  # Camera warm-up

    good_count = 0
    defective_count = 0
    try:
        while True:
            data = ser.read()
            # 시리얼 데이터가 "0"인 경우에만 프레임 캡처
            if data == b"0":
                print("Conveyor stopped. Processing frame...")
                # 버퍼 비우기: 연속적으로 몇 번 읽어서 최신 프레임 확보
                for _ in range(5):
                    ret, frame = cam.read()
                if not ret:
                    print("Failed to capture frame.")
                    continue
                # 객체 인식 수행
                predictions = process_frame(frame)
                if predictions:
                    counts = draw_predictions(frame, predictions)
                    missing_parts = show_missing_parts(counts)
                    # 판정
                    if not missing_parts:
                        good_count += 1
                        good_bulb.configure(bg="green", text=f"Good Product\nCount: {good_count}")
                        log_message = "Status: Good Product"
                    else:
                        defective_count += 1
                        defective_bulb.configure(bg="orange", text=f"Defective Product\nCount: {defective_count}")
                        log_message = "Status: Defective Product\n" + "\n".join(missing_parts)

                    # Update log with image and message
                    resized_frame = cv2.resize(frame, (200, 150))
                    update_log(resized_frame, log_message)

                    # Conveyor belt restart
                    ser.write(b"1")  # Resume conveyor belt
    finally:
        cam.release()
        cv2.destroyAllWindows()

# Start processing in a separate thread
process_thread = threading.Thread(target=start_processing, daemon=True)
process_thread.start()

# Start the UI loop
root.mainloop()
