from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog

# LOAD TRAINED MODEL

model = YOLO("yolov8n.pt")

# ---------------- VIDEO DETECTION ---------------- #

def detect_video():

    path = filedialog.askopenfilename()

    if path == "":
        return

    cap = cv2.VideoCapture(path)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, conf=0.5)

        boxes = results[0].boxes
        pothole_count = len(boxes)

        frame = results[0].plot()

        # -------- SIZE CALCULATION --------
        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = x2 - x1
            height = y2 - y1

            approx_cm = int(width * 0.2)

            cv2.putText(frame,
                        f"{width}px (~{approx_cm}cm)",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255,255,255),
                        2)

        # -------- RQI CALCULATION --------
        rqi_value = max(0, 100 - pothole_count * 12)

        if rqi_value >= 80:
            rqi = "Excellent"
            traffic = "Low Traffic"
            speed_now = 60

        elif rqi_value >= 60:
            rqi = "Good"
            traffic = "Moderate Traffic"
            speed_now = 45

        elif rqi_value >= 40:
            rqi = "Poor"
            traffic = "High Traffic"
            speed_now = 30

        else:
            rqi = "Very Poor"
            traffic = "Very High Traffic"
            speed_now = 15

        speed_repaired = 60

        # -------- DISPLAY INFO --------

        cv2.putText(frame, f"Potholes Detected: {pothole_count}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(frame, f"RQI: {rqi_value} ({rqi})", (20,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(frame, f"Traffic Level: {traffic}", (20,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

        cv2.putText(frame, f"Recommended Speed: {speed_now} km/h", (20,130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"If Road Repaired Speed: {speed_repaired} km/h", (20,160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow("Pothole Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- IMAGE DETECTION ---------------- #

def detect_image():

    path = filedialog.askopenfilename()

    if path == "":
        return

    frame = cv2.imread(path)

    results = model(frame, conf=0.25)

    boxes = results[0].boxes
    pothole_count = len(boxes)

    frame = results[0].plot()

    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        width = x2 - x1
        height = y2 - y1

        approx_cm = int(width * 0.2)

        cv2.putText(frame,
                    f"{width}px (~{approx_cm}cm)",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    2)

    # -------- RQI CALCULATION --------

    rqi_value = max(0, 100 - pothole_count * 12)

    if rqi_value >= 80:
        rqi = "Excellent"
        traffic = "Low Traffic"
        speed_now = 60

    elif rqi_value >= 60:
        rqi = "Good"
        traffic = "Moderate Traffic"
        speed_now = 45

    elif rqi_value >= 40:
        rqi = "Poor"
        traffic = "High Traffic"
        speed_now = 30

    else:
        rqi = "Very Poor"
        traffic = "Very High Traffic"
        speed_now = 15

    speed_repaired = 60

    cv2.putText(frame, f"Potholes Detected: {pothole_count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"RQI: {rqi_value} ({rqi})", (20,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Traffic Level: {traffic}", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

    cv2.putText(frame, f"Recommended Speed: {speed_now} km/h", (20,130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"If Road Repaired Speed: {speed_repaired} km/h", (20,160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("Pothole Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------- GUI ---------------- #

root = tk.Tk()

root.title("AI Pothole Detection System")
root.geometry("500x400")
root.configure(bg="black")


title = tk.Label(
    root,
    text="AI POTHOLE DETECTION",
    bg="black",
    fg="white",
    font=("Arial", 22, "bold")
)

title.pack(pady=40)


btn_video = tk.Button(
    root,
    text="Detect From Video",
    command=detect_video,
    bg="white",
    fg="black",
    font=("Arial", 12, "bold"),
    width=20,
    height=2
)

btn_video.pack(pady=10)


btn_image = tk.Button(
    root,
    text="Detect From Image",
    command=detect_image,
    bg="white",
    fg="black",
    font=("Arial", 12, "bold"),
    width=20,
    height=2
)

btn_image.pack(pady=10)


btn_exit = tk.Button(
    root,
    text="Exit",
    command=root.destroy,
    bg="red",
    fg="white",
    font=("Arial", 12, "bold"),
    width=20,
    height=2
)

btn_exit.pack(pady=20)


root.mainloop()