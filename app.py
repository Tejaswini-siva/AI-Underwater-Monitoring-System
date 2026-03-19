import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO("yolov8n.pt")

st.set_page_config(page_title="AI Underwater Monitoring", layout="wide")

st.title(" AI Underwater Monitoring System")
st.markdown("Detect marine life and underwater objects using AI")

# Sidebar
st.sidebar.title("Settings")
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.4)

# CLAHE Enhancement
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return enhanced


input_type = st.radio("Select Input Type", ["Image", "Video", "Live Camera"])
uploaded_file = st.file_uploader("Upload File", type=["jpg","jpeg","png","mp4","avi"])

detection_counts = {}

# IMAGE
if input_type == "Image" and uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    enhanced = apply_clahe(image)
    results = model(enhanced, conf=confidence)

    annotated = results[0].plot()

    st.image(enhanced, caption="Enhanced Image", channels="BGR")
    st.image(annotated, caption="Detection Results", channels="BGR")

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        detection_counts[label] = detection_counts.get(label, 0) + 1


# VIDEO
if input_type == "Video" and uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        enhanced = apply_clahe(frame)
        results = model(enhanced, conf=confidence)
        annotated = results[0].plot()

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detection_counts[label] = detection_counts.get(label, 0) + 1

        stframe.image(annotated, channels="BGR")

    cap.release()


# LIVE CAMERA
if input_type == "Live Camera":
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Camera not working")
            break

        enhanced = apply_clahe(frame)
        results = model(enhanced, conf=confidence)
        annotated = results[0].plot()

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detection_counts[label] = detection_counts.get(label, 0) + 1

        FRAME_WINDOW.image(annotated, channels="BGR")

    camera.release()


# STATISTICS
if detection_counts:
    st.subheader(" Detection Statistics")

    df = pd.DataFrame(list(detection_counts.items()), columns=["Object", "Count"])
    st.dataframe(df)

    fig, ax = plt.subplots()
    ax.bar(df["Object"], df["Count"])
    plt.xticks(rotation=45)
    st.pyplot(fig)
