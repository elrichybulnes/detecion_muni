import streamlit as st
import cv2
from detector import detectar_objetos
from collections import defaultdict
import tempfile

CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

st.set_page_config(layout="wide")
st.title("Demo de Detección de Objetos - Municipio")

uploaded_file = st.file_uploader("Subí un video (MP4)", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    conteo_total = defaultdict(int)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        objetos = detectar_objetos(frame)

        for obj in objetos:
            cls_id = obj["class_id"]
            if cls_id < len(CLASSES):
                clase = CLASSES[cls_id]
                conteo_total[clase] += 1
                x1, y1, x2, y2 = obj["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, clase, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
    st.subheader("Conteo total por clase detectada")
    st.json(dict(conteo_total))

# App actualizada para forzar redeploy
