import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Detector de Objetos - YOLOv8 TFLite", layout="centered")

st.title("📸 Detector de Objetos con YOLOv8n.tflite")
st.write("Sube una imagen y detectaremos personas, autos, bicicletas y más.")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Imagen original", use_column_width=True)

    # Cargar modelo TFLite
    interpreter = tf.lite.Interpreter(model_path="yolov8n_float32.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocesamiento
    img_resized = cv2.resize(img_array, (640, 640)) / 255.0
    input_data = np.expand_dims(img_resized.astype(np.float32), axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    results = interpreter.get_tensor(output_details[0]['index'])[0]

    # Dibujar resultados sobre la imagen
    output_img = img_array.copy()
    h, w, _ = img_array.shape
    for res in results:
        conf = res[4]
        if conf > 0.4:
            class_id = int(res[5])
            x, y, bw, bh = res[0:4]
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(output_img, f"ID:{class_id} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    st.image(output_img, caption="Resultado con detecciones", use_column_width=True)