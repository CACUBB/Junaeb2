#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import cv2
import numpy as np
import urllib.request
import os
import ultralytics
from ultralytics import YOLO


# In[2]:


with open("COCO.txt", "r") as f:
    class_names = f.read().splitlines()


# In[3]:


#torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])


# In[4]:


model = YOLO('yolov8n.pt')


# In[ ]:


def predict_yolo(frame):
    results = model(frame, verbose=False)

    # Obtener las 5 detecciones con mayor confianza
    detections = results[0].boxes.data.cpu().numpy()
    detections = sorted(detections, key=lambda x: x[4], reverse=True)[:5]  # Ordenar por confianza

    return detections

# Captura de video desde la cámara
webcam = cv2.VideoCapture(0)

while True:
    check, frame = webcam.read()
    if not check:
        break

    detections = predict_yolo(frame)

    # Dibujar los bounding boxes y etiquetas
    for (x1, y1, x2, y2, conf, cls) in detections:
        if  conf > 0.5:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f"{class_names[int(cls)]} ({conf*100:.2f}%)"  # Usar el nombre de la clase
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow("YOLOv11 Detection", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




