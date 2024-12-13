import streamlit as st
import cv2
import numpy as np
import imutils
import random
from PIL import Image

# Titre de l'application
st.title("Détection d'objets avec OpenCV")

# Chargement du modèle
@st.cache_resource
def load_model():
    prototxt = 'deploy.prototxt'
    model = 'res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    return net

# Charger le modèle
net = load_model()

# Charger une image depuis le système ou prendre une photo
uploaded_file = st.file_uploader("Chargez une image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = imutils.resize(image, width=400)
    (h, w) = image.shape[:2]

    # Prétraitement pour le modèle
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Détections valides
    valid_detections = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.25:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            valid_detections.append((startX, startY, endX, endY, confidence))

    # Réorganiser les détections aléatoirement
    if valid_detections:
        random.shuffle(valid_detections)
        chosen_detection = valid_detections[0]  # Prenez la première après mélange
        (startX, startY, endX, endY, confidence) = chosen_detection

        # Dessiner la boîte
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        st.image(image, caption=f"Objet choisi avec une confiance de {confidence:.2f}", use_column_width=True)
    else:
        st.write("Aucun objet valide détecté.")
