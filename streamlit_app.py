import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Titre de l'application
st.title("Détection d'objets avec OpenCV et Webcam")

# Chargement du modèle
@st.cache_resource
def load_model():
    prototxt = 'deploy.prototxt'
    model = 'res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    return net

# Charger le modèle
net = load_model()

# Activer la webcam
start_webcam = st.button("Activer la webcam")
capture_image = st.button("Capturer une image")

# Affichage vidéo avec OpenCV
if start_webcam:
    st.write("**Appuyez sur 'Capturer une image' pour analyser une image.**")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Erreur: Impossible d'accéder à la webcam.")
            break

        # Afficher le flux vidéo dans l'application Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB", caption="Flux Webcam", use_column_width=True)

        # Vérifier si le bouton pour capturer est cliqué
        if capture_image:
            image = frame.copy()
            cap.release()
            cv2.destroyAllWindows()
            break

    if capture_image and 'image' in locals():
        # Prétraitement pour le modèle
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # Détections valides
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.25:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Dessiner la boîte autour de l'objet détecté
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Afficher l'image traitée
        st.image(image, caption="Image capturée avec détections", use_column_width=True)
else:
    st.write("Appuyez sur 'Activer la webcam' pour commencer.")