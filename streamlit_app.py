import streamlit as st
import cv2
import imutils
import numpy as np
from PIL import Image
import io

# Active la caméra si la case est cochée
enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)

if picture:
    # Afficher l'image capturée
    st.image(picture)

    # Convertir l'image Streamlit en un format OpenCV (numpy array)
    image_bytes = picture.getvalue()  # Récupérer les octets de l'image
    image = Image.open(io.BytesIO(image_bytes))  # Charger l'image depuis les octets
    image = np.array(image)  # Convertir en numpy array

    # Redimensionner l'image pour avoir une largeur maximale de 400 pixels
    image = imutils.resize(image, width=400)
    (h, w) = image.shape[:2]
    print("[INFO] loading model...")

    # Charger le modèle de détection d'objets
    prototxt = 'deploy.prototxt'
    model = 'res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Afficher les dimensions de l'image
    print(f"Image dimensions: {h}x{w}")
