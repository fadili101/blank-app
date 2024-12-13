import streamlit as st
import cv2
import imutils
import numpy as np
from PIL import Image
import io
import random

# Active la caméra si la case est cochée
enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)

if picture:
    # Afficher l'image capturée
    

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
    
    # Liste pour stocker les objets détectés valides
    valid_detections = []

    for i in range(0, detections.shape[2]):
        # Extraire la confiance associée à la détection
        confidence = detections[0, 0, i, 2]

        # Filtrer les détections faibles (seuil de 0.5)
        if confidence > 0.5:
            # Calculer les coordonnées du rectangle de détection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ajouter la détection valide à la liste
            valid_detections.append((startX, startY, endX, endY, confidence))

    # Choisir un seul objet parmi les détections valides
    if valid_detections:
        chosen_detection = random.choice(valid_detections)
        (startX, startY, endX, endY, confidence) = chosen_detection

        # Dessiner le rectangle autour de l'objet choisi
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        print(f"Objet choisi avec confiance : {confidence:.2f}")
    else:
        print("Aucun objet valide détecté.")

    # Afficher l'image avec l'objet choisi
    st.image(image)
