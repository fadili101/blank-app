import streamlit as st
import cv2
import imutils
import numpy as np
from PIL import Image
import io
import random
import base64

# Afficher un logo au début de l'application
st.set_page_config(page_title="Mon Application", page_icon=":star:", layout="wide")
# Créer une disposition avec colonnes
# Utiliser HTML pour insérer une image avec une largeur de 100%
# Charger une image locale
image_path = "banner.jpg"

# Convertir l'image en base64
with open(image_path, "rb") as file:
    img_base64 = base64.b64encode(file.read()).decode()

# Afficher l'image avec largeur 100%
st.markdown(
    f"""
    <style>
    .logo-container {{
        display: flex;
        justify-content: center;
    }}
    .logo-container img {{
        max-width: 100%;
        height: auto;
    }}
    </style>
    <div class="logo-container">
        <img src="data:image/jpg;base64,{img_base64}" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)
# Ajouter un logo
#st.image("banner.jpg", width="100%")  # Chemin vers le logo (local ou URL)

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
    
    # Redimensionner l'image et préparer le blob pour la détection
    image_resized = cv2.resize(image, (300, 300))
    blob = cv2.dnn.blobFromImage(image_resized, 1.0, (300, 300), (104.0, 177.0, 123.0))
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    
    # Liste pour stocker les objets détectés valides
    valid_detections = []

    for i in range(0, detections.shape[2]):
        # Extraire la confiance associée à la détection
        confidence = detections[0, 0, i, 2]

        # Filtrer les détections faibles (seuil de 0.2)
        if confidence > 0.2:
            # Calculer les coordonnées du rectangle de détection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ajouter la détection valide à la liste
            valid_detections.append((startX, startY, endX, endY, confidence))

    # Dessiner tous les objets en rouge et celui sélectionné en vert
    if valid_detections:
        # Choisir un objet aléatoire parmi les détections valides
        chosen_detection = random.choice(valid_detections)
        (startX, startY, endX, endY, confidence) = chosen_detection

        # Dessiner les rectangles rouges pour tous les objets détectés
        for detection in valid_detections:
            (startX, startY, endX, endY, confidence) = detection
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)  # Rouge
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # Dessiner le rectangle vert autour de l'objet choisi
        (startX, startY, endX, endY, confidence) = chosen_detection
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)  # Vert
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        print(f"Objet choisi avec confiance : {confidence:.2f}")
    else:
        print("Aucun objet valide détecté.")

    # Afficher l'image avec les objets détectés
    st.image(image)
