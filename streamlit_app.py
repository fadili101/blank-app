import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Titre de l'application
st.title("Détection d'objets en direct avec Webcam")

# Chargement du modèle
@st.cache_resource
def load_model():
    try:
        prototxt = 'deploy.prototxt'
        model = 'res10_300x300_ssd_iter_140000.caffemodel'
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        return net
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Charger le modèle
net = load_model()
if net is None:
    st.stop()

# Classe pour la détection en direct
class ObjectDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.net = net

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        (h, w) = img.shape[:2]

        # Prétraitement pour le modèle
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        # Boucle sur les détections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                text = f"{confidence:.2f}"
                # Dessiner la boîte et le texte
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Ajouter le streamer
webrtc_ctx = webrtc_streamer(
    key="object-detection",
    video_transformer_factory=ObjectDetectionTransformer,
    media_stream_constraints={"video": True, "audio": False},  # Désactiver l'audio
)

# Vérifier si la webcam est activée
if webrtc_ctx.state.playing:
    st.write("La webcam est activée. Attendez quelques secondes pour voir le flux vidéo.")
else:
    st.warning("La webcam ne semble pas être activée. Assurez-vous que votre caméra est connectée et autorisée dans le navigateur.")
