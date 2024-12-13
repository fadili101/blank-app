import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Titre de l'application
st.title("Détection d'objets en direct avec Webcam")

# Chargement du modèle
@st.cache_resource
def load_model():
    prototxt = 'deploy.prototxt'
    model = 'res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    return net

net = load_model()

# Classe pour la détection en direct
class ObjectDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.net = net

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        (h, w) = img.shape[:2]

        # Prétraitement
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
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img

# Ajouter le streamer
webrtc_streamer(key="object-detection", video_transformer_factory=ObjectDetectionTransformer)
