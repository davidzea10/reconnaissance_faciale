"""
Reconnaissance faciale — Application Streamlit
Modèles chargés depuis les .joblib (SVM + scaler + PCA + noms).
"""

from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image

# --- Chemins (même dossier que app.py) ---
APP_DIR = Path(__file__).resolve().parent
PATH_MODEL = APP_DIR / "model_svm_facial.joblib"
PATH_SCALER = APP_DIR / "scaler_facial.joblib"
PATH_PCA = APP_DIR / "pca_facial.joblib"
PATH_PERSONNES = APP_DIR / "personnes_facial.joblib"

# Taille des images utilisée à l'entraînement
IMG_SIZE = (128, 128)


# --- Étape 2 : chargement des modèles (une seule fois, mis en cache) ---
@st.cache_resource
def charger_modeles():
    """Charge le modèle SVM, le scaler, la PCA et la liste des noms."""
    model = joblib.load(PATH_MODEL)
    scaler = joblib.load(PATH_SCALER)
    pca = joblib.load(PATH_PCA)
    personnes = joblib.load(PATH_PERSONNES)
    return model, scaler, pca, personnes


def predire_visage(image, model, scaler, pca, personnes):
    """
    Prédit la personne sur une image.
    image : fichier uploadé (file-like) ou chemin (str/Path) ou PIL.Image.
    Retourne le nom prédit (str).
    """
    if hasattr(image, "read"):
        img = Image.open(image).convert("L")
    elif isinstance(image, (str, Path)):
        img = Image.open(image).convert("L")
    elif isinstance(image, Image.Image):
        img = image.convert("L")
    else:
        raise TypeError("image doit être un fichier, un chemin ou une PIL.Image")
    img = img.resize(IMG_SIZE)
    x = np.array(img).reshape(1, -1)
    x = scaler.transform(x)
    x_pca = pca.transform(x)
    label = model.predict(x_pca)[0]
    return personnes[label]


# --- Chargement au démarrage ---
model, scaler, pca, personnes = charger_modeles()

# --- Page Streamlit (étape 3 + caméra) ---
st.set_page_config(page_title="Reconnaissance faciale", layout="centered")
st.title("Reconnaissance faciale — SVM")
st.caption("Uploadez une image ou prenez une photo avec la caméra.")

tab_upload, tab_camera = st.tabs(["📁 Uploader une image", "📷 Prendre une photo"])

with tab_upload:
    uploaded = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"], key="upload")
    if uploaded is not None:
        col1, col2 = st.columns(2)
        with col1:
            img_display = Image.open(uploaded)
            st.image(img_display, caption="Image chargée", use_container_width=True)
        with col2:
            if st.button("Prédire", type="primary", key="btn_upload"):
                with st.spinner("Prédiction en cours…"):
                    nom = predire_visage(img_display, model, scaler, pca, personnes)
                st.success(f"Personne reconnue : **{nom}**")
    else:
        st.info("Choisissez un fichier image (JPG, PNG).")

with tab_camera:
    cam_photo = st.camera_input("Prendre une photo avec la caméra", key="camera")
    if cam_photo is not None:
        img_cam = Image.open(cam_photo)
        st.image(img_cam, caption="Photo prise", use_container_width=True)
        if st.button("Prédire", type="primary", key="btn_camera"):
            with st.spinner("Prédiction en cours…"):
                nom = predire_visage(img_cam, model, scaler, pca, personnes)
            st.success(f"Personne reconnue : **{nom}**")
    else:
        st.info("Autorisez l'accès à la caméra et prenez une photo.")
