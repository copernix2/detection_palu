import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

# Configuration de la page
st.set_page_config(page_title="Détection du Paludisme", page_icon="🧬", layout="centered")

# Chargement du modèle
MODEL_PATH = "model2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Titre
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>🧪 Détection de Paludisme</h1>
    <h5 style='text-align: center; color: #6c757d;'>Téléversez une image de cellule sanguine pour détecter une infection paludique</h5>
    <hr style="border-top: 1px solid #bbb;">
    """,
    unsafe_allow_html=True
)

# Upload de l'image
uploaded_file = st.file_uploader("📤 Veuillez téléverser une image (JPG, JPEG ou PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Chargement et affichage de l'image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='🖼️ Image téléversée', use_column_width=True)

    # Prétraitement
    image = image.resize((50, 50))  # Adapter selon le modèle entraîné
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prédiction
    prediction = model.predict(image_array)
    prob = prediction[0][0]

    # Résultat
    st.markdown("---")
    st.subheader("📊 Résultat de la prédiction :")

    if prob >= 0.5:
        st.error(f"🔴 **Cellule infectée**\n\nProbabilité : `{prob:.2%}`")
    else:
        st.success(f"🟢 **Cellule non infectée**\n\nProbabilité : `{(1 - prob):.2%}`")
