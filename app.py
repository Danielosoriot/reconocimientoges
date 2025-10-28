import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Anuel AI Recon",
    page_icon="🎤",
    layout="centered"
)

# --- FONDO CON DEGRADADO NEGRO → ROJO ---
page_bg = """
<style>
.stApp {
    background: linear-gradient(180deg, #000000 0%, #660000 50%, #cc0000 100%);
    background-attachment: fixed;
    color: white;
    font-family: 'Trebuchet MS', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    text-shadow: 2px 2px 8px #000000;
}
.css-1d391kg, .css-1v3fvcr {
    background-color: rgba(0, 0, 0, 0.7) !important;
    border-radius: 15px;
    padding: 12px;
}
.stButton>button {
    background: linear-gradient(90deg, #ff0000, #660000);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: bold;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #ff3333, #990000);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- ENCABEZADO ---
st.title("🎶 ANUEL AI RECON 🔥")
st.markdown("### *‘Real hasta la muerte... pero digital’* 💀")
st.write("Versión de Python:", platform.python_version())

# --- CARGA DEL MODELO ---
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# --- IMAGEN PRINCIPAL ---
col1, col2, col3 = st.columns([1,2,1])
with col2:
    image = Image.open('anuel2.png')
    st.image(image, width=330, caption="Modo Anuel activado 🎧")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 📸 Cámara con Flow")
    st.write("Usa tu modelo entrenado en Teachable Machine para reconocer tus movimientos al ritmo del trap 🧠🎤")
    st.markdown("---")
    st.markdown("**Consejo:** buena luz = mejores resultados 🔦")

# --- ENTRADA DE CÁMARA ---
img_file_buffer = st.camera_input("Haz tu foto con flow 😎")

# --- PROCESAMIENTO DE IMAGEN ---
if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    # --- RESULTADOS ---
    st.markdown("---")
    st.subheader("🔍 Resultado del modelo:")
    if prediction[0][0] > 0.5:
        st.success(f"🌀 Movimiento: **Izquierda** | Probabilidad: {prediction[0][0]:.2f}")
        st.balloons()
    elif prediction[0][1] > 0.5:
        st.success(f"🔥 Movimiento: **Arriba** | Probabilidad: {prediction[0][1]:.2f}")
        st.snow()
    else:
        st.warning("👀 No se pudo identificar claramente el movimiento. Intenta otra pose.")

    st.markdown("### 💿 *‘Otro palo más de la inteligencia artificial’* 🎶")

