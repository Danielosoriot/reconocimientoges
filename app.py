import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import platform
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Anuel AI Recon 🔥",
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
st.write("🧠 Versión de Python:", platform.python_version())

# --- CARGA DEL MODELO ---
st.markdown("## ⚙️ Cargando modelo de reconocimiento...")
model_path = "saved_model_anuel"  # Asegúrate de tener este modelo exportado desde Teachable Machine

try:
    model = tf.keras.models.load_model(model_path)
    st.success("✅ Modelo cargado con éxito")
except Exception as e:
    st.error("❌ Error al cargar el modelo. Verifica que esté en formato SavedModel y compatible con tu versión de Python.")
    st.code(str(e))
    st.stop()

# --- IMAGEN PRINCIPAL ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        portada = Image.open("anuel2.png")
        st.image(portada, width=330, caption="Modo Anuel activado 🎧")
    except:
        st.warning("⚠️ No se encontró la imagen 'anuel2.png'. Puedes agregarla para más flow.")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 📸 Cámara con Flow")
    st.write("Usa tu modelo entrenado en Teachable Machine para reconocer tus movimientos al ritmo del trap 🧠🎤")
    st.markdown("---")
    st.markdown("💡 Consejo: buena luz = mejores resultados 🔦")
    st.markdown("💬 *“Yo no tengo enemigos, tengo fanáticos confundidos.”* — Anuel AA")

# --- ENTRADA DE CÁMARA ---
img_file_buffer = st.camera_input("Haz tu foto con flow 😎")

# --- PROCESAMIENTO Y PREDICCIÓN ---
if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    img = img.resize((224, 224))
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)

    # --- RESULTADOS ---
    st.markdown("---")
    st.subheader("🔍 Resultado del modelo:")
    labels = ["Izquierda", "Arriba", "Derecha"]  # Ajusta según tu modelo

    max_index = np.argmax(prediction[0])
    confidence = prediction[0][max_index]

    if confidence > 0.5:
        movimiento = labels[max_index]
        st.success(f"🎯 Movimiento detectado: **{movimiento}** | Confianza: {confidence:.2f}")
        if movimiento == "Izquierda":
            st.balloons()
        elif movimiento == "Arriba":
            st.snow()
        else:
            st.toast("🔥 ¡Flow detectado!", icon="🎧")
    else:
        st.warning("👀 No se pudo identificar claramente el movimiento. Intenta otra pose.")

    st.markdown("### 💿 *‘Otro palo más de la inteligencia artificial’* 🎶")

# --- FOOTER ---
st.markdown("---")
st.caption("💿 App creada con el flow de Anuel AA | Reconocimiento AI | Real Hasta La Muerte 💀")
