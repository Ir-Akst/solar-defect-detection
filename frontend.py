import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Solar Panel Defect Detection",
    page_icon="⚡",
    layout="centered"
)

# =========================================
# MODEL CONFIG
# =========================================

IMG_SIZE = 224
THRESHOLD = 0.4

# =========================================
# BUILD MODEL ARCHITECTURE
# =========================================

@st.cache_resource
def load_model():

    base_model = MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.load_weights("weights.h5")

    return model

model = load_model()

# =========================================
# IMAGE PREPROCESSING
# =========================================

def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image)

    img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# =========================================
# UI
# =========================================

st.title("⚡ Solar Panel Defect Detection")

st.write(
    """
    Upload a thermal solar panel image to detect whether
    the panel is defective or normal.
    """
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================
# PREDICTION
# =========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Analyzing image..."):

        img = preprocess_image(image)

        prob = model.predict(img)[0][0]

        prediction = "Defective" if prob > THRESHOLD else "Normal"

    st.subheader("Prediction Result")

    if prediction == "Defective":

        st.error(f"⚠️ {prediction}")

    else:

        st.success(f"✅ {prediction}")

    st.write(f"### Probability Score: {prob:.4f}")

    st.progress(float(prob))

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.caption(
    "Built using TensorFlow, MobileNetV2, and Streamlit"
)