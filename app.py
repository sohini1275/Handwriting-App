import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import cv2

from tensorflow.keras.models import load_model


# =====================================
# LOAD MODELS & LABELS
# =====================================

@st.cache_resource
def load_resources():

    english_model = load_model("English/english_model.keras")
    hindi_model = load_model("Hindi/hindi_model.h5")
    japanese_model = load_model("Japanese/Hiragana/hiragana_model.h5")

    with open("English/english_labels.txt", "r", encoding="utf-8") as f:
        english_labels = [line.strip() for line in f]

    with open("Hindi/hindi_labels.txt", "r", encoding="utf-8") as f:
        hindi_labels = [line.strip() for line in f]

    with open("Japanese/Hiragana/hiragana_labels.txt", "r", encoding="utf-8") as f:
        japanese_labels = [line.strip() for line in f]

    return (
        english_model,
        hindi_model,
        japanese_model,
        english_labels,
        hindi_labels,
        japanese_labels
    )


(
    english_model,
    hindi_model,
    japanese_model,
    english_labels,
    hindi_labels,
    japanese_labels
) = load_resources()


# =====================================
# PAGE
# =====================================

st.set_page_config(
    page_title="Handwriting Recognition",
    layout="wide"
)

st.title("✍️ Multilingual Handwriting Recognition")
st.write("English • Hindi • Japanese")


# =====================================
# LANGUAGE SELECT
# =====================================

language = st.sidebar.selectbox(
    "Select Language",
    ["English", "Hindi", "Japanese"]
)


# =====================================
# PICK MODEL
# =====================================

if language == "English":
    model = english_model
    labels = english_labels

elif language == "Hindi":
    model = hindi_model
    labels = hindi_labels

else:
    model = japanese_model
    labels = japanese_labels


# =====================================
# CANVAS
# =====================================

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)


# =====================================
# PREDICT
# =====================================

if st.button("Predict"):

    if canvas_result.image_data is not None:

        # Canvas image
        img = canvas_result.image_data

        # Convert RGBA -> Grayscale
        img = cv2.cvtColor(
            img.astype(np.uint8),
            cv2.COLOR_RGBA2GRAY
        )

        # Convert to PIL Image
        img = Image.fromarray(img)

        # SAME PREPROCESSING AS COLAB

        # Invert
        img = ImageOps.invert(img)

        # Crop character
        bbox = img.getbbox()

        if bbox:
            img = img.crop(bbox)

        # Add padding
        img = ImageOps.expand(
            img,
            border=20,
            fill=255
        )

        # Resize
        img = img.resize((64, 64))

        # Normalize
        img_array = np.array(img).astype("float32") / 255.0

        img_array = img_array.reshape(
            1,
            64,
            64,
            1
        )

        # Debug image (optional)
        st.image(
            img,
            caption="Processed Image",
            width=150
        )

        # Prediction
        prediction = model.predict(
            img_array,
            verbose=0
        )

        predicted_index = np.argmax(prediction)

        predicted_label = labels[predicted_index]

        confidence = float(
            np.max(prediction) * 100
        )

        st.success(
            f"Prediction: {predicted_label}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        st.subheader("Top 3 Predictions")

        top3 = np.argsort(
            prediction[0]
        )[-3:][::-1]

        for idx in top3:

            st.write(
                f"{labels[idx]} : "
                f"{prediction[0][idx] * 100:.2f}%"
            )