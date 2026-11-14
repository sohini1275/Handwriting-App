
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np

from tensorflow.keras.models import load_model


# =====================================
# LOAD MODELS & LABELS
# =====================================

@st.cache_resource
def load_resources():

    # Load models
    english_model = load_model("English/english_model.h5")
    hindi_model = load_model("Hindi/hindi_model.h5")
    japanese_model = load_model("Japanese/Hiragana/hiragana_model.h5")
    korean_model = load_model("Korean/korean_model.h5")
    russian_model = load_model("Russian/russian_model.h5")

    # Load labels
    with open("English/english_labels.txt", "r", encoding="utf-8") as f:
        english_labels = [line.strip() for line in f]

    with open("Hindi/hindi_labels.txt", "r", encoding="utf-8") as f:
        hindi_labels = [line.strip() for line in f]

    with open("Japanese/Hiragana/hiragana_labels.txt", "r", encoding="utf-8") as f:
        japanese_labels = [line.strip() for line in f]

    with open("Korean/korean_labels.txt", "r", encoding="utf-8") as f:
        korean_labels = [line.strip() for line in f]

    with open("Russian/russian_labels.txt", "r", encoding="utf-8") as f:
        russian_labels = [line.strip() for line in f]

    return (
        english_model,
        hindi_model,
        japanese_model,
        korean_model,
        russian_model,
        english_labels,
        hindi_labels,
        japanese_labels,
        korean_labels,
        russian_labels
    )


(
    english_model,
    hindi_model,
    japanese_model,
    korean_model,
    russian_model,
    english_labels,
    hindi_labels,
    japanese_labels,
    korean_labels,
    russian_labels
) = load_resources()


# =====================================
# PAGE
# =====================================

st.set_page_config(
    page_title="Multilingual Handwriting Recognition",
    layout="wide"
)

st.title("✍️ Multilingual Handwriting Recognition")
st.write("English • Hindi • Japanese • Korean • Russian")


# =====================================
# LANGUAGE SELECT
# =====================================

language = st.sidebar.selectbox(
    "Select Language",
    ["English", "Hindi", "Japanese", "Korean", "Russian"]
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

elif language == "Japanese":
    model = japanese_model
    labels = japanese_labels

elif language == "Korean":
    model = korean_model
    labels = korean_labels

else:
    model = russian_model
    labels = russian_labels


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

        # Get canvas image
        img = canvas_result.image_data

        # Convert RGBA to grayscale using Pillow
        img = Image.fromarray(
            img.astype(np.uint8),
            mode="RGBA"
        )

        img = img.convert("L")

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

        # Debug image
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
