from tensorflow.keras.models import load_model

model = load_model("english_model.keras", compile=False)
model.save("english_model.h5")