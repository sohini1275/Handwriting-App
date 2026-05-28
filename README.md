# ✍️ Multilingual Handwriting Recognition

A deep learning-based handwriting recognition system that recognizes handwritten characters in multiple languages using Convolutional Neural Networks (CNNs).

Supported languages:
- English
- Hindi
- Japanese Hiragana

---

## 🚀 Features

- Handwritten character recognition
- Multilingual support
- CNN-based deep learning models
- Image preprocessing and prediction
- Pretrained models included

---

## 🧠 Model Details

- Built using TensorFlow/Keras
- Input size: 28×28 grayscale images
- CNN architecture for character classification
- Separate models for each language

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Pillow (PIL)
- Streamlit

---

## 📂 Project Structure

```text
English/
Hindi/
Japanese/Hiragana/
```

---

## ▶️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Train Models

### English

```bash
python English/train_english.py
```

### Hindi

```bash
python Hindi/train_hindi.py
```

### Hiragana

```bash
python Japanese/Hiragana/train_hiragana.py
```

---

## 💾 Included Models

- english_model.h5
- hindi_model.h5
- hiragana_model.h5

---

## 🌟 Future Improvements

- Katakana recognition
- Real-time webcam prediction
- Interactive drawing canvas
- Web deployment
