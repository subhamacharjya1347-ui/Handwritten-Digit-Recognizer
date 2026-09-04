# ✍️ Handwritten Digit Recognizer

A machine learning application that recognizes handwritten digits from **0 to 9** using a Convolutional Neural Network (CNN) trained on the **MNIST dataset**.

The project provides an interactive graphical interface where users can draw a digit and receive the model's prediction along with its confidence score.

---

## 🚀 Features

- ✍️ Draw handwritten digits using the mouse
- 🔢 Recognizes digits from **0 to 9**
- 🤖 CNN-based deep learning model
- 📊 Displays prediction confidence
- 📈 Displays probability for each digit
- 🖥️ Interactive Tkinter GUI
- 🧹 Clear drawing and make another prediction
- ✅ Achieves **98.86% test accuracy**

---

## 📊 Model Performance

| Metric | Result |
|--------|--------|
| Dataset | MNIST |
| Test Accuracy | **98.86%** |
| Test Loss | **0.0366** |
| Classes | 10 (0–9) |

### Example Prediction

The application successfully predicted a handwritten **4** with:

**Confidence: 95.85%**

---

## 🧠 How It Works

The application follows these steps:

1. The user draws a digit on the drawing canvas.
2. The drawing is captured as an image.
3. The image is converted into the required format.
4. The image is resized and preprocessed.
5. The trained CNN model analyzes the image.
6. The model predicts the most likely digit.
7. The application displays:
   - Predicted digit
   - Confidence percentage
   - Probability of each digit

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Tkinter
- Pillow (PIL)
- OpenCV
- MNIST Dataset

---

## 📂 Project Structure

```text
Handwritten-Digit-Recognizer/
│
├── digit_recognizer.py
├── README.md
└── requirements.txt