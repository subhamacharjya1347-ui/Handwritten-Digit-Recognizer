# ============================================================
# HANDWRITTEN DIGIT RECOGNIZER
# Digits: 0 to 9
# ============================================================

import tensorflow as tf
import numpy as np
import tkinter as tk

from PIL import Image, ImageDraw
import cv2


# ============================================================
# PROGRAM INFORMATION
# ============================================================

print("=" * 60)
print("        HANDWRITTEN DIGIT RECOGNIZER")
print("=" * 60)

print("TensorFlow version:", tf.__version__)


# ============================================================
# LOAD MNIST DATASET
# ============================================================

print("\nLoading MNIST dataset...")

from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training images:", X_train.shape)
print("Testing images :", X_test.shape)


# ============================================================
# NORMALIZE DATA
# ============================================================

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Add channel dimension
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("Image shape:", X_train.shape)


# ============================================================
# BUILD CNN MODEL
# ============================================================

print("\nBuilding CNN model...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(28, 28, 1)),

    # First convolution layer
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Second convolution layer
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Convert feature maps to vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # 10 output classes: 0,1,2,3,4,5,6,7,8,9
    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("CNN model compiled successfully!")


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

print("\nModel training completed!")


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss    : {test_loss:.4f}")


# ============================================================
# TEST MNIST IMAGE
# ============================================================

index = 0

test_prediction = model.predict(
    X_test[index:index + 1],
    verbose=0
)

test_digit = np.argmax(test_prediction)

print("\n" + "=" * 60)
print("MNIST TEST")
print("=" * 60)

print("Actual digit   :", y_test[index])
print("Predicted digit:", test_digit)


# ============================================================
# CREATE GUI
# ============================================================

print("\n" + "=" * 60)
print("STARTING HANDWRITTEN DIGIT RECOGNIZER")
print("=" * 60)


root = tk.Tk()

root.title("Handwritten Digit Recognizer")

root.geometry("1050x750")

root.configure(bg="#f2f2f2")


# ============================================================
# TITLE
# ============================================================

title_label = tk.Label(
    root,
    text="Handwritten Digit Recognizer",
    font=("Arial", 30, "bold"),
    bg="#f2f2f2"
)

title_label.pack(pady=(20, 5))


subtitle_label = tk.Label(
    root,
    text="Draw a digit from 0 to 9",
    font=("Arial", 16),
    bg="#f2f2f2"
)

subtitle_label.pack(pady=(0, 15))


# ============================================================
# MAIN FRAME
# ============================================================

main_frame = tk.Frame(
    root,
    bg="#f2f2f2"
)

main_frame.pack()


# ============================================================
# LEFT FRAME
# ============================================================

left_frame = tk.Frame(
    main_frame,
    bg="#f2f2f2"
)

left_frame.grid(
    row=0,
    column=0,
    padx=30,
    pady=10
)


drawing_title = tk.Label(
    left_frame,
    text="Drawing Area",
    font=("Arial", 20, "bold"),
    bg="#f2f2f2"
)

drawing_title.pack(pady=10)


# ============================================================
# DRAWING CANVAS
# ============================================================

CANVAS_SIZE = 420

canvas = tk.Canvas(
    left_frame,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black",
    highlightthickness=2,
    highlightbackground="gray"
)

canvas.pack()


# ============================================================
# PIL IMAGE
# ============================================================

# This image stores exactly what the user draws.
# Black background = 0
# White drawing = 255

drawing_image = Image.new(
    "L",
    (CANVAS_SIZE, CANVAS_SIZE),
    0
)

draw = ImageDraw.Draw(drawing_image)


# ============================================================
# DRAW DIGIT
# ============================================================

BRUSH_SIZE = 25


def draw_digit(event):

    x = event.x
    y = event.y

    # Draw on Tkinter canvas
    canvas.create_oval(
        x - BRUSH_SIZE // 2,
        y - BRUSH_SIZE // 2,
        x + BRUSH_SIZE // 2,
        y + BRUSH_SIZE // 2,
        fill="white",
        outline="white"
    )

    # Draw on PIL image
    draw.ellipse(
        (
            x - BRUSH_SIZE // 2,
            y - BRUSH_SIZE // 2,
            x + BRUSH_SIZE // 2,
            y + BRUSH_SIZE // 2
        ),
        fill=255
    )


def draw_line(event):

    x = event.x
    y = event.y

    # Draw line on PIL image
    # This makes drawing smoother.
    draw.line(
        (
            x,
            y,
            x,
            y
        ),
        fill=255,
        width=BRUSH_SIZE
    )


canvas.bind(
    "<B1-Motion>",
    draw_digit
)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_digit():

    # Convert PIL image to NumPy
    image = np.array(drawing_image)

    # --------------------------------------------------------
    # Find all white pixels
    # --------------------------------------------------------

    coords = cv2.findNonZero(image)

    # If nothing was drawn
    if coords is None:
        return None

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(coords)

    # --------------------------------------------------------
    # Crop only the digit
    # --------------------------------------------------------

    digit = image[
        y:y + h,
        x:x + w
    ]

    # --------------------------------------------------------
    # Add some padding
    # --------------------------------------------------------

    padding = 30

    digit = cv2.copyMakeBorder(
        digit,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # --------------------------------------------------------
    # Resize while maintaining shape
    # --------------------------------------------------------

    height, width = digit.shape

    scale = 20.0 / max(height, width)

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    digit = cv2.resize(
        digit,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Create 28x28 MNIST image
    # --------------------------------------------------------

    final_image = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    # Calculate center position
    start_x = (28 - new_width) // 2
    start_y = (28 - new_height) // 2

    final_image[
        start_y:start_y + new_height,
        start_x:start_x + new_width
    ] = digit

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    final_image = final_image.astype(
        "float32"
    ) / 255.0

    # Add batch and channel dimensions
    final_image = final_image.reshape(
        1,
        28,
        28,
        1
    )

    return final_image


# ============================================================
# RIGHT FRAME
# ============================================================

right_frame = tk.Frame(
    main_frame,
    bg="#f2f2f2"
)

right_frame.grid(
    row=0,
    column=1,
    padx=30,
    pady=10,
    sticky="n"
)


prediction_title = tk.Label(
    right_frame,
    text="Prediction",
    font=("Arial", 22, "bold"),
    bg="#f2f2f2"
)

prediction_title.pack(pady=10)


# ============================================================
# RESULT LABEL
# ============================================================

result_label = tk.Label(
    right_frame,
    text="Predicted Digit: -",
    font=("Arial", 25, "bold"),
    bg="#f2f2f2"
)

result_label.pack(pady=15)


confidence_label = tk.Label(
    right_frame,
    text="Confidence: -",
    font=("Arial", 18),
    bg="#f2f2f2"
)

confidence_label.pack(pady=5)


# ============================================================
# PROBABILITY TITLE
# ============================================================

probability_title = tk.Label(
    right_frame,
    text="Digit Probabilities",
    font=("Arial", 18, "bold"),
    bg="#f2f2f2"
)

probability_title.pack(
    pady=(25, 10)
)


# ============================================================
# PROBABILITY BARS
# ============================================================

probability_frame = tk.Frame(
    right_frame,
    bg="#f2f2f2"
)

probability_frame.pack()


probability_bars = []

probability_labels = []


for digit_number in range(10):

    row = tk.Frame(
        probability_frame,
        bg="#f2f2f2"
    )

    row.pack(
        pady=3
    )

    label = tk.Label(
        row,
        text=f"{digit_number}: 0.00%",
        font=("Arial", 12),
        width=12,
        anchor="w",
        bg="#f2f2f2"
    )

    label.pack(
        side=tk.LEFT
    )

    bar = tk.Canvas(
        row,
        width=250,
        height=18,
        bg="white",
        highlightthickness=1,
        highlightbackground="gray"
    )

    bar.pack(
        side=tk.LEFT
    )

    probability_labels.append(label)
    probability_bars.append(bar)


# ============================================================
# MODEL ACCURACY
# ============================================================

accuracy_label = tk.Label(
    right_frame,
    text=f"Model Accuracy: {test_accuracy * 100:.2f}%",
    font=("Arial", 17, "bold"),
    bg="#f2f2f2"
)

accuracy_label.pack(
    pady=25
)


# ============================================================
# PREDICTION HISTORY
# ============================================================

history_title = tk.Label(
    right_frame,
    text="Prediction History",
    font=("Arial", 18, "bold"),
    bg="#f2f2f2"
)

history_title.pack(
    pady=(5, 5)
)


history_list = tk.Listbox(
    right_frame,
    width=35,
    height=7,
    font=("Arial", 11)
)

history_list.pack()


# ============================================================
# PREDICT FUNCTION
# ============================================================

def predict_digit():

    processed_image = preprocess_digit()

    # Check if user has drawn anything
    if processed_image is None:

        result_label.config(
            text="Please draw a digit!"
        )

        confidence_label.config(
            text="Confidence: -"
        )

        return

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    # Get digit 0 to 9
    digit = int(
        np.argmax(prediction[0])
    )

    # Get confidence
    confidence = float(
        np.max(prediction[0]) * 100
    )

    # --------------------------------------------------------
    # Display prediction
    # --------------------------------------------------------

    result_label.config(
        text=f"Predicted Digit: {digit}"
    )

    confidence_label.config(
        text=f"Confidence: {confidence:.2f}%"
    )

    # --------------------------------------------------------
    # Update probability bars
    # --------------------------------------------------------

    for i in range(10):

        probability = float(
            prediction[0][i] * 100
        )

        probability_labels[i].config(
            text=f"{i}: {probability:.2f}%"
        )

        # Clear old bar
        probability_bars[i].delete(
            "all"
        )

        # Draw new bar
        bar_width = int(
            250 * probability / 100
        )

        probability_bars[i].create_rectangle(
            0,
            0,
            bar_width,
            18,
            fill="green",
            outline="green"
        )

    # --------------------------------------------------------
    # Add to history
    # --------------------------------------------------------

    history_list.insert(
        0,
        f"Digit {digit}  -  {confidence:.2f}%"
    )

    print(
        f"User digit prediction: {digit} "
        f"({confidence:.2f}%)"
    )


# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_canvas():

    # Clear Tkinter canvas
    canvas.delete("all")

    # Clear PIL image
    draw.rectangle(
        (
            0,
            0,
            CANVAS_SIZE,
            CANVAS_SIZE
        ),
        fill=0
    )

    # Reset result
    result_label.config(
        text="Predicted Digit: -"
    )

    confidence_label.config(
        text="Confidence: -"
    )

    # Reset probability bars
    for i in range(10):

        probability_labels[i].config(
            text=f"{i}: 0.00%"
        )

        probability_bars[i].delete(
            "all"
        )


# ============================================================
# BUTTON FRAME
# ============================================================

button_frame = tk.Frame(
    left_frame,
    bg="#f2f2f2"
)

button_frame.pack(
    pady=20
)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = tk.Button(
    button_frame,
    text="Predict",
    command=predict_digit,
    font=("Arial", 16, "bold"),
    width=12,
    height=2
)

predict_button.pack(
    side=tk.LEFT,
    padx=10
)


# ============================================================
# CLEAR BUTTON
# ============================================================

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    font=("Arial", 16, "bold"),
    width=12,
    height=2
)

clear_button.pack(
    side=tk.LEFT,
    padx=10
)


# ============================================================
# INSTRUCTIONS
# ============================================================

instruction_label = tk.Label(
    left_frame,
    text="Draw ONE digit clearly: 0 1 2 3 4 5 6 7 8 9",
    font=("Arial", 13),
    bg="#f2f2f2"
)

instruction_label.pack(
    pady=5
)


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()