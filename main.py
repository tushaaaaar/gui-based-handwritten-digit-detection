import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
import tensorflow as tf

# Load or train the model
try:
    model = tf.keras.models.load_model('handwritten.keras')
    print("Model loaded successfully!")
except:
    print("Training new model...")
    mnist = tf.keras.datasets.mnist
    x_train, y_train, x_test, y_test = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3)
    model.save('handwritten.keras')
    print("Model trained and saved successfully!")


# Function to predict the digit
def predict_digit(image):
    image = image.resize((28, 28)).convert('L')
    image = ImageOps.invert(image)
    image_array = np.array(image) / 255.0
    processed_image = image_array.reshape(1, 28, 28)
    prediction = model.predict(processed_image)
    return np.argmax(prediction)

# GUI for the application
class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")

        # Fix the window size
        self.root.geometry("300x350")  # Set a fixed size (width x height)
        self.root.resizable(False, False)  # Disable resizing

        # Canvas for drawing and displaying images
        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack(pady=10)

        # Buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict_canvas)
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.upload_button = tk.Button(self.button_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(side=tk.LEFT, padx=5)

        self.result_label = tk.Label(root, text="Draw or upload a digit and press Predict.", font=("Helvetica", 12))
        self.result_label.pack(pady=10)

        # Bind drawing event
        self.canvas.bind("<B1-Motion>", self.draw)

        # Initialize blank image for drawing and uploaded image placeholder
        self.image = Image.new("RGB", (200, 200), "white")
        self.draw_obj = ImageDraw.Draw(self.image)
        self.uploaded_image = None

    # Draw on the canvas
    def draw(self, event):
        x, y = event.x, event.y
        r = 8  # Brush size
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
        self.draw_obj.ellipse([x - r, y - r, x + r, y + r], fill='black')

    # Clear the canvas
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (200, 200), "white")
        self.draw_obj = ImageDraw.Draw(self.image)
        self.uploaded_image = None
        self.result_label.config(text="Canvas cleared. Draw or upload an image.")

    # Predict digit from the canvas
    def predict_canvas(self):
        digit = predict_digit(self.image)
        self.result_label.config(text=f"Predicted Digit: {digit}")

    # Upload an image, display it on the canvas, and predict
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            # Load and display the uploaded image
            uploaded_image = Image.open(file_path).resize((200, 200))
            self.uploaded_image = ImageTk.PhotoImage(uploaded_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.uploaded_image)

            # Predict the uploaded image
            digit = predict_digit(uploaded_image)
            self.result_label.config(text=f"Predicted Digit: {digit}")

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
