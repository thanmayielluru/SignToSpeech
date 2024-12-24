import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk
import os
from gtts import gTTS
from playsound import playsound

# Load the trained model
# model = load_model('signs2.h5')
model = load_model('signlanguagedetectionmodelspace.h5')  
labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','Space']

class SignLanguageApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        # Set video resolution
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Create 'predicted_images' directory if it doesn't exist
        if not os.path.exists("predicted_images"):
            os.makedirs("predicted_images")

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        # Label to display predicted sign
        self.label = tk.Label(window, text="Predicted sign: ", font=("Helvetica", 16))
        self.label.pack()

        # Create buttons
        self.btn_append = tk.Button(window, text="Append Prediction", width=20, command=self.append_prediction)
        self.btn_append.pack(side=tk.LEFT, padx=10)

        self.btn_clear = tk.Button(window, text="Clear All", width=20, command=self.clear_text)
        self.btn_clear.pack(side=tk.LEFT, padx=10)

        self.btn_speak = tk.Button(window, text="Convert to Speech", width=20, command=self.convert_to_speech)
        self.btn_speak.pack(side=tk.LEFT, padx=10)

        self.btn_quit = tk.Button(window, text="Quit", width=20, command=self.quit_app)
        self.btn_quit.pack(side=tk.LEFT, padx=10)

        self.textbox = tk.Text(window, height=5, width=52)
        self.textbox.pack(pady=10)

        # Prediction state
        self.current_prediction = None

        # Start video loop
        self.delay = 10
        self.update()
        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Define region of interest (ROI) for the hand
            roi = frame[100:300, 100:300]

            # Predict sign from ROI continuously
            self.predict_sign(roi)

            # Draw ROI boundary
            cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 0), 2)

            # Convert image to RGB for Tkinter display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Display the video feed on canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

        self.window.after(self.delay, self.update)

    def predict_sign(self, roi):
        try:
            # Convert the ROI to grayscale
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding to the grayscale ROI
            roi_adaptive = cv2.adaptiveThreshold(
                roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Resize and normalize ROI
            roi_resized = cv2.resize(roi_adaptive, (128, 128)) / 255.0
            roi_reshaped = roi_resized.reshape(1, 128, 128, 1)

            # Predict the sign using the model
            predictions = model.predict(roi_reshaped)
            predicted_class = np.argmax(predictions, axis=1)[0]
            self.current_prediction = labels[predicted_class]

            # Update the label to display the current prediction
            self.label.config(text=f"Predicted sign: {self.current_prediction}")
        except Exception as e:
            print(f"Prediction error: {e}")
            self.current_prediction = None

    def append_prediction(self):
        # if self.current_prediction:
        # # Append the current prediction directly to the text box without a space
        #     current_text = self.textbox.get(1.0, tk.END).strip()  # Get the current text
        #     self.textbox.delete(1.0, tk.END)  # Clear the text box
        #     self.textbox.insert(tk.END, current_text + self.current_prediction)  # Append the prediction
        if self.current_prediction:
            if self.current_prediction == "Space":
            # Append a space to the text box
                self.textbox.insert(tk.END, " ")
            else:
            # Append the current prediction to the text box
                self.textbox.insert(tk.END, self.current_prediction)


    def clear_text(self):
        self.textbox.delete(1.0, tk.END)

    def convert_to_speech(self):
        try:
            # Get the text from the text box
            text = self.textbox.get(1.0, tk.END).strip()
            if text:
                # Use gTTS to convert the text to speech
                tts = gTTS(text=text, lang='en')
                audio_file = "output.mp3"
                tts.save(audio_file)

                # Play the audio file
                playsound(audio_file)

                # Remove the audio file after playback
                os.remove(audio_file)
        except Exception as e:
            print(f"Speech conversion error: {e}")

    def quit_app(self):
        self.window.quit()
        self.vid.release()
        cv2.destroyAllWindows()

# Run the application
SignLanguageApp(tk.Tk(), "Sign Language to Text")
