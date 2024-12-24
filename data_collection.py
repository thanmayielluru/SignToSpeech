    # import cv2
    # import os
    # import numpy as np

    # # Directory setup
    # directory = 'data/'
    # if not os.path.exists(directory):
    #     os.mkdir(directory)

    # # Create directories for each letter and blank
    # letters = [chr(i) for i in range(65, 91)] + ['blank']
    # for letter in letters:
    #     letter_dir = os.path.join(directory, letter)
    #     if not os.path.exists(letter_dir):
    #         os.mkdir(letter_dir)

    # # Minimum value for thresholding (constant)
    # minValue = 70

    # # Preprocessing function (same adaptive thresholding for consistency)
    # def preprocess_frame(frame):
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     blur = cv2.GaussianBlur(gray, (5, 5), 2)
        
    #     # Apply adaptive thresholding to ensure the same result for each frame
    #     th3 = cv2.adaptiveThreshold(
    #         blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    #     )
    #     _, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #     return res

    # # Start video capture
    # cap = cv2.VideoCapture(0)

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     # Draw ROI on the frame
    #     cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    #     cv2.imshow("Frame", frame)

    #     # Extract and preprocess ROI
    #     roi = frame[40:300, 0:300]
    #     processed_roi = preprocess_frame(roi)
    #     roi_resized = cv2.resize(processed_roi, (128, 128))

    #     # Display processed ROI
    #     cv2.imshow("Processed ROI", roi_resized)

    #     # Wait for user input
    #     interrupt = cv2.waitKey(10) & 0xFF

    #     # Map user input to directory
    #     if chr(interrupt).lower() in [chr(i).lower() for i in range(97, 123)] + ['.']:
    #         label = chr(interrupt).upper() if interrupt != ord('.') else 'blank'
    #         label_dir = os.path.join(directory, label)
    #         count = len(os.listdir(label_dir))
    #         save_path = os.path.join(label_dir, f"{label}_{count}.jpg")
    #         cv2.imwrite(save_path, roi_resized)
    #         print(f"Saved: {save_path}")

    #     # Quit on 'q'
    #     if interrupt == ord('.'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

    # import cv2
    # import os
    # import numpy as np

    # # Directory setup
    # base_directory = 'data/'

    # # Create main directories for train and test if they don't exist
    # for folder in ['train', 'test']:
    #     folder_path = os.path.join(base_directory, folder)
    #     if not os.path.exists(folder_path):
    #         os.mkdir(folder_path)
        
    #     # Create 26 subfolders for each letter and blank in both train and test folders
    #     for letter in [chr(i) for i in range(65, 91)] + ['blank']:
    #         letter_dir = os.path.join(folder_path, letter)
    #         if not os.path.exists(letter_dir):
    #             os.mkdir(letter_dir)

    # # Minimum value for thresholding (constant)
    # minValue = 70

    # # Preprocessing function (same adaptive thresholding for consistency)
    # def preprocess_frame(frame):
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     blur = cv2.GaussianBlur(gray, (5, 5), 2)
        
    #     # Apply adaptive thresholding to ensure the same result for each frame
    #     th3 = cv2.adaptiveThreshold(
    #         blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    #     )
    #     _, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #     return res

    # # Start video capture
    # cap = cv2.VideoCapture(0)

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     # Draw ROI on the frame
    #     cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    #     cv2.imshow("Frame", frame)

    #     # Extract and preprocess ROI
    #     roi = frame[40:300, 0:300]
    #     processed_roi = preprocess_frame(roi)
    #     roi_resized = cv2.resize(processed_roi, (128, 128))

    #     # Display processed ROI
    #     cv2.imshow("Processed ROI", roi_resized)

    #     # Wait for user input
    #     interrupt = cv2.waitKey(10) & 0xFF

    #     # Map user input to directory
    #     if chr(interrupt).lower() in [chr(i).lower() for i in range(97, 123)] + ['.']:
    #         label = chr(interrupt).upper() if interrupt != ord('.') else 'blank'
    #         train_dir = os.path.join(base_directory, 'train', label)
    #         test_dir = os.path.join(base_directory, 'test', label)

    #         # Count the total number of images saved for the label in train and test
    #         train_images = len(os.listdir(train_dir))
    #         test_images = len(os.listdir(test_dir))

    #         # Save the image in the appropriate folder (test if < 100, else train)
    #         if test_images < 100:
    #             save_path = os.path.join(test_dir, f"{label}_{test_images}.jpg")
    #             cv2.imwrite(save_path, roi_resized)
    #             print(f"Saved: {save_path} in test")
    #         elif train_images < 400:
    #             save_path = os.path.join(train_dir, f"{label}_{train_images}.jpg")
    #             cv2.imwrite(save_path, roi_resized)
    #             print(f"Saved: {save_path} in train")

    #     # Quit on 'q'
    #     if interrupt == ord('.'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()


import cv2
import os
import numpy as np

# Directory setup
base_directory = 'data/'

# Create main directories for train and test if they don't exist
for folder in ['train', 'test']:
    folder_path = os.path.join(base_directory, folder)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # Create 26 subfolders for each letter, blank, and space in both train and test folders
    for label in [chr(i) for i in range(65, 91)] + ['blank', 'space']:
        label_dir = os.path.join(folder_path, label)
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

# Minimum value for thresholding (constant)
minValue = 70

# Preprocessing function (same adaptive thresholding for consistency)
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    # Apply adaptive thresholding to ensure the same result for each frame
    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    _, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw ROI on the frame
    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    cv2.imshow("Frame", frame)

    # Extract and preprocess ROI
    roi = frame[40:300, 0:300]
    processed_roi = preprocess_frame(roi)
    roi_resized = cv2.resize(processed_roi, (128, 128))

    # Display processed ROI
    cv2.imshow("Processed ROI", roi_resized)

    # Wait for user input
    interrupt = cv2.waitKey(10) & 0xFF

    # Map user input to directory
    if chr(interrupt).lower() in [chr(i).lower() for i in range(97, 123)] + ['.', ' ']:
        if interrupt == ord('.'):
            label = 'blank'
        elif interrupt == ord(' '):
            label = 'space'
        else:
            label = chr(interrupt).upper()

        train_dir = os.path.join(base_directory, 'train', label)
        test_dir = os.path.join(base_directory, 'test', label)

        # Count the total number of images saved for the label in train and test
        train_images = len(os.listdir(train_dir))
        test_images = len(os.listdir(test_dir))

        # Save the image in the appropriate folder (test if < 100, else train)
        if test_images < 100:
            save_path = os.path.join(test_dir, f"{label}_{test_images}.jpg")
            cv2.imwrite(save_path, roi_resized)
            print(f"Saved: {save_path} in test")
        elif train_images < 400:
            save_path = os.path.join(train_dir, f"{label}_{train_images}.jpg")
            cv2.imwrite(save_path, roi_resized)
            print(f"Saved: {save_path} in train")

    # Quit on 'q'
    if interrupt == ord('.'):
        break

cap.release()
cv2.destroyAllWindows()
