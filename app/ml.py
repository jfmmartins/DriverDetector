import os
import cv2 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

#################
# Dataset Loading
#################

DATASET_DIR = "../archive/train"
CATEGORIES = ["Open_Eyes","Closed_Eyes"]

def train():
    data = []
    labels = []

    IMG_SIZE=64

    for idx,category in enumerate(CATEGORIES):
        path = os.path.join(DATASET_DIR, category)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                data.append(img.reshape(IMG_SIZE, IMG_SIZE, 1))
                labels.append(idx)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")


    data = np.array(data, dtype=np.float32)
    labels = np.array(labels)

    print(f"Loaded {len(data)} samples.")

    ##################
    # Train Test Split
    ##################

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    ##################
    # Training
    ##################

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.utils import to_categorical

    # one-hot encoding
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train_cat, epochs=10, batch_size=32, validation_split=0.1)

    ##################
    # Eval
    ##################

    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    print(f"Test Accuracy: {test_acc:.4f}")

    model.save("models/eye_state_cnn.h5")
