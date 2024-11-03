from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

data = np.load("asl_keypoints_dataset.npz")
keypoints_data = data['samples']
labels = data['labels']
keypoints_data = keypoints_data.reshape((keypoints_data.shape[0], 21, 3))

encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

X_train, X_val, y_train, y_val = train_test_split(
    keypoints_data, labels_categorical, test_size=0.2, random_state=42
)

model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(21, 3)),
    Conv1D(128, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(labels_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler]
)

model.save("asl_hand_gesture_model_improved2.h5")
print("Model complete.")