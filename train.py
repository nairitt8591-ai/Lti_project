import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
import math
import os

from model import build_contextual_model

# --- Configuration ---
SEQUENCE_LENGTH = 8
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = "gesture_model_contextual.h5"

# --- 1. Data Generator Class ---
class SignLanguageGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, sequence_length, label_encoder):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.label_encoder = label_encoder
        self.indices = self._create_indices()

    def __len__(self):
        # Denotes the number of batches per epoch
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        # Gets one batch of data
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Pre-allocate memory for the batch
        X_batch = np.zeros((len(batch_indices), self.sequence_length, 64, 64, 3), dtype=np.float32)
        y_batch = np.zeros(len(batch_indices), dtype=np.int64)

        for i, data_index in enumerate(batch_indices):
            start_index = data_index['start']
            end_index = start_index + self.sequence_length
            
            # Load, normalize, and assign the sequence
            sequence = self.x[start_index:end_index].astype(np.float32) / 255.0
            X_batch[i] = sequence
            y_batch[i] = data_index['label']
            
        return X_batch, y_batch

    def _create_indices(self):
        # Creates a list of pointers to sequences instead of creating the sequences in memory
        print(f"Creating sequence indices...")
        indices = []
        for label_str in self.label_encoder.classes_:
            class_indices = np.where(self.y == label_str)[0]
            if len(class_indices) < self.sequence_length:
                continue
            
            label_int = self.label_encoder.transform([label_str])[0]
            for i in range(len(class_indices) - self.sequence_length + 1):
                start_index = class_indices[i]
                indices.append({'start': start_index, 'label': label_int})
        np.random.shuffle(indices)
        return indices

# --- 2. Load Data ---
print("🚀 Starting the training pipeline...")
print("Loading data using memory mapping (won't use RAM yet)...")
X_train_flat = np.load("X_train.npy", mmap_mode='r')
y_train_flat = np.load("y_train.npy")
X_test_flat = np.load("X_test.npy", mmap_mode='r')
y_test_flat = np.load("y_test.npy")

# --- 3. Label Encoding ---
label_encoder = LabelEncoder()
label_encoder.fit(y_train_flat)
np.save("label_classes.npy", label_encoder.classes_)
num_classes = len(label_encoder.classes_)

# --- 4. Create Generators ---
print("Setting up data generators...")
train_generator = SignLanguageGenerator(X_train_flat, y_train_flat, BATCH_SIZE, SEQUENCE_LENGTH, label_encoder)
test_generator = SignLanguageGenerator(X_test_flat, y_test_flat, BATCH_SIZE, SEQUENCE_LENGTH, label_encoder)

if len(train_generator.indices) == 0 or len(test_generator.indices) == 0:
    print("\n❌ FATAL ERROR: Not enough data to create sequences.")
    exit()

# --- 5. Build and Train the Model ---
print("\nBuilding contextual model...")
model = build_contextual_model(
    sequence_length=SEQUENCE_LENGTH,
    frame_shape=(64, 64, 3),
    num_classes=num_classes
)
model.summary()

callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1, mode='max'),
    EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode='min'),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, verbose=1, mode='min')
]

print("\n🔥 Starting model training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks
)

# --- 6. Evaluation and Visualization ---
print(f"\n✅ Training complete. Best model saved to '{MODEL_SAVE_PATH}'")
print(f"Max validation accuracy reached: {max(history.history['val_accuracy']):.4f}")

print("Generating training history plots...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("training_history.png")
print("✅ Plots saved.")