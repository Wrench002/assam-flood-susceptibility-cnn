import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = r"C:\Users\admin\Desktop\CNN_Flood\Final Stack"

# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading datasets...")

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))

y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("Datasets loaded.")

# =========================================================
# ONE HOT ENCODING
# =========================================================

y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

# =========================================================
# CNN MODEL
# =========================================================

model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(15,15,5)
    ),

    MaxPooling2D((2,2)),

    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.3),

    Dense(2, activation='softmax')
])

# =========================================================
# COMPILE
# =========================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================================================
# TRAIN
# =========================================================

print("\nTraining CNN...")

history = model.fit(
    X_train,
    y_train_cat,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    verbose=1
)

# =========================================================
# EVALUATION
# =========================================================

print("\nEvaluating model...")

test_loss, test_acc = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0
)

print(f"\nTest Accuracy: {test_acc:.4f}")

# =========================================================
# PREDICTIONS
# =========================================================

y_probs = model.predict(X_test)

y_pred = np.argmax(y_probs, axis=1)

# =========================================================
# METRICS
# =========================================================

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

roc_auc = roc_auc_score(y_test, y_probs[:,1])

print(f"\nROC-AUC Score: {roc_auc:.4f}")

# =========================================================
# SAVE MODEL
# =========================================================

model.save(os.path.join(DATA_DIR, "flood_cnn_model.h5"))

print("\nModel saved successfully.")

# =========================================================
# TRAINING CURVES
# =========================================================

plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training Accuracy")

plt.legend()

plt.savefig(os.path.join(DATA_DIR, "training_accuracy.png"))

plt.close()

print("\nTraining curve saved.")