import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input  # Added for proper preprocessing
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # Added scheduler
import json
import os

# ================= GPU CHECK =================
gpus = tf.config.list_physical_devices("GPU")
print("GPUs Available:", gpus)

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU memory growth enabled")

# ================= PATHS =================
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/valid"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# ================= DATA =================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Changed from rescale for EfficientNet
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # Same for val

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes
print("✅ Classes:", NUM_CLASSES)

# ================= SAVE CLASS NAMES =================
class_indices = train_gen.class_indices
class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]

with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("✅ class_names.json saved")

# ================= MODEL =================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # Frozen for now

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(base_model.input, output)

model.compile(
    optimizer=Adam(learning_rate=0.001),  # Increased LR for better learning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= CALLBACKS =================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(
        "plant_disease_model_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,  # Weights only to avoid serialization error
        verbose=1
    ),
    ReduceLROnPlateau(  # Added to auto-adjust LR if stuck
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# ================= TRAIN =================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    # verbose=1  # For progress bar
)

# model.save("plant_disease_model_final.h5")  # comment out to avoid error
model.save_weights("plant_disease_final_weights.h5")
print("✅ Weights saved (no serialization issue)")