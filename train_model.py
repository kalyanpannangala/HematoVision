import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ===============================
# 1️⃣ Dataset Paths (Based on your folder structure)
# ===============================

train_dir = "dataset2-master/dataset2-master/images/TRAIN"
test_dir  = "dataset2-master/dataset2-master/images/TEST"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

# ===============================
# 2️⃣ Data Preprocessing & Augmentation
# ===============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Class Indices:", train_generator.class_indices)

# ===============================
# 3️⃣ Load MobileNetV2 (Transfer Learning)
# ===============================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze base model
base_model.trainable = False

# ===============================
# 4️⃣ Add Custom Classification Layers
# ===============================

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ===============================
# 5️⃣ Compile Model
# ===============================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# 6️⃣ Callbacks (Recommended for Internship)
# ===============================

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ===============================
# 7️⃣ Train Model
# ===============================

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# ===============================
# 8️⃣ Evaluate Model
# ===============================

loss, accuracy = model.evaluate(validation_generator)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# ===============================
# 9️⃣ Save Final Model
# ===============================

model.save("Blood_Cell.h5")

print("\nModel saved successfully as Blood_Cell.h5")