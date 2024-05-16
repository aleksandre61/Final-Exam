# task4
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.datasets import cifar10
# Load CIFAR-10 dataset
(x_train, y_train), (_, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255

# Define data augmentation for both car and gun images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load gun images with data augmentation
gun_images_dir = "gun_images"
gun_datagen = ImageDataGenerator(rescale=1./255)
gun_generator = gun_datagen.flow_from_directory(
    gun_images_dir,
    target_size=(32, 32),
    batch_size=32,  # Adjust based on your hardware limitations
    class_mode='binary'  # Binary classification (car or gun)
)

# Get a batch of data from each generator
train_data_batch, train_labels_batch = next(gun_generator)
gun_data_batch, gun_labels_batch = next(gun_generator)

# Concatenate batches along axis 0 (samples)
x_train_combined = np.concatenate([train_data_batch, gun_data_batch], axis=0)
y_train_combined = np.concatenate([train_labels_batch, gun_labels_batch])

# Shuffled indices for combined data
indices = np.arange(len(x_train_combined))
np.random.shuffle(indices)

# Shuffled combined data
x_train_combined = x_train_combined[indices]
y_train_combined = y_train_combined[indices]

# Consider exploring pre-trained models for the base (e.g., VGG16 with transfer learning)
model = models.Sequential([
    # ... (replace with pre-trained model layers and fine-tuning)
])

# Compile the model (consider additional metrics like precision, recall)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_combined, y_train_combined, epochs=5, batch_size=32)
