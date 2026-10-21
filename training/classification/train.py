import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import os

# Parameters
image_size = (224, 224)  # Resize images
batch_size = 32

# Set dataset path
dataset_path = 'E:/Projects/Cancer Detection/classification'

# Image Data Generators for training and validation
datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values to [0, 1]
    validation_split=0.2  # 80-20 train-validation split
)

# Train and validation data generators
train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load pre-trained ResNet50 model without the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the pre-trained model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # 3 classes: colon, lung, skin

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base layers (pre-trained weights)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# Fine-tune the model by unfreezing some of the base model layers
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Re-compile the model for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training (fine-tuning)
history_finetune = model.fit(
    train_gen,
    epochs=5,
    validation_data=val_gen
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Save the trained model
model.save('cancer_classifier.h5')

print("Model training complete and saved.")
