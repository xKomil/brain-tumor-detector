import os
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,  MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_augmentation import train_generator, test_generator

num_classes = 4 

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Dodanie warstwy max pooling po modelu bazowym
x = base_model.output
x = MaxPooling2D(pool_size=(2, 2))(x)  # Dodaj warstwę max pooling
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) 
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    epochs=10,
    callbacks=[early_stopping, reduce_lr]
)

for layer in base_model.layers:
    layer.trainable = True

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    epochs=10,
    callbacks=[early_stopping, reduce_lr] 
)

loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

model.save('brain_tumor_model.h5')

print(f"Test accuracy: {accuracy * 100:.2f}%")
