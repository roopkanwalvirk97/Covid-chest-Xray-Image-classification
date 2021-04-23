from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt

print("CREATION OF IMAGE CLASSIFICATION MODEL")

print()
print()

print("STEP: 1 IMAGE PRE-PROCESSING")
trainImageGenerator = ImageDataGenerator(rescale=1.0 / 255)
testImageGenerator = ImageDataGenerator(rescale=1.0 / 255)

trainImages = trainImageGenerator.flow_from_directory(
                                        'covid19dataset/train',
                                        target_size=(64, 64),
                                        batch_size=8,
                                        class_mode='binary')

testImages = testImageGenerator.flow_from_directory(
                                        'covid19dataset/test',
                                        target_size=(64, 64),
                                        batch_size=8,
                                        class_mode='binary')
def ImagePlot(images):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

sample_training_images, _ = next(trainImages)


print("STEP: 2 CREATION OF CONVOLUTIONAL NEURAL NETWORK MODEL")
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Here, we are adding the NeuralNet in ConvNet Model
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print("STEP:3 TRAINING THE MODEL")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Here, we have used fit_generator instead of fit because we have used ImageDataGenerator for our Image Data Set Creation
history = model.fit_generator(trainImages, epochs=5, validation_data=testImages)

print("STEP:4 VISUALISATION OF ACCURACY AND LOSS")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('ACCURACY GRAPH')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper left')
plt.title('LOSS GRAPH')

plt.show()

# Here, save function is used to save the model
model.save("Mymodel.h5")  # h5 format was used in various versions of keras and it is a bit previous model.
print("MODEL IS SAVED !!")
