from tensorflow.keras.models import load_model
import numpy as np
import cv2


#Here, pre defined trained model is loaded.
# So, there is no need to create and train the model again and again
model = load_model("Mymodel.h5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Testing of the Normal Image from the covid19dataswet
image = cv2.imread("covid19dataset/test/normal/ryct.2020200028.fig1a.jpeg")
image = cv2.resize(image, (64, 64))
image = np.reshape(image, [1, 64, 64, 3])


resultantClasses = model.predict_classes(image)
#Assigning the label
labelProvided = ["COVID-19 INFECTED XRAY IMAGE", "NORMAL XRAY IMAGE"]
print(resultantClasses)
print(labelProvided[resultantClasses[0][0]])

