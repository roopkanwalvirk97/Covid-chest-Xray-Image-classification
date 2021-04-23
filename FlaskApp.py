"""
    Here Image Classification is done from the Saved Model along with Flask App

"""

from flask import *

from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Creation of Python App to run on Flask Server
app = Flask(__name__)

def predictionOfCOVID(imageToGETTested):

    model = load_model("Mymodel.h5") #Loading of model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    image = cv2.imread(imageToGETTested)
    image = cv2.resize(image, (64, 64))
    image = np.reshape(image, [1, 64, 64, 3])

    classes = model.predict_classes(image)   # prediction of the image

    label = ["COVID-19 INFECTED XRAY IMAGE", "NORMAL XRAY IMAGE"]

    return label[classes[0][0]]


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload-image', methods=['POST'])
def uploadAnImage():
    if request.method == 'POST': #  Validating whether the user is uploading the file in POST Request
        file = request.files['image']
        file.save(file.filename)

        labelProvided = predictionOfCOVID(file.filename)

        return render_template('result.html', name=labelProvided)


if __name__ == '__main__':
    #Execution of App and enabling debugging for the app
    app.run(debug=True)