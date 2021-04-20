import cv2 
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image 
import sys
sys.path.append("..")
from extract_bottleneck_features import *
from keras.models import load_model
import numpy as np

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os

## make static as upload folder and restrict extensions of an image
UPLOAD_FOLDER  = './static'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# Load dog names from txt
with open("../dog_names.txt", "r") as f:
    dog_names = f.read().splitlines()

## Face detector
face_cascade = cv2.CascadeClassifier('..\haarcascades\haarcascade_frontalface_alt.xml')

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

## Gog detector
ResNet50_model = ResNet50(weights='imagenet')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

## Dog breed prediction
Xception_model = load_model('../saved_models/weights.best.Xception.hdf5')

def Xception_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Xception_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def predict_dog_breed(img_path):
    '''
    Predict a dog's breed by given image
    
    INPUT: 
        img_path: An image's path
    
    OUTPUT:
        predict_breed: The breed of the dog or the human in the photo
    '''
    ## Recognize the given photo that is an human or a dog and predict its breed
    if face_detector(img_path):
        prediction = Xception_predict_breed(img_path)
        return 'This is a human, but it looks like ... {}!!'.format(prediction)
    elif dog_detector(img_path):
        prediction = Xception_predict_breed(img_path)
        return 'This is a dog, and its breed is ... {}!!'.format(prediction)
    else:
        return 'An error occurred! Cannot recognize the photo!!'

## The page to upload an image
@app.route('/')
def upload():
    return render_template('upload.html')

## Predict the dog breed from a uploaded image and display it
@app.route('/go', methods=['POST'])
def go():
    # React while uploading 
    if request.method == 'POST':
        file = request.files['file']

        # Save the image and predict the breed of it
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_dog_breed(file_path)

    # This will render the go.html Please see that file. 
    return render_template('go.html', prediction=prediction, filePath=file_path)

