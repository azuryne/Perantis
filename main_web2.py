from flask import Flask, flash, request, redirect, url_for
from flask.globals import request
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2 as cv
import tensorflow as tf 
import torch 
from torch import nn 
from torch.utils.data import DataLoader
import torch.nn.functional as F

app = Flask(__name__)

UPLOAD_FOLDER = r'/Users/azureennaja/Desktop/Perantis/cv-master/notes/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

save_mode = 0  # 1001


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(img, originalFileName):
    filePath = os.path.join(app.config['UPLOAD_FOLDER'], originalFileName)
    cv.imwrite(filePath, img)
    originalUri = originalFileName
    return originalUri

# /
@app.route('/blood_tf_cnn', methods=['GET', 'POST'])
def blood_tf_cnn():

    # /?originalUri=xxxxxx&resizedUri=yyyyy
    originalUri = '' if request.args.get(
        'originalUri') is None else request.args.get('originalUri')
    predictedLabel = '' if request.args.get(
        'predictedLabel') is None else request.args.get('predictedLabel')
    error = '' if request.args.get(
        'error') is None else request.args.get('error')


    if request.method == 'POST':

        # read the uploaded image
        # check 
        if 'imgFile' not in request.files:
            error = 'No file part'
            return redirect(url_for('face', error = error))

        file = request.files['imgFile']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            error = 'No selected file'
            return redirect(url_for('face', error = error))
        
        # good file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # fix file extension 
            root, ext = os.path.splitext(filename)
            ext = ext.lower()
            ext = '.jpeg' if ext == '.jpg' else ext

            # get raw file 
            f = file.read()

            # convert to numpy array 1D
            imgArray = np.frombuffer(f, np.uint8)

            # create image by converting the 1D array
            img = cv.imdecode(imgArray, cv.IMREAD_COLOR)

            # save the original image 
            # get file path to save
            originalFileName = 'original' + ext
            originalUri = save_file(img, originalFileName)

            classNames  = ['basophil', 'eusinophil', 'erythroblast', 'ig', 'lymphocyte',
                            'monocyte', 'neutrophil', 'platelet'] 


            imgRGB = img[:,:,::-1]

            N = len(imgRGB)
            shape = (1, 200, 200, 3)
            y = np.empty(shape)
            y[0] = cv.resize(imgRGB, [200,200], interpolation=cv.INTER_NEAREST)

            testImages = y / 255.0
            

            # load model 
            # load SavedModel format

            exportPath = 'tf_model_3/tf_90'
            model = tf.keras.models.load_model(exportPath)


            # predict
            predictions = model.predict(testImages)
            predictions[0]

            i = np.argmax(predictions[0])
            predictedLabel = classNames[i]


        # redirect to GET and display image
        return redirect(url_for('blood_tf_cnn', originalUri = originalUri, predictedLabel = predictedLabel))

        

    return f'''
    <!doctype html>
    <title>Computer Vision </title>
    <h1>Upload Image</h1>
     <div style = "color : red">
      {error}
    </div>
    <form method=post enctype=multipart/form-data>
      <input type=file name=imgFile>
      <input type=submit value=Upload>      
    </form>
    <div>
     <h2>Tensorflow CNN</h2>
    </div>
    <div>
     <h3>Prediction</h3>
      <p style="color: blue">{predictedLabel}</p>
      <img src="static/uploads/{originalUri}" />
    </div>
    '''

# /
@app.route('/blood_tf', methods=['GET', 'POST'])
def blood_tf():

    # /?originalUri=xxxxxx&resizedUri=yyyyy
    originalUri = '' if request.args.get(
        'originalUri') is None else request.args.get('originalUri')
    predictedLabel = '' if request.args.get(
        'predictedLabel') is None else request.args.get('predictedLabel')
    error = '' if request.args.get(
        'error') is None else request.args.get('error')


    if request.method == 'POST':

        # read the uploaded image
        # check 
        if 'imgFile' not in request.files:
            error = 'No file part'
            return redirect(url_for('face', error = error))

        file = request.files['imgFile']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            error = 'No selected file'
            return redirect(url_for('face', error = error))
        
        # good file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # fix file extension 
            root, ext = os.path.splitext(filename)
            ext = ext.lower()
            ext = '.jpeg' if ext == '.jpg' else ext

            # get raw file 
            f = file.read()

            # convert to numpy array 1D
            imgArray = np.frombuffer(f, np.uint8)

            # create image by converting the 1D array
            img = cv.imdecode(imgArray, cv.IMREAD_COLOR)

            # save the original image 
            # get file path to save
            originalFileName = 'original' + ext
            originalUri = save_file(img, originalFileName)

            classNames  = ['basophil', 'eusinophil', 'erythroblast', 'ig', 'lymphocyte',
                            'monocyte', 'neutrophil', 'platelet'] 

            # prepare test image
            imgRGB = img[:,:,::-1]

            N = len(imgRGB)
            shape = (1, 200, 200, 3)
            y = np.empty(shape)
            y[0] = cv.resize(imgRGB, [200,200], interpolation=cv.INTER_NEAREST)

            testImages = y / 255.0
            

            # load model 
            # load SavedModel format

            exportPath = 'tf_model_4/tf_68'
            model = tf.keras.models.load_model(exportPath)


            # predict
            probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
            predictions = probability_model.predict(testImages)
            predictions[0]

            i = np.argmax(predictions[0])
            predictedLabel = classNames[i]


        # redirect to GET and display image
        return redirect(url_for('blood_tf', originalUri = originalUri, predictedLabel = predictedLabel))

        

    return f'''
    <!doctype html>
    <title>Computer Vision </title>
    <h1>Upload Image</h1>
     <div style = "color : red">
      {error}
    </div>
    <form method=post enctype=multipart/form-data>
      <input type=file name=imgFile>
      <input type=submit value=Upload>      
    </form>
    <div>
    <div>
     <h2>Tensorflow</h2>
    </div>
     <h3>Prediction</h3>
      <p style="color: blue">{predictedLabel}</p>
      <img src="static/uploads/{originalUri}" />
    </div>
    '''

# /
@app.route('/blood_pt', methods=['GET', 'POST'])
def blood_pt():

    # /?originalUri=xxxxxx&resizedUri=yyyyy
    originalUri = '' if request.args.get(
        'originalUri') is None else request.args.get('originalUri')
    predictedLabel = '' if request.args.get(
        'predictedLabel') is None else request.args.get('predictedLabel')
    error = '' if request.args.get(
        'error') is None else request.args.get('error')


    if request.method == 'POST':

        # read the uploaded image
        # check 
        if 'imgFile' not in request.files:
            error = 'No file part'
            return redirect(url_for('face', error = error))

        file = request.files['imgFile']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            error = 'No selected file'
            return redirect(url_for('face', error = error))
        
        # good file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # fix file extension 
            root, ext = os.path.splitext(filename)
            ext = ext.lower()
            ext = '.jpeg' if ext == '.jpg' else ext

            # get raw file 
            f = file.read()

            # convert to numpy array 1D
            imgArray = np.frombuffer(f, np.uint8)

            # create image by converting the 1D array
            img = cv.imdecode(imgArray, cv.IMREAD_COLOR)

            # save the original image 
            # get file path to save
            originalFileName = 'original' + ext
            originalUri = save_file(img, originalFileName)

            classNames  = ['basophil', 'eusinophil', 'erythroblast', 'ig', 'lymphocyte',
                            'monocyte', 'neutrophil', 'platelet'] 

            # prepare test image
            imgRGB = img[:,:,::-1]

            N = len(imgRGB)
            shape = (1, 200, 200, 3)
            y = np.empty(shape)
            y[0] = cv.resize(imgRGB, [200,200], interpolation=cv.INTER_NEAREST)

            testImages = y / 255.0

            # convert to pytorch
            testImages = torch.Tensor(testImages)

            testLabels = os.path.splitext(filename)[0]

            all_data = []

            for i in range(len(testImages)):
                all_data.append([testImages[i], testLabels[i]])

            #testLabels = testLabels.astype('int64')

            device = "cpu"
            input_features = 3*200*200

            class NeuralNetwork(nn.Module):

                def __init__(self):
                    super(NeuralNetwork, self).__init__()
                    self.flatten = nn.Flatten()
                    self.linear_relu_stack = nn.Sequential(
                    nn.Linear(input_features, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 8)
                    )

                def forward(self, x):
                    x = self.flatten(x)
                    logits = self.linear_relu_stack(x)
                    return logits

            model = NeuralNetwork().to(device)
            model.load_state_dict(torch.load("pth_500_81.pth"))

            model.eval()
            x = [testImages][0]
            x = x.view(1, -1)
            with torch.no_grad():
                pred = model(x)
                predictedLabel = classNames[pred[0].argmax(0).item()]


        # redirect to GET and display image
        return redirect(url_for('blood_pt', originalUri = originalUri, predictedLabel = predictedLabel))

        

    return f'''
    <!doctype html>
    <title>Computer Vision </title>
    <h1>Upload Image</h1>
     <div style = "color : red">
      {error}
    </div>
    <form method=post enctype=multipart/form-data>
      <input type=file name=imgFile>
      <input type=submit value=Upload>      
    </form>
    <div>
     <h2>Pytorch</h2>
    </div>
    <div>
     <h3>Prediction</h3>
      <p style="color: blue">{predictedLabel}</p>
      <img src="static/uploads/{originalUri}" />
    </div>
    '''

# /
@app.route('/blood_pt_cnn', methods=['GET', 'POST'])
def blood_pt_cnn():

    # /?originalUri=xxxxxx&resizedUri=yyyyy
    originalUri = '' if request.args.get(
        'originalUri') is None else request.args.get('originalUri')
    predictedLabel = '' if request.args.get(
        'predictedLabel') is None else request.args.get('predictedLabel')
    error = '' if request.args.get(
        'error') is None else request.args.get('error')


    if request.method == 'POST':

        # read the uploaded image
        # check 
        if 'imgFile' not in request.files:
            error = 'No file part'
            return redirect(url_for('face', error = error))

        file = request.files['imgFile']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            error = 'No selected file'
            return redirect(url_for('face', error = error))
        
        # good file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # fix file extension 
            root, ext = os.path.splitext(filename)
            ext = ext.lower()
            ext = '.jpeg' if ext == '.jpg' else ext

            # get raw file 
            f = file.read()

            # convert to numpy array 1D
            imgArray = np.frombuffer(f, np.uint8)

            # create image by converting the 1D array
            img = cv.imdecode(imgArray, cv.IMREAD_COLOR)

            # save the original image 
            # get file path to save
            originalFileName = 'original' + ext
            originalUri = save_file(img, originalFileName)

            classNames  = ['basophil', 'eusinophil', 'erythroblast', 'ig', 'lymphocyte',
                            'monocyte', 'neutrophil', 'platelet'] 

            # prepare test image
            imgRGB = img[:,:,::-1]

            N = len(imgRGB)
            shape = (1, 200, 200, 3)
            y = np.empty(shape)
            y[0] = cv.resize(imgRGB, [200,200], interpolation=cv.INTER_NEAREST)

            testImages = y / 255.0

            testImages = testImages.reshape(3, 200, 200)

            # convert to pytorch
            testImages = torch.Tensor(testImages)

            testLabels = os.path.splitext(filename)[0]

            all_data = []

            for i in range(len(testImages)):
                all_data.append([testImages[i], testLabels[i]])

            #testLabels = testLabels.astype('int64')

            device = "cpu"

            class NeuralNetwork(nn.Module):
                def __init__(self):
                    super(NeuralNetwork, self).__init__()
                    self.conv1 = nn.Conv2d(3,32,3,1)
                    self.conv2 = nn.Conv2d(32,64,3,1)
                    self.fc1 = nn.Linear(614656, 64)
                    self.fc2 = nn.Linear(64, 8)
                    
                def forward(self,x):
                    x = self.conv1(x)
                    x = F.relu(x)
                    x = self.conv2(x)
                    x = F.relu(x)
                    x = F.max_pool2d(x,2)
                    x = torch.flatten(x, 1)
                    x = self.fc1(x)
                    x = F.relu(x)
                    x = self.fc2(x)
                    output = F.log_softmax(x, dim=1)

                    return output

            model = NeuralNetwork().to(device)

            model = NeuralNetwork().to(device)
            model.load_state_dict(torch.load("pth_500cnn2_40.pth"))

            model.eval()
            x = [testImages][0]
            x = x.unsqueeze(0) #add dimension at position zero 

            with torch.no_grad():
                pred = model(x)
                predictedLabel = classNames[pred[0].argmax(0).item()]


        # redirect to GET and display image
        return redirect(url_for('blood_pt_cnn', originalUri = originalUri, predictedLabel = predictedLabel))

        

    return f'''
    <!doctype html>
    <title>Computer Vision </title>
    <h1>Upload Image</h1>
     <div style = "color : red">
      {error}
    </div>
    <form method=post enctype=multipart/form-data>
      <input type=file name=imgFile>
      <input type=submit value=Upload>      
    </form>
    <div>
     <h2>Pytorch CNN</h2>
    </div>
    <div>
     <h3>Prediction</h3>
      <p style="color: blue">{predictedLabel}</p>
      <img src="static/uploads/{originalUri}" />
    </div>
    '''
    

if __name__ == "__main__":
    app.run(host="0.0.0.0")